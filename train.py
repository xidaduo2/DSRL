import argparse
import os
import copy
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model.segmentation_DSRL.espnetv2_dsrl_v1 import ESPNetv2Segmentation
from model.segmentation_DSRL.espnetv2_dsrl_v1 import espnetv2_seg
# from utils import AverageMeter, calc_psnr
from utils.metrics import AverageMeter
from dataloaders import make_data_loader
from utils.lr_scheduler import LR_Scheduler
from utils.fa_loss import FALoss
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, default='output')  # , required=True)
    parser.add_argument('--lr', type=float, default=1e-2)   # 1e-4
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--tensorboard-log-name', type=str, default="_test_dsrl")
    parser.add_argument('--experiment-dir', type=str, default="experiment_dir_test_dsrl")
    parser.add_argument('--dataset', type=str, default="cityscapes")
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--num-classes', type=int, default=19)
    parser.add_argument('--net-name', type=str, default="psp")
    parser.add_argument('--omaga-fa', type=float, default=1.0)
    parser.add_argument('--omaga-sr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--base-size', type=int, default=1024)
    parser.add_argument('--crop-size', type=int, default=1024)
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')

    # mdoel details
    parser.add_argument('--cross-os', type=float, default=2.0, help='Factor by which feature for cross')
    parser.add_argument('--model', default="espnetv2_dsrl", choices=['espnetv2_dsrl', 'espnetv2'], help='Model name')
    parser.add_argument('--ckpt-file', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--data-path', default="", help='Data directory')
    parser.add_argument('--dataset', default='city', help='Dataset name')
    parser.add_argument('--savedir', default="result", help='save prediction directory')
    # input details
    parser.add_argument('--im-size', type=int, nargs="+", default=[512, 256], help='Image size for testing (W x H)')
    # parser.add_argument('--split', default='train', choices=['train', 'val', 'test'], help='data split')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')

    args = parser.parse_args()


    # args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    summary = TensorboardSummary(args.experiment_dir)
    writer = summary.create_summary()
    evaluator = Evaluator(args.num_classes)
    # writer = SummaryWriter(comment=args.tensorboard_log_name)
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader, nclass = make_data_loader(args)

    # DSRL (semantic segmentation net guide super-resolution net
    # model = DSRL(in_ch=3, out_ch=args.num_classes).to(device)
    # ===================== load model ============================
    args.classes = 19+1    # cityscapes
    model = espnetv2_seg(args)
    model = model.to(device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))



    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_pred = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = AverageMeter()
        train_dice_PGU_epoch = AverageMeter()

        # with tqdm(total=(len(train_loader) - len(train_loader) % args.batch_size)) as t:
        t = tqdm(val_loader, desc='\r')
        t.set_description('epoch: {}/{}'.format(epoch, args.epochs - 1))

        for i, sample in enumerate(t):
            image, target = sample['image'], sample['label'].long().to(device)
            input_img = torch.nn.functional.interpolate(image, size=[i // 2 for i in image.size()[2:]], mode='bilinear', align_corners=True)
            input_img = input_img.to(device)

            # x_seg_up=sssr_decoder=output   x_sr_up=sisr_decoder_feature=output_sr
            # fea_sr=sssr_decoder_transformed=fea_seg  x_sr_up=sisr_decoder_feature
            # output, output_sr, fea_seg, fea_sr = model(input_img)
            output, fea_seg, fea_sr, output_sr = model(input_img)

            loss = criterion(output, target)
            # print(loss)
            loss += args.omaga_sr * torch.nn.MSELoss()(output_sr, image.to(device))
            loss += args.omaga_fa * FALoss()(fea_seg, fea_sr)

            epoch_losses.update(loss.item(), len(image))

            scheduler(optimizer, i, epoch, best_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            # t.set_postfix({'dice':train_dice_PGU_epoch.avg, 'loss':epoch_losses.avg})
            t.update(len(image))

            # Show 10 * 3 inference results each epoch
            # if i % (len(train_loader) // 10) == 0:
            #     global_step = i + len(train_loader) * epoch
            #     summary.visualize_image(writer, args.dataset, image, target, output, global_step)

        # torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_PGU_dice = AverageMeter()
        eval_acc = AverageMeter()
        eval_miou = AverageMeter()

        tbar = tqdm(val_loader, desc='\r')
        tbar.set_description('val epoch: {}/{}'.format(epoch, args.epochs - 1))
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label'].long().to(device)
            input_img = torch.nn.functional.interpolate(image, size=[i // 2 for i in image.size()[2:]], mode='bilinear', align_corners=True)
            input_img = input_img.to(device)

            with torch.no_grad():
                output, _, _, _ = model(input_img)
            loss = criterion(output, target)

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        writer.add_scalar("train/loss", epoch_losses.average, epoch)
        writer.add_scalar('val/mIoU', mIoU, epoch)
        writer.add_scalar('val/Acc', Acc, epoch)
        writer.add_scalar('val/Acc_class', Acc_class, epoch)
        writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

        new_pred = mIoU
        if new_pred > best_pred:
            is_best = True
            best_pred = new_pred
            best_epoch = epoch
            # saver.save_checkpoint({
            #     'epoch': epoch + 1,
            #     'state_dict': model.module.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'best_pred': best_pred,
            # }, is_best)
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, mIoU: {:.2f}'.format(best_epoch, best_pred))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
