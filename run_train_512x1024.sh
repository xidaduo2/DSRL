

#Configs
DATA_PATH=D:/xidaduo/DATASETS/CityScapes
OUTPUT_PATH=train_espnetv2_dsrl_512x1024_output
WEIGHT_FILE=ckpt-segmentation/espnetv2_dsrl/512x1024/espnetv2_2.0_2048_best.pth
#
#echo "========Step1: Perform Testing========"
CUDA_VISIBLE_DEVICES=0 python train.py --model espnetv2_dsrl --s 2.0 --dataset cityscapes --data-path ${DATA_PATH} --savedir ${OUTPUT_PATH} --ckpt-file ${WEIGHT_FILE}

