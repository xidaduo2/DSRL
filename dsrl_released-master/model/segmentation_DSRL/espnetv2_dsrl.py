# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch.nn import init
from nn_layers.espnet_utils import *
from nn_layers.efficient_pyramid_pool import EfficientPyrPool
from nn_layers.efficient_pt import EfficientPWConv
from model.classification.espnetv2 import EESPNet
from torch.nn import functional as F


def Deconv_BN_ACT(in_plane, out_plane):
    conv_trans = nn.ConvTranspose2d(in_plane, out_plane ,kernel_size=3, stride=2,padding=1, output_padding=1, bias=False)
    norm =  nn.BatchNorm2d(out_plane)
    act = nn.PReLU(out_plane)
    return nn.Sequential(conv_trans, norm, act)

class ESPNetv2Segmentation(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the Semantic Segmenation
    '''

    def __init__(self, args, classes=21, dataset='pascal'):
        super().__init__()

        # =============================================================
        #                       BASE NETWORK
        # =============================================================

        #   same as Line27-31 in ESPNetv2, except the dec_planes
        
        base_dec_planes = dec_feat_dict[dataset]
        dec_planes = [4*base_dec_planes, 3*base_dec_planes, 2*base_dec_planes, classes]
        #=============================================================
        #                   SEGMENTATION BASE NETWORK
        #=============================================================
        #
        #  same as Line36-66 in ESPNetv2      
        #   
        #=============================================================
        #                   SSSR BRANCH
        #=============================================================           
        self.bu_dec_l5 = Deconv_BN_ACT(dec_planes[3],dec_planes[3])
        self.bu_dec_l6 = Deconv_BN_ACT(dec_planes[3],dec_planes[3])

        self.init_params()

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def init_params(self):
        '''
        Function to initialze the parameters
        '''

    def get_basenet_params(self):


    def get_segment_params(self):


    def forward(self, x):

        x_size = (x.size(2), x.size(3))

        ###### this model outputs: enc_out_l4

        # bottom-up decoding
        ###### this model outputs: bu_out0
        
        # Decoding block

        ###### this model outputs: bu_out1
        
        #decoding block
        ###### this model outputs: bu_out2
        
        # decoding block
        ###### this model outputs: bu_out
        
        # sssr block
        
        sssr_1 = self.bu_dec_l5(bu_out)
        sssr_out = self.b
        u_dec_l6(sssr_1)

        return  sssr_out 

def espnetv2_seg(args):
    classes = args.classes
    scale=args.s
    dataset=args.dataset
    model = ESPNetv2Segmentation(args, classes=classes, dataset=dataset)
    return model



