#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 18:32:56 2022

@author: mbarbato
"""

import torch
import torch.nn.functional as F
from torch import nn, utils, Tensor
import numpy as np

class multisourceRemoteSensingUNet(nn.Module):
    def encoding_block(self, in_channels, out_channels, kernel_size=3):
         block = torch.nn.Sequential(
             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=((kernel_size-1)//2)),
             torch.nn.ReLU(),
             torch.nn.BatchNorm2d(out_channels),
             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=((kernel_size-1)//2)),
             torch.nn.ReLU(),
             torch.nn.BatchNorm2d(out_channels),
             )
         return block
     
    def decoding_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
         block = torch.nn.Sequential(
             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=((kernel_size-1)//2)),
             torch.nn.ReLU(),
             torch.nn.BatchNorm2d(mid_channel),
             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=((kernel_size-1)//2)),
             torch.nn.ReLU(),
             torch.nn.BatchNorm2d(mid_channel),
             # torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
             torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
             )        
         return block

    def output_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
         block = torch.nn.Sequential(
             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=((kernel_size-1)//2)),
             torch.nn.ReLU(),
             torch.nn.BatchNorm2d(mid_channel),
             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=((kernel_size-1)//2)),
             torch.nn.ReLU(),
             torch.nn.BatchNorm2d(mid_channel),
             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=((kernel_size-1)//2)),
             torch.nn.ReLU(),
             torch.nn.BatchNorm2d(out_channels),
             )
         return block
     
    def extract_source_features_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
         block = torch.nn.Sequential(
             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=((kernel_size-1)//2)),
             torch.nn.ReLU(),
             torch.nn.BatchNorm2d(mid_channel),
             torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=((kernel_size-1)//2)),
             torch.nn.ReLU(),
             torch.nn.BatchNorm2d(out_channels),
             )
         return block
     
    def bottleneck_block(self, in_channels, mid_channels, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=((kernel_size-1)//2)),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(mid_channels),
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=out_channels, padding=((kernel_size-1)//2)),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(out_channels),

                # torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
                torch.nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, output_padding=0) # Upsampling
                )
        return block
     
        
    def __init__(self, in_channels_swir, in_channels_vnir, out_channels): #to define dimensions of each input
        super(multisourceRemoteSensingUNet, self).__init__()
        # super().__init__()
        
        # ============================ Architecture ===========================
        
        self.extract_rgb_1 = self.extract_source_features_block(3, mid_channel=8, out_channels=16, kernel_size=7)
        # self.extract_rgb_2 = self.extract_source_features_block(16, mid_channel=32, out_channels=64, kernel_size=3)
        
        self.extract_pan_1 = self.extract_source_features_block(1, mid_channel=8, out_channels=16, kernel_size=7)
        # self.extract_pan_2 = self.extract_source_features_block(16, mid_channel=32, out_channels=64, kernel_size=3)
        
        # self.extract_swir_1 = self.extract_source_features_block(in_channels=in_channels_swir, mid_channel=128, out_channels=64, kernel_size=7)
        # self.extract_swir_2 = self.extract_source_features_block(64, mid_channel=32, out_channels=16, kernel_size=3)
        self.extract_swir_1 = self.extract_source_features_block(in_channels=in_channels_swir, mid_channel=62, out_channels=16, kernel_size=7)
        
        self.extract_vnir_1 = self.extract_source_features_block(in_channels=in_channels_vnir, mid_channel=32, out_channels=16, kernel_size=7)
        # self.extract_vnir_2 = self.extract_source_features_block(16, mid_channel=32, out_channels=64, kernel_size=3)
        
        self.extract_dem_1 = self.extract_source_features_block(1, mid_channel=8, out_channels=16, kernel_size=7)
        # self.extract_dem_2 = self.extract_source_features_block(16, mid_channel=32, out_channels=64, kernel_size=3)
        

        # self.encode_1 =self.encoding_block(in_channels=80, out_channels=256) # 16*5 = 80
        # self.conv_maxpool_encode_1 = torch.nn.MaxPool2d(kernel_size=2)
        # self.encode_2 =self.encoding_block(in_channels=256, out_channels=512)
        # self.conv_maxpool_encode_2 = torch.nn.MaxPool2d(kernel_size=2)
        
        # # Bottleneck
        # self.bottleneck = self.bottleneck_block(in_channels=512,
        #                                         mid_channels=1024,
        #                                         out_channels=512,
        #                                         kernel_size=3)
    
        # # Decode
        # self.decode_2 = self.decoding_block(1024, 512, 256) # 512*2
        # self.segmetation_output = self.output_block(512, 80, out_channels) # 256*2
        
        self.encode_1 =self.encoding_block(in_channels=80, out_channels=128) # 16*5 = 80
        self.conv_maxpool_encode_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.encode_2 =self.encoding_block(in_channels=128, out_channels=256)
        self.conv_maxpool_encode_2 = torch.nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.bottleneck = self.bottleneck_block(in_channels=256,
                                                mid_channels=512,
                                                out_channels=256,
                                                kernel_size=3)
    
        # Decode
        self.decode_2 = self.decoding_block(512, 256, 128) # 512*2
        self.segmetation_output = self.output_block(256, 80, out_channels) # 256*2
        
    
    def forward(self, t_rgb, t_pan, t_swir, t_vnir, t_dem):
        
        # =========================== Multi-sources ===========================

        t_rgb_features = self.extract_rgb_1(t_rgb)
        t_pan_features = self.extract_pan_1(t_pan)
        t_vnir_features = self.extract_vnir_1(t_vnir)
        t_swir_features = self.extract_swir_1(t_swir)
        # t_swir_features = self.extract_swir_2(self.extract_swir_1(t_swir))
        t_dem_features = self.extract_dem_1(t_dem)
        
        t_fusion = torch.cat((t_rgb_features,
                              t_pan_features,
                              t_vnir_features,
                              t_swir_features,
                              t_dem_features), dim=1)
        
        # ============================== Encoder ==============================

        # First layer        
        t_encode_1 = self.encode_1(t_fusion)
        t_encode_pool_1 = self.conv_maxpool_encode_1(t_encode_1)
        
        # Second layer
        t_encode_2 = self.encode_2(t_encode_pool_1)       
        t_encode_pool_2 = self.conv_maxpool_encode_2(t_encode_2)

        # ============================ Bottleneck =============================

        t_bottleneck = self.bottleneck(t_encode_pool_2)

        # ============================== Decoder ==============================

        # Decode
        # decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        
        cat_feat_2 = torch.cat((t_encode_2, t_bottleneck), dim=1)
        t_decode_1 = self.decode_2(cat_feat_2)
        cat_feat_1 = torch.cat((t_encode_1, t_decode_1), dim=1)
        output = self.segmetation_output(cat_feat_1) # decode 1 + output
        
        return  output
     