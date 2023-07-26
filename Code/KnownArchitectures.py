# import os,sys,time,math
import json
import argparse
# import PIL
# from PIL import Image
# from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import utils # transforms, models, utils
# import torchvision.datasets as dsetre
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from einops import rearrange, repeat # , reduce

import segmentation_models_pytorch as smp
# import my_segmentation_models as msmp
from Base import Base
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import pickle

from multisourceRemoteSensingUNet_model import multisourceRemoteSensingUNet
from middle_fusion_rgb_hs import Middle_fusion_en as mf_rgb_hs
from middle_fusion_rgb_hs_dem import Middle_fusion_en as mf_rgb_hs_dem

# seed
import random
from pytorch_lightning import seed_everything

from late_fusion_model import Unet_late_fusion

class KnownArchitectures(Base):
    def __init__(self, params):
        # init base
        super(KnownArchitectures, self).__init__(params)

        # early fusion
        if self.conf['method'] == 'early_fusion':

            # # define architecture
            # self.net = smp.Unet(
            #     encoder_name=self.conf['encoder_name'],
            #     encoder_weights= (self.conf['encoder_weights'] if self.conf['encoder_weights'] != "None" else None),
            #     in_channels=self.conf['input_channels'],
            #     classes=self.conf['n_classes_landuse'] + self.conf['n_classes_agricolture'],
            # )

            # torch.nn.init.xavier_uniform_(self.net.encoder.conv1.weight) # reinitialize first layer

            # no pca but pansharpened
            if self.conf['input_channels']==3:
                input_channels = 3
            if self.conf['input_channels']==4:
                input_channels = 182
            elif self.conf['input_channels']==7:
                input_channels = 3+182
            elif self.conf['input_channels']==8:
                input_channels = 3+182+1

            # define architecture
            self.net = smp.Unet(
                encoder_name=self.conf['encoder_name'],
                encoder_weights= (self.conf['encoder_weights'] if self.conf['encoder_weights'] != "None" else None),
                in_channels=input_channels,
                classes=self.conf['n_classes_landuse'] + self.conf['n_classes_agricolture'],
            )

            torch.nn.init.xavier_uniform_(self.net.encoder.conv1.weight) # reinitialize first layer

        # middle fusion
        elif self.conf['method'] == 'middle_fusion':
            # fusion encoders
            if self.conf['input_channels'] == 7:
                # self.fusion_en = mf_rgb_hs(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                #                             conf_hs={'channels':[4,16,32,64], 'kernels':[3,3,3]})
                # in_channels_middle_fusion = 64+64

                self.fusion_en = mf_rgb_hs(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                                            conf_hs={'channels':[182,128,64], 'kernels':[3,3]})
                in_channels_middle_fusion = 64+64
            elif self.conf['input_channels'] == 8:
                # self.fusion_en = mf_rgb_hs_dem(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                #                                 conf_hs={'channels':[4,16,32,64], 'kernels':[3,3,3]},
                #                                 conf_dem={'channels':[1,16,32,64], 'kernels':[3,3,3]})

                self.fusion_en = mf_rgb_hs_dem(conf_rgb={'channels':[3,16,32,64], 'kernels':[3,3,3]},
                                                conf_hs={'channels':[182,128,64], 'kernels':[3,3]},
                                                conf_dem={'channels':[1,16,32,64], 'kernels':[3,3,3]})

                in_channels_middle_fusion = 64+64+64

            # define architecture
            self.net = smp.Unet(
                encoder_name=self.conf['encoder_name'],
                encoder_weights= (self.conf['encoder_weights'] if self.conf['encoder_weights'] != "None" else None),
                in_channels=in_channels_middle_fusion,
                classes=self.conf['n_classes_landuse'] + self.conf['n_classes_agricolture'],
            )

            torch.nn.init.xavier_uniform_(self.net.encoder.conv1.weight) # reinitialize first layer

        # late fusion
        elif self.conf['method'] == 'late_fusion':

            if self.conf['input_channels']==3:
                input_channels = 3
            elif self.conf['input_channels']==7:
                input_channels = 3+182
            elif self.conf['input_channels']==8:
                input_channels = 3+182+1

            # define architecture
            self.net = Unet_late_fusion( # check!!
                encoder_name=self.conf['encoder_name'],
                encoder_weights= (self.conf['encoder_weights'] if self.conf['encoder_weights'] != "None" else None),
                in_channels=input_channels,
                classes=self.conf['n_classes_landuse'] + self.conf['n_classes_agricolture'],
            )

        # Custom network
        # self.net = multisourceRemoteSensingUNet(in_channels_swir=123, in_channels_vnir=63, out_channels=self.conf['n_classes_landuse'] + self.conf['n_classes_agricolture'])

        self.mean_dict = self.load_dict(self.conf['mean_dict_01'])
        self.std_dict = self.load_dict(self.conf['std_dict_01'])
        self.max_dict = self.load_dict(self.conf['max_dict_01'])
        self.loaded_min_dict_before_normalization = self.load_dict(self.conf['min_dict'])
        self.loaded_max_dict_before_normalization = self.load_dict(self.conf['max_dict'])
        # self.mse = self.load_dict("/home/mbarbato/Downloads/mse_metrics.pkl")

        # self.conf['pca'] = False

        # print('break')

    def load_dict(self, name):
        with open(name, 'rb') as f:
            loaded_dict = pickle.load(f)

        return loaded_dict
        
    # # Original forward
    def forward(self, batch):

        if not(self.conf['pca']):
            rgb, pan, nir, swir, dem = batch

            inp = torch.cat([rgb, dem], axis=1)

            return self.net(inp)
        else:
            rgb, hs, dem = batch

            # early fusion
            if self.conf['method'] == 'early_fusion':
                if self.conf['input_channels'] == 3:
                    inp = rgb
                elif self.conf['input_channels'] == 4:
                    inp = hs
                elif self.conf['input_channels'] == 7:
                    inp = torch.cat([rgb, hs], axis=1)
                elif self.conf['input_channels'] == 8:
                    inp = torch.cat([rgb, hs, dem], axis=1)

                # apply
                return self.net(inp)
            
            # middle fusion
            elif self.conf['method'] == 'middle_fusion':
                if self.conf['input_channels'] == 7:
                    inp = rgb, hs
                elif self.conf['input_channels'] == 8:
                    inp = rgb, hs, dem

                inp = self.fusion_en(inp)

                return self.net(inp)

            # late fusion
            elif self.conf['method'] == 'late_fusion':
                if self.conf['input_channels'] == 7:
                    inp = rgb, hs
                elif self.conf['input_channels'] == 8:
                    inp = rgb, hs, dem

                return self.net(inp)


    def create_transform_function(self, transform_list):
        # create function
        def transform_inputs(inps):
            # create transformation

            if not(self.conf['pca']):
                # split inputs
                rgb, pan, vnir, swir, dem, gt_lu, gt_ag = inps

                normalize_rgb, normalize_pan, normalize_vnir, normalize_swir, normalize_dem, transforms_augmentation = transform_list

                swir_indexes2keep = np.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                                13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                                                26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
                                                48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                                                60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
                                                73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83, 102, 103,
                                                104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                                                120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
                                                133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                                                146, 147, 148, 149, 150, 151]) # band 39 added to the selection based on corrupted pixels because of streak effect
                                                            
                vnir_indexes2keep = np.arange(60) # remove overlapping
                
                # ipdb.set_trace()
                transforms = A.Compose([transforms_augmentation],
                                        additional_targets={'pan': 'image',
                                                            'swir': 'image',
                                                            'vnir': 'image',
                                                            'dem': 'image',
                                                            'gt_ag': 'mask'})
                
                rgb = (rgb.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['rgb']) / (self.loaded_max_dict_before_normalization['rgb'] - self.loaded_min_dict_before_normalization['rgb'])
                pan = (pan.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['pan']) / (self.loaded_max_dict_before_normalization['pan'] - self.loaded_min_dict_before_normalization['pan'])
                
                vnir = vnir.permute(1,2,0).numpy()[:,:,vnir_indexes2keep]
                vnir = (vnir - self.loaded_min_dict_before_normalization['vnir']) / (self.loaded_max_dict_before_normalization['vnir'] - self.loaded_min_dict_before_normalization['vnir'])
                
                swir = swir.permute(1,2,0).numpy()[:,:,swir_indexes2keep]
                swir = (swir - self.loaded_min_dict_before_normalization['swir']) / (self.loaded_max_dict_before_normalization['swir'] - self.loaded_min_dict_before_normalization['swir'])

                dem = (dem.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['dem']) / (self.loaded_max_dict_before_normalization['dem'] - self.loaded_min_dict_before_normalization['dem'])

                rgb = normalize_rgb(image=rgb)['image']
                pan = normalize_pan(image=pan)['image']
                vnir = normalize_vnir(image=vnir)['image']
                swir = normalize_swir(image=swir)['image']
                dem = normalize_dem(image=dem)['image']


                sample = transforms(image=rgb,
                                    mask=gt_lu.permute(1,2,0).numpy(),
                                    pan=pan,
                                    vnir=vnir,
                                    swir=swir,
                                    dem=dem,
                                    gt_ag=gt_ag.permute(1,2,0).numpy()
                                    )
                
                # get images
                rgb = sample['image']
                gt_lu = sample['mask'].long().permute(2,0,1).squeeze(dim=0)
                gt_ag = sample['gt_ag'].long().permute(2,0,1).squeeze(dim=0)
                pan = sample['pan']
                swir = sample['swir']
                vnir = sample['vnir']
                dem = sample['dem']
                # return results
                return rgb, pan, vnir, swir, dem, gt_lu, gt_ag

            else:
                # split inputs
                rgb, hs, dem, gt_lu, gt_ag = inps
                normalize_rgb, normalize_hs, normalize_dem, transforms_augmentation = transform_list

                # ipdb.set_trace()
                transforms = A.Compose([transforms_augmentation],
                                        additional_targets={'hs': 'image',
                                                            'dem': 'image',
                                                            'gt_ag': 'mask'})

                rgb = (rgb.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['rgb']) / (self.loaded_max_dict_before_normalization['rgb'] - self.loaded_min_dict_before_normalization['rgb'])
                # hs = (hs.numpy() - self.loaded_min_dict_before_normalization['hs']) / (self.loaded_max_dict_before_normalization['hs'] - self.loaded_min_dict_before_normalization['hs'])
                hs = (hs.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['hs']) / (self.loaded_max_dict_before_normalization['hs'] - self.loaded_min_dict_before_normalization['hs'])
                dem = (dem.permute(1,2,0).numpy() - self.loaded_min_dict_before_normalization['dem']) / (self.loaded_max_dict_before_normalization['dem'] - self.loaded_min_dict_before_normalization['dem'])

                rgb = normalize_rgb(image=rgb)['image']
                hs = normalize_hs(image=hs)['image']
                dem = normalize_dem(image=dem)['image']

                sample = transforms(image=rgb,
                                    mask=gt_lu.permute(1,2,0).numpy(),
                                    hs=hs,
                                    dem=dem,
                                    gt_ag=gt_ag.permute(1,2,0).numpy()
                                    )
                
                # get images
                rgb = sample['image']
                gt_lu = sample['mask'].long().permute(2,0,1).squeeze(dim=0)
                gt_ag = sample['gt_ag'].long().permute(2,0,1).squeeze(dim=0)
                hs = sample['hs']
                dem = sample['dem']

                # return results
                return rgb, hs, dem, gt_lu, gt_ag # Change back

        # return the function
        return transform_inputs

    def train_transforms(self):
        # define training size
        train_size = self.conf['train_size'] if 'train_size' in self.conf else self.conf['input_size']
        # create transformation

        if not(self.conf['pca']):
            normalize_rgb = A.Normalize(mean=self.mean_dict['rgb'], std=self.std_dict['rgb'], max_pixel_value=self.max_dict['rgb'])
            normalize_pan = A.Normalize(mean=self.mean_dict['pan'], std=self.std_dict['pan'], max_pixel_value=self.max_dict['pan'])
            normalize_vnir = A.Normalize(mean=self.mean_dict['vnir'], std=self.std_dict['vnir'], max_pixel_value=self.max_dict['vnir'])
            normalize_swir = A.Normalize(mean=self.mean_dict['swir'], std=self.std_dict['swir'], max_pixel_value=self.max_dict['swir'])
            normalize_dem = A.Normalize(mean=self.mean_dict['dem'], std=self.std_dict['dem'], max_pixel_value=self.max_dict['dem'])

            transforms_augmentation = A.Compose([A.Resize(*self.conf['input_size']),
                A.crops.transforms.RandomCrop(*train_size),
                # A.Rotate((-90,90)),#, value=0, border_mode=cv2.BORDER_CONSTANT),
                A.Rotate(limit=90),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.Transpose(p=0.3),
                # A.ShiftScaleRotate(p=0.3, value=0, border_mode=cv2.BORDER_CONSTANT),
                ToTensorV2()
            ])

            transforms = normalize_rgb, normalize_pan, normalize_vnir, normalize_swir, normalize_dem, transforms_augmentation
        else:
            normalize_rgb = A.Normalize(mean=self.mean_dict['rgb'], std=self.std_dict['rgb'], max_pixel_value=self.max_dict['rgb'])
            normalize_hs = A.Normalize(mean=self.mean_dict['hs'], std=self.std_dict['hs'], max_pixel_value=self.max_dict['hs'])
            normalize_dem = A.Normalize(mean=self.mean_dict['dem'], std=self.std_dict['dem'], max_pixel_value=self.max_dict['dem'])

            transforms_augmentation = A.Compose([A.Resize(*self.conf['input_size']),
                A.crops.transforms.RandomCrop(*train_size),
                # A.Rotate((-90,90)),#, value=0, border_mode=cv2.BORDER_CONSTANT),
                A.Rotate(limit=[-180, 180]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                # A.ShiftScaleRotate(p=0.3, value=0, border_mode=cv2.BORDER_CONSTANT),
                ToTensorV2()
            ])

            transforms = normalize_rgb, normalize_hs, normalize_dem, transforms_augmentation

        # create transform function
        return self.create_transform_function(transforms)
        

    def val_transforms(self):
        train_size = self.conf['train_size'] if 'train_size' in self.conf else self.conf['input_size']

        if not(self.conf['pca']):
            # create transformation
            normalize_rgb = A.Normalize(mean=self.mean_dict['rgb'], std=self.std_dict['rgb'], max_pixel_value=self.max_dict['rgb'])
            normalize_pan = A.Normalize(mean=self.mean_dict['pan'], std=self.std_dict['pan'], max_pixel_value=self.max_dict['pan'])
            normalize_vnir = A.Normalize(mean=self.mean_dict['vnir'], std=self.std_dict['vnir'], max_pixel_value=self.max_dict['vnir'])
            normalize_swir = A.Normalize(mean=self.mean_dict['swir'], std=self.std_dict['swir'], max_pixel_value=self.max_dict['swir'])
            normalize_dem = A.Normalize(mean=self.mean_dict['dem'], std=self.std_dict['dem'], max_pixel_value=self.max_dict['dem'])

            transforms_augmentation = A.Compose([
                A.Resize(*self.conf['input_size']),
                # A.crops.transforms.RandomCrop(*train_size),
                ToTensorV2()
            ])

            transforms = normalize_rgb, normalize_pan, normalize_vnir, normalize_swir, normalize_dem, transforms_augmentation
        else:
            # create transformation
            normalize_rgb = A.Normalize(mean=self.mean_dict['rgb'], std=self.std_dict['rgb'], max_pixel_value=self.max_dict['rgb'])
            normalize_hs = A.Normalize(mean=self.mean_dict['hs'], std=self.std_dict['hs'], max_pixel_value=self.max_dict['hs'])
            normalize_dem = A.Normalize(mean=self.mean_dict['dem'], std=self.std_dict['dem'], max_pixel_value=self.max_dict['dem'])

            transforms_augmentation = A.Compose([
                A.Resize(*self.conf['input_size']),
                # A.crops.transforms.RandomCrop(*train_size),
                ToTensorV2()
            ])

            transforms = normalize_rgb, normalize_hs, normalize_dem, transforms_augmentation
    
        # create transform function
        return self.create_transform_function(transforms)
    
    def test_transforms(self):
        return self.val_transforms()
        

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    # train or test
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True

    Base.main(KnownArchitectures)
        