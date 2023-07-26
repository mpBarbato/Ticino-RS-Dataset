import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
# from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import rasterio
import tifffile as tiff
import einops
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import rasterio as rs


class Dataset(data.Dataset):

    def __init__(self, root_dir, csv_file, pca=False, trans=None):
        # save
        self.fns = pd.read_csv(csv_file, names=['fns'], header=None)
        self.root_dir = root_dir
        self.pca = pca
        self.trans = trans

    def __len__(self):
        return len(self.fns)

    # cv2 read unchanged
    def read_tif(self, sub_dir, fn):
        # define filename
        fn = os.path.join(self.root_dir, sub_dir, fn)
        # read tif
        # data = tiff.imread(fn)
        data = rs.open(fn).read()
        # add channel
        if data.ndim < 3:
            data = np.expand_dims(data, axis=-1)
        # src = rasterio.open(fn)
        # data = src.read() #.copy()
        # import ipdb; ipdb.set_trace()
        # if np.isnan(data).any():
        #     import ipdb; ipdb.set_trace()
        # src.close()
        return data

    def __getitem__(self, idx):
        # defines sources
        # sources = ['RGB', 'PAN', 'VNIR', 'SWIR', 'DEM', 'gt_landuse/tif', 'gt_agricolture/tif']
        
        if not(self.pca):
            sources = ['Sources/RGB', 'Sources/PAN', 'Sources/VNIR', 'Sources/SWIR', 'Sources/DEM', 'Labeling/Landuse/tif', 'Labeling/Agriculture/tif']
            # load them
            imgs = [self.read_tif(cur_source,self.fns.iloc[idx]['fns']) for cur_source in sources]
            # make them proper type
            num_modalities = 5
            imgs[:num_modalities] = [torch.from_numpy(cur_img).float() for cur_img in imgs[:num_modalities]]
            imgs[num_modalities:] = [torch.from_numpy(cur_img).long().squeeze(-1) for cur_img in imgs[num_modalities:]]
            # change channel order
            # imgs[:5] = [einops.rearrange(cur_img, 'h w c -> c h w') for cur_img in imgs[:5]] # TO Re-insert
            # set rgb between 0 and 1
            # imgs[0] /= 2**8 # TO Re-insert
            # debug
            gt_lu = imgs[-2]
            gt_ag = imgs[-1]
            # if gt_lu.max() > 10 or gt_lu.min() < 0:
            #     import ipdb; ipdb.set_trace()
            # if gt_ag.max() > 19 or gt_ag.min() < 0:
            #     import ipdb; ipdb.set_trace()
            # debug
            # apply transforms
            # import ipdb; ipdb.set_trace()

        else:
            sources = ['Sources/RGB', 'Sources/HS', 'Sources/DEM', 'Labeling/Landuse/tif', 'Labeling/Agriculture/tif']
            # load them
            imgs = []
            for cur_source in sources:
                if cur_source.split('/')[1] == 'HS':
                    # imgs.append(np.load(os.path.join(self.root_dir, cur_source, self.fns.iloc[idx]['fns'].split('.')[0] + '.npy')))
                    imgs.append(self.read_tif('/ssd_data/pansh_data/', self.fns.iloc[idx]['fns']))
                else:
                    imgs.append(self.read_tif(cur_source, self.fns.iloc[idx]['fns']))

            # make them proper type
            num_modalities = 3
            imgs[:num_modalities] = [torch.from_numpy(cur_img).float() for cur_img in imgs[:num_modalities]]
            imgs[num_modalities:] = [torch.from_numpy(cur_img).long().squeeze(-1) for cur_img in imgs[num_modalities:]]
            # change channel order
            # imgs[:5] = [einops.rearrange(cur_img, 'h w c -> c h w') for cur_img in imgs[:5]] # TO Re-insert
            # set rgb between 0 and 1
            # imgs[0] /= 2**8 # TO Re-insert
            # debug
            gt_lu = imgs[-2]
            gt_ag = imgs[-1]
            # if gt_lu.max() > 10 or gt_lu.min() < 0:
            #     import ipdb; ipdb.set_trace()
            # if gt_ag.max() > 19 or gt_ag.min() < 0:
            #     import ipdb; ipdb.set_trace()
            # debug
            # apply transforms
            # import ipdb; ipdb.set_trace()

        if self.trans is not None:
            imgs = self.trans(imgs)
        # return

        # return imgs, self.fns.iloc[idx]['fns']

        inputs = imgs[:num_modalities]
        targets = imgs[num_modalities:]

        # remove park (temporary)
        # gt_lu = imgs[-2]
        # gt_lu[gt_lu == 4] = 3
        # gt_lu[gt_lu > 4] = gt_lu[gt_lu > 4] - 1
        
        # remove classes from agriculture
        # gt_ag = imgs[-1]
        # list2remove = [2,3,4,5,7,12,13,14,16,18]
        # val2remove = 0
        # for i in range(20):
        #     if i in list2remove:
        #         if i == 3:
        #             gt_ag[gt_ag == i] = 19 # forest in natural vegetation
        #         if i == 16:
        #             gt_ag[gt_ag == i] = 17 # natural barren in water
        #         else:
        #             gt_ag[gt_ag == i] = 0

        #         val2remove += 1
        #     else:
        #         gt_ag[gt_ag == i] = gt_ag[gt_ag == i] - val2remove

        targets = gt_lu, gt_ag

        return inputs, targets, self.fns.iloc[idx]['fns']


if __name__ == '__main__':
    import torch.nn.functional as F
    conf = {
        "root_dir": "/mnt/disco_pirelli/datasets/hyperspectral_segmentation/Dataset",
        "train_csv":"/mnt/disco_pirelli/datasets/hyperspectral_segmentation/Dataset/List/train_set.txt",
        "batch_size": 8,
        "num_workers": 0,
    }
    # 
    def transform_inputs(inps):
        conf = {"input_size": [256, 352]}
        # split inputs
        rgb, pan, vnir, swir, dem, gt_lu, gt_ag = inps
        # apply transformations
        transforms = A.Compose([#T.RandomCrop((100, 100)),
                            A.Resize(conf['input_size'][0], conf['input_size'][1]),
                            A.Rotate((-90,90), value=-1),
                            A.HorizontalFlip(0.3),
                            A.VerticalFlip(0.3),
                            ToTensorV2()
                            ],
                           additional_targets={'pan': 'image',
                                               'swir': 'image',
                                               'vnir': 'image',
                                               'dem': 'image',
                                               'gt_ag': 'mask'}
                           )
        #
        print(vnir.min(), vnir.max())
        sample = transforms(image=rgb.permute(1,2,0).numpy(),
                            mask=gt_lu.numpy(),
                            pan=pan.permute(1,2,0).numpy(),
                            vnir=vnir.permute(1,2,0).numpy(),
                            swir=swir.permute(1,2,0).numpy(),
                            dem=dem.permute(1,2,0).numpy(),
                            gt_ag=gt_ag.numpy()
                            )
            
        rgb = sample['image']
        gt_lu = sample['mask']
        gt_ag = sample['gt_ag']
        pan = sample['pan']
        swir = sample['swir']
        vnir = sample['vnir']
        dem = sample['dem']
        print(vnir.min(), vnir.max())
        # return results
        return rgb, pan, vnir, swir, dem, gt_lu, gt_ag

    ds = Dataset(conf['root_dir'], conf['train_csv'], transform_inputs)
    dl = data.DataLoader(ds, batch_size=conf['batch_size'], num_workers=conf['num_workers'], shuffle=True)

    # device = 'cuda'
    device = 'cpu'

    minn = None
    maxx = None
    from tqdm import tqdm

    for rgb, pan, nir, swir, dem, gt_lu, gt_ag in tqdm(ds):
        if minn is None or gt_ag.min() < minn:
            minn = gt_ag.min()
        if maxx is None or gt_ag.max() > maxx:
            maxx = gt_ag.max()

    print(f'min = {minn}')
    print(f'max = {maxx}')


# lu 0-8
# ag 0-19
