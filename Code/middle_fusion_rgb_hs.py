import torch
import torch.nn as nn

class Middle_fusion_en(nn.Module):

    def __init__(self,
                 conf_rgb={'channels':[3,3], 'kernels':[3]},
                 conf_hs={'channels':[4,4], 'kernels':[3]},
                ):

        """
        conf_modality is a dict with:
            channels
            kernels

        e.g.
            conf_rgb = {'channels':[3,32,64], 'kernels':[7,5]}
            conf_hs = {'channels':[4,32,64], 'kernels':[7,5]}
            conf_dem = {'channels':[1,32,64], 'kernels':[7,5]}
        """

        super(Middle_fusion_en, self).__init__()

        if(len(conf_rgb['channels']) != len(conf_rgb['kernels'])+1):
             raise Exception("RGB configurations is wrong, channels length must be equal to kernels length + 1")
        if(len(conf_hs['channels']) != len(conf_hs['kernels'])+1):
            raise Exception("Hs configurations is wrong, channels length must be equal to kernels length + 1") 

        # rgb convoltuions
        self.conv_rgb = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=conf_rgb['channels'][i], out_channels=conf_rgb['channels'][i+1],
                          kernel_size=conf_rgb['kernels'][i], padding=conf_rgb['kernels'][i]//2, stride=1),
                nn.ReLU()) for i in range(len(conf_rgb['kernels']))]
            )

        # hs convoltuions
        self.conv_hs = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=conf_hs['channels'][i], out_channels=conf_hs['channels'][i+1],
                          kernel_size=conf_hs['kernels'][i], padding=conf_hs['kernels'][i]//2, stride=1),
                nn.ReLU()) for i in range(len(conf_hs['kernels']))]
            )

    def forward(self, inp):
        rgb, hs = inp

        rgb = self.conv_rgb(rgb)
        hs = self.conv_hs(hs)

        return torch.cat((rgb, hs), dim=1)


