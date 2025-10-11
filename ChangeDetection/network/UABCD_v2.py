import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones.pvtv2 import *
import warnings
warnings.filterwarnings('ignore')


class UABCD_v2(nn.Module):
    def __init__(self,num_classes):
        super(UABCD_v2, self).__init__()
        channel = 32
        self.conv1 = BasicConv2d(2*64, 64, 1)
        self.conv2 = BasicConv2d(2*128, 128, 1)
        self.conv3 = BasicConv2d(2*320, 320, 1)
        self.conv4 = BasicConv2d(2*512, 512, 1)

        self.conv_4 = BasicConv2d(512, channel, 3, 1, 1)
        self.conv_3 = BasicConv2d(320, channel, 3, 1, 1)
        self.conv_2 = BasicConv2d(128, channel, 3, 1, 1)
        self.conv_1 = BasicConv2d(64, channel, 3, 1, 1)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.coarse_out = nn.Sequential(nn.Conv2d(4 * channel, channel, kernel_size=3, stride=1, padding=1, bias=True),\
                                      nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, bias=True))

        self.init_backbone_weights()
        
    def init_backbone_weights(self):
        self.backbone = pvt_v2_b2()  
        path = './pretrain_backbones/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict, strict=False)
    
    def forward(self, A, B, mask=None):
        layer_1_A, layer_2_A, layer_3_A, layer_4_A = self.backbone(A)
        layer_1_B, layer_2_B, layer_3_B, layer_4_B = self.backbone(B)

        if mask!= None:
            layer_1_A = layer_1_A * (1+ F.interpolate(mask, size=layer_1_A.size()[2:], mode="bilinear", align_corners=True))
            layer_2_A = layer_2_A * (1+ F.interpolate(mask, size=layer_2_A.size()[2:], mode="bilinear", align_corners=True))
            layer_3_A = layer_3_A * (1+ F.interpolate(mask, size=layer_3_A.size()[2:], mode="bilinear", align_corners=True))
            layer_4_A = layer_4_A * (1+ F.interpolate(mask, size=layer_4_A.size()[2:], mode="bilinear", align_corners=True))

            layer_1_B = layer_1_B * (1+ F.interpolate(mask, size=layer_1_B.size()[2:], mode="bilinear", align_corners=True))
            layer_2_B = layer_2_B * (1+ F.interpolate(mask, size=layer_2_B.size()[2:], mode="bilinear", align_corners=True))
            layer_3_B = layer_3_B * (1+ F.interpolate(mask, size=layer_3_B.size()[2:], mode="bilinear", align_corners=True))
            layer_4_B = layer_4_B * (1+ F.interpolate(mask, size=layer_4_B.size()[2:], mode="bilinear", align_corners=True))
        
        layer_1 = self.conv_1(self.conv1(torch.cat((layer_1_A, layer_1_B), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((layer_2_A, layer_2_B), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((layer_3_A, layer_3_B), dim=1)))
        layer_4 = self.conv_4(self.conv4(torch.cat((layer_4_A, layer_4_B), dim=1)))
        
        Fusion = torch.cat((self.upsample8(layer_4), self.upsample4(layer_3), self.upsample2(layer_2), layer_1), 1)
        Guidance_out = self.coarse_out(Fusion)
        out = self.upsample4(Guidance_out)

        return  out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes,1e-5,0,True,True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class AggUnit(nn.Module):
    def __init__(self, features):
        super(AggUnit, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = F.interpolate( output, scale_factor=2, mode="bilinear", align_corners=True)

        return output


if __name__ == '__main__':
    A = torch.rand(4,3,256,256).cuda()
    B = torch.rand(4,3,256,256).cuda()

    model = UABCD(latent_dim=8, num_classes=1).cuda()

    outs = model(A,B)

    for o in outs:
        print(o.shape)

