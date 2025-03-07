import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
from mmdet3d.registry import MODELS


class BinarizeIndictator(autograd.Function):
    @staticmethod
    def forward(ctx, indicator):
        out = (indicator >= .1).float()
        return out
    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class freeze_conv(nn.Conv2d):
    def __init__(self ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        weight_shape = self.weight.shape
        c1, c2, _, _ = weight_shape

        self.register_parameter(name = 'B1_residual', param = torch.nn.Parameter(torch.zeros(weight_shape)))
        self.register_parameter(name = 'B1_indicator', param = torch.nn.Parameter(torch.ones([1])*.15))

        self.register_parameter(name = 'B2_residual', param = torch.nn.Parameter(torch.zeros(weight_shape)))
        self.register_parameter(name = 'B2_indicator', param = torch.nn.Parameter(torch.ones([1])*.15))

        self.register_parameter(name='B3_residual', param=torch.nn.Parameter(torch.zeros(weight_shape)))
        self.register_parameter(name='B3_indicator', param=torch.nn.Parameter(torch.ones([1]) * .15))

        self.weight.requires_grad = False
        self.weight.grad = None
    def forward(self, x ,flag = [1,0,0]):
        flag_tensor =  torch.tensor(np.array(flag))
        I1 = BinarizeIndictator.apply(self.B1_indicator)
        I2 = BinarizeIndictator.apply(self.B2_indicator)
        I3 = BinarizeIndictator.apply(self.B3_indicator)

        w = self.weight
        x_ = F.conv2d(x, w,self.bias, self.stride,self.padding,self.dilation,self.groups)
        x1 = flag_tensor[0] * I1 *  F.conv2d(x, self.B1_residual ,self.bias,self.stride,self.padding,self.dilation,self.groups)
        x2 = flag_tensor[1] * I2 * F.conv2d(x, self.B2_residual, self.bias, self.stride,self.padding,self.dilation,self.groups)
        x3 = flag_tensor[2] * I3 * F.conv2d(x, self.B3_residual, self.bias, self.stride,self.padding,self.dilation,self.groups)
        x = x_ + x1 + x2 + x3

        return x

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return freeze_conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return freeze_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False)

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias= True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        #layers = list()
        self.transpose= transpose
        if self.transpose:
            padding = kernel_size // 2 -1
            self.layer = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)
        else:
            if kernel_size == 1:
                self.layer = conv1x1(in_planes=in_channel, out_planes=out_channel)
            elif kernel_size == 3:
                self.layer = conv3x3(in_planes=in_channel, out_planes=out_channel, stride=stride, groups=1,
                                     padding=padding)

        self.relu= relu
        if self.relu:
            self.act = nn.GELU()
            #layers.append() # nn.ReLU(inplace=True)
        #self.main = nn.Sequential(*layers)

    def forward(self, x, flag = [1,0,0]):
        if self.relu:
            if self.transpose:
                return self.act(self.layer(x))  #self.main(x,flag =flag)
            else:
                return self.act(self.layer(x,flag =flag))  #self.main(x,flag =flag)

        else:
            return self.layer(x,flag =flag)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = BasicConv(in_channel=Cin, out_channel=G, kernel_size=kSize, stride=1, relu=True)
    def forward(self, x,flag = [1,0,0]):
        out = self.conv(x, flag =flag)
        return torch.cat((x, out), 1)

class RDBlock(nn.Module):
    def __init__(self, in_channel, out_channel, nConvLayers=3):
        super(RDBlock, self).__init__()
        G0 = in_channel
        G = in_channel
        C = nConvLayers

        self.conv0 = RDB_Conv(G0 , G)
        self.conv1 = RDB_Conv(G0 + 1 * G , G)
        self.conv2 = RDB_Conv(G0 + 2 * G , G)
        # Local Feature Fusion
        self.LFF = BasicConv(in_channel=G0 + C * G, out_channel=out_channel, kernel_size=1, stride=1, relu=False)

    def forward(self, x,flag = [1,0,0] ):
        out = self.conv0(x,flag =flag)
        out = self.conv1(out,flag =flag)
        out = self.conv2(out,flag =flag)
        out = self.LFF(out,flag =flag) + x
        return out


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        self.layer1 = RDBlock(out_channel, out_channel)
        self.layer2 = RDBlock(out_channel, out_channel)
        self.layer3 = RDBlock(out_channel, out_channel)

    def forward(self, x, flag = [1,0,0]):
        out = self.layer1(x, flag =flag)
        out = self.layer2(out, flag =flag)
        out = self.layer3(out, flag =flag)
        return out


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        self.layer1 = RDBlock(channel, channel)
        self.layer2 = RDBlock(channel, channel)
        self.layer3 = RDBlock(channel, channel)

    def forward(self, x,flag = [1,0,0]):
        out = self.layer1(x,flag =flag)
        out = self.layer2(out,flag =flag)
        out = self.layer3(out,flag =flag)
        return out



class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()

        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
    def forward(self, x1, x2, x4, flag = [1,0,0] ):
        x = torch.cat([x1, x2, x4], dim=1)
        out = self.conv1(x, flag =flag )
        out = self.conv2(out, flag =flag )
        return out


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.layer1 = BasicConv(32, out_plane // 2, kernel_size=3, stride=1, relu=True)
        self.layer2 = BasicConv(out_plane // 2, out_plane-32, kernel_size=1, stride=1, relu=True)
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x, flag = [1,0,0]):
        out = self.layer1(x,flag =flag)
        out = self.layer2(out,flag =flag)
        x = torch.cat([x,out ], dim=1)
        return self.conv(x,flag =flag)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2,flag = [1,0,0]):
        x = x1 * x2
        out = x1 + self.merge(x,flag= flag)
        return out


@MODELS.register_module()
class UNet_Stage2_K3(nn.Module):
    def __init__(self, base_channel=24, num_res=6):
        super(UNet_Stage2_K3, self).__init__()

        base_channel = base_channel

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(32, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 24, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])
        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2),
            AFF(base_channel * 7, base_channel * 4)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
    def getIndicators_B1(self):
        indicators = []
        for i in self.named_parameters():
            if 'B1_indicator' in i[0]:
                indicators.append(i[1])
        return indicators

    def getIndicators_B2(self):
        indicators = []
        for i in self.named_parameters():
            if 'B2_indicator' in i[0]:
                indicators.append(i[1])
        return indicators
    def getIndicators_B3(self):
        indicators = []
        for i in self.named_parameters():
            if 'B3_indicator' in i[0]:
                indicators.append(i[1])
        return indicators

    def forward(self, x, flag = [1,0,0] ):

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2,flag = flag)
        z4 = self.SCM1(x_4,flag = flag)

        res1 = self.Encoder[0](x,flag = flag)

        z = self.feat_extract[0](res1,flag = flag)
        z = self.FAM2(z, z2 ,flag = flag)
        res2 = self.Encoder[1](z ,flag = flag)

        z = self.feat_extract[1](res2 ,flag = flag)
        z = self.FAM1(z, z4 ,flag = flag)

        z = self.Encoder[2](z ,flag = flag)

        #-----------------------------inter-------------------#
        z12 = F.interpolate(res1, scale_factor=0.5)
        z14 = F.interpolate(res1, scale_factor=0.25)
        z21 = F.interpolate(res2, scale_factor=2)
        z24 = F.interpolate(res2, scale_factor=0.5)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        z = self.AFFs[2](z14, z24, z ,flag = flag)
        res2 = self.AFFs[1](z12, res2, z42 , flag = flag)
        res1 = self.AFFs[0](res1, z21, z41 , flag = flag)
        # -----------------------------inter-------------------#

        z = self.Decoder[0](z, flag = flag)
        z = self.feat_extract[2](z, flag = flag)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z, flag = flag)
        z = self.Decoder[1](z, flag = flag)
        z = self.feat_extract[3](z, flag = flag)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z, flag = flag)
        z = self.Decoder[2](z, flag = flag)
        z = self.feat_extract[4](z, flag = flag)
        return z


