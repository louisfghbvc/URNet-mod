import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class sMLPBlock(nn.Module):
    def __init__(self, h=224, w=224, c=3):
        super().__init__()
        self.proj_h = nn.Linear(h, h)
        self.proj_w = nn.Linear(w, w)
        self.fuse = nn.Linear(3*c,c)
    
    def forward(self,x):
        # b c h w
        x_h = self.proj_h(x.permute(0,1,3,2)).permute(0,1,3,2)
        x_w = self.proj_w(x)
        x_id = x
        x_fuse = torch.cat([x_h, x_w, x_id], dim = 1)
        out = self.fuse(x_fuse.permute(0,2,3,1)).permute(0,3,1,2)
        return out

# mlpblock compress lightweight
class sMLPBlockLight(nn.Module):
    def __init__(self, h=224, w=224, c=3):
        super().__init__()
        self.reduction = default_conv(in_channels=c, out_channels=1, kernel_size=1)
        self.proj_h = nn.Linear(h, h)
        self.proj_w = nn.Linear(w, w)
        self.fuse = nn.Linear(3,1)
        self.expansion = default_conv(in_channels=1, out_channels=c, kernel_size=1)
    
    def forward(self,x):
        # b 1 h w
        x_h = self.proj_h(x.permute(0,1,3,2)).permute(0,1,3,2)
        x_w = self.proj_w(x)
        x_id = x
        x_fuse = torch.cat([x_h, x_w, x_id], dim = 1)
        out = self.fuse(x_fuse.permute(0,2,3,1)).permute(0,3,1,2)
        return self.expansion(out)

