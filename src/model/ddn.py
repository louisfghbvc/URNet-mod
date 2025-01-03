import torch
import torch.nn as nn

from model import common
from model.block import *
from model.block_rfdn import *


def make_model(args, parent=False):
    return DDN(args)

# Distillation of Distillation net
class DDN(nn.Module):
    def __init__(self, args):
        super(DDN, self).__init__()
        nf = args.n_feats
        scale = args.scale[0]
        ng = args.n_rfddbgroups

        self.test_only = args.test_only
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = ACBlock(args.n_colors, nf, kernel_size=3)
        
        self.body = [RFDBGroup(nf, add=False, bone=E_RFDDB1x1, index=i) for i in range(ng)]
        self.body = nn.ModuleList(self.body)

        self.convE = common.default_conv(nf, nf, kernel_size=3)
        self.conv = common.default_conv(nf, nf, kernel_size=3)
        
        self.anrb = ANRBsoft(nf)

        self.tail_up = pixelshuffle_block(nf, args.n_colors, upscale_factor=scale)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        input_x = x

        # body part
        remain = []
        for body in self.body:
            x, v = body(x)
            remain.append(v)
        remain = torch.cat(remain, dim=1)

        x = self.conv(x)
        remain = self.convE(remain)
        x += remain
        x += input_x
        x = self.anrb(x)

        sr_out = self.tail_up(x)
        sr_out = self.add_mean(sr_out)

        return sr_out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

