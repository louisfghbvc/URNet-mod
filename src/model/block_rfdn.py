from collections import OrderedDict


import torch.nn as nn

import torch
import torch.nn.functional as F

from model import common

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


# no fuse, use residual
class PPMv2(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, in_dim, bins=(1, 3, 6, 8)):
        super(PPMv2, self).__init__()
        reduction_dim = int(in_dim/len(bins))
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                # nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = []
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return x + torch.cat(out, dim=1)

class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# compute channel mean, max
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = common.default_conv(2, 1, kernel_size)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, conv=None, reduction=2, sig=False):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = [
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            activation('lrelu', neg_slope=0.05),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        ]
        if sig:
            self.conv_du.append(nn.Sigmoid())
        self.conv_du = nn.Sequential(*self.conv_du)

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        # return x * y
        return y 

# CCA + SA
class CBAM(nn.Module):
    def __init__(self, gate_channels, conv, reduction_ratio=4, no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = CCALayer(gate_channels, conv, reduction_ratio)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

# CCA + SA
class RAMm(nn.Module):
    def __init__(self, n_feats, conv, reduction=4):
        super(RAMm, self).__init__()
        self.ca = CCALayer(n_feats, conv, reduction=reduction)
        self.sa = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # b c h w -> b 1 h w
        sa_mean = torch.mean(x, 1).unsqueeze(1)
        ca = self.ca(x)
        sa = self.sa(sa_mean)
        sa = F.interpolate(sa, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        att = ca + sa
        att = self.sigmoid(att)
        return x * att

# CCA + SA, multiply
class IRAMm(nn.Module):
    def __init__(self, n_feats, conv, reduction=4):
        super(IRAMm, self).__init__()
        self.ca = CCALayer(n_feats, conv, reduction=reduction)
        self.sa = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # b c h w -> b 1 h w
        sa_mean = torch.mean(x, 1).unsqueeze(1)
        ca = self.ca(x)
        sa = self.sa(sa_mean)
        sa = F.interpolate(sa, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        att = ca * sa
        att = self.sigmoid(att)
        return x * att


# ECCA + SA
class RAMm2(nn.Module):
    def __init__(self, n_feats, conv, reduction=4):
        super(RAMm2, self).__init__()
        self.ca = ECCALayer(n_feats, conv, reduction=reduction)
        self.sa = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # b c h w -> b 1 h w
        sa_mean = torch.mean(x, 1).unsqueeze(1)
        ca = self.ca(x)
        sa = self.sa(sa_mean)
        sa = F.interpolate(sa, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        att = ca + sa
        att = self.sigmoid(att)
        return x * att

# CCA + ESA
class RAMm3(nn.Module):
    def __init__(self, n_feats, conv, reduction=4):
        super(RAMm3, self).__init__()
        self.ca = CCALayer(n_feats, conv, reduction=reduction)
        self.sa = ESA(n_feats, conv)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # b c h w -> b 1 h w
        ca = self.ca(x)
        sa = self.sa(x)
        att = ca + sa
        att = self.sigmoid(att)
        return x * att

# pure SA
class SA(nn.Module):
    def __init__(self, n_feats, conv, reduction=4):
        super(SA, self).__init__()
        self.sa = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # b c h w -> b 1 h w
        sa_mean = torch.mean(x, 1).unsqueeze(1)
        sa = self.sa(sa_mean)
        sa = F.interpolate(sa, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        att = self.sigmoid(sa)
        return x * att

# contrast-aware channel attention module + ECA, cur fix 3
class ECCALayer(nn.Module):
    def __init__(self, kernel_size, conv, reduction=4):
        super(ECCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # kernel size fix 3
        self.conv_du = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        return x * m

# fuse CCA
class ESAv2(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESAv2, self).__init__()

        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        # CCA
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # ESA
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)

        # CCA
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv4(y)

        # add then, sigmoid
        m = self.sigmoid(c4 + y)
        
        return x * m

# fuse CCA
class ESAv3(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESAv3, self).__init__()

        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        # CCA
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # channel pool and std
        y = self.contrast(x) + self.avg_pool(x)

        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)

        m = self.sigmoid(c4)
        
        return x * m

# high frequency attention tweak
class MCA(nn.Module):
    def __init__(self, n_feats, conv):
        super(MCA, self).__init__()
        f = n_feats // 4

        # down scale and up scale
        self.reduction = conv(n_feats, f, kernel_size=1)
        self.conv1 = conv(f, f, kernel_size=1)
        self.expansion = conv(f, n_feats, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n, c, h, w = x.size()

        # fetch h, w feature
        w_max = F.adaptive_avg_pool2d(x, output_size=(h, 1))
        h_max = F.adaptive_avg_pool2d(x, output_size=(1, w))
        # hw_f = torch.cat([w_max, h_max], dim=-1)
        hw_f = torch.matmul(w_max, h_max)

        # reduce
        y = self.reduction(x - hw_f)

        # convert feature and to point
        y = self.conv1(y)
        y = self.relu(y)

        # recover
        y = self.expansion(y)

        m = self.sigmoid(y)

        return x * m

# local attention V2
class MCAv2(nn.Module):
    def __init__(self, n_feats, conv):
        super(MCAv2, self).__init__()
        f = n_feats // 4

        # down scale and up scale
        self.reduction = conv(n_feats, f, kernel_size=1)
        self.conv_h = nn.Conv1d(f, f, kernel_size=3, padding=1, bias=False)
        self.conv_w = nn.Conv1d(f, f, kernel_size=3, padding=1, bias=False)
        self.expansion = conv(f, n_feats, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reduction(x)
        n, c, h, w = x.size()

        # fetch h, w feature
        # n, c, h, 1
        hf = F.adaptive_avg_pool2d(x, output_size=(h, 1))
        hf = self.conv_h(hf.squeeze(-1)).unsqueeze(-1)
        hf = self.sigmoid(hf)

        # n, c, 1, w
        wf = F.adaptive_avg_pool2d(x, output_size=(1, w))
        wf = self.conv_w(wf.squeeze(-2)).unsqueeze(-2)
        wf = self.sigmoid(wf)

        out = x * hf.expand_as(x) + x * wf.expand_as(x)

        return self.expansion(out)

# local attention V3
class MCAv3(nn.Module):
    def __init__(self, n_feats, conv):
        super(MCAv3, self).__init__()
        f = n_feats // 4

        # down scale and up scale
        self.reduction = conv(n_feats, f, kernel_size=1)
        self.conv_h = nn.Conv1d(f, f, kernel_size=3, padding=1, bias=True)
        self.conv_w = nn.Conv1d(f, f, kernel_size=3, padding=1, bias=True)
        self.fuse = conv(3*f, f, kernel_size=1)
        self.expansion = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reduction(x)
        n, c, h, w = x.size()

        # fetch h, w feature
        # n, c, h, 1
        hf = F.adaptive_avg_pool2d(x, output_size=(h, 1))
        hf = self.conv_h(hf.squeeze(-1)).unsqueeze(-1)
        hf = self.sigmoid(hf)

        # n, c, 1, w
        wf = F.adaptive_avg_pool2d(x, output_size=(1, w))
        wf = self.conv_w(wf.squeeze(-2)).unsqueeze(-2)
        wf = self.sigmoid(wf)

        # concate all features
        x_fuse = torch.cat([x * hf.expand_as(x), x * wf.expand_as(x), x], dim=1)
        out = self.fuse(x_fuse)

        return self.expansion(out)

# local attention V4
class MCAv4(nn.Module):
    def __init__(self, n_feats, conv):
        super(MCAv4, self).__init__()
        f = n_feats // 4

        # down scale and up scale
        self.reduct_h = nn.Conv1d(n_feats, f, kernel_size=3, padding=1, bias=True)
        self.expansion_h = nn.Conv1d(f, n_feats, kernel_size=3, padding=1, bias=True)
        self.reduct_w = nn.Conv1d(n_feats, f, kernel_size=3, padding=1, bias=True)
        self.expansion_w = nn.Conv1d(f, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # fuse
        self.fuse = conv(3*n_feats, n_feats, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()

        # fetch h, w feature
        # n, c, h, 1
        hf = F.adaptive_avg_pool2d(x, output_size=(h, 1))
        hf = self.reduct_h(hf.squeeze(-1)).unsqueeze(-1)
        hf = self.relu(hf)
        hf = self.expansion_h(hf.squeeze(-1)).unsqueeze(-1)
        hf = self.sigmoid(hf)
        
        # n, c, 1, w
        wf = F.adaptive_avg_pool2d(x, output_size=(1, w))
        wf = self.reduct_w(wf.squeeze(-2)).unsqueeze(-2)
        wf = self.relu(wf)
        wf = self.expansion_w(wf.squeeze(-2)).unsqueeze(-2)
        wf = self.sigmoid(wf)

        # concate all features
        x_fuse = torch.cat([x * hf.expand_as(x), x * wf.expand_as(x), x], dim=1)
        out = self.fuse(x_fuse)

        return out

class E_RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, add=False, shuffle=False, att=ESA):
        super(E_RFDB, self).__init__()
        self.add = add
        self.shuffle = shuffle
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = att(in_channels, nn.Conv2d)

    def forward(self, input):
        if self.shuffle: # channel shuffle
            input = common.channel_shuffle(input, 2)

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        if self.shuffle: # channel shuffle
            out_fused = common.channel_shuffle(out_fused, 2)

        if self.add:
            return out_fused + input
        else:
            return out_fused

# 1x1 convolution 
class E_RFDB1x1(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, add=False, shuffle=False, att=ESA):
        super(E_RFDB1x1, self).__init__()
        self.add = add
        self.shuffle = shuffle
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 1)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 1)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 1)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = att(in_channels, nn.Conv2d)
        if add:
            self.res = conv_layer(in_channels, in_channels, 1)


    def forward(self, input):
        if self.shuffle: # channel shuffle
            input = common.channel_shuffle(input, 2)

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        if self.shuffle: # channel shuffle
            out_fused = common.channel_shuffle(out_fused, 2)

        if self.add:
            return out_fused + self.res(input)
        else:
            return out_fused


# 1x1 convolution all you need
class EEFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, add=False, shuffle=False, att=ESA):
        super(EEFDB, self).__init__()
        self.add = add
        self.shuffle = shuffle
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r_sq = conv_layer(self.rc, self.dc, 1) # squeeze
        self.c1_r_ex = conv_layer(self.dc, self.rc, 1) # extand
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r_sq = conv_layer(self.rc, self.dc, 1)
        self.c2_r_ex = conv_layer(self.dc, self.rc, 1)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r_sq = conv_layer(self.rc, self.dc, 1)
        self.c3_r_ex = conv_layer(self.dc, self.rc, 1)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = att(in_channels, nn.Conv2d)

    def forward(self, input):
        if self.shuffle: # channel shuffle
            input = common.channel_shuffle(input, 2)

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r_sq(input))
        r_c1 = self.act(r_c1)
        r_c1 = self.c1_r_ex(r_c1) + input # residual

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r_sq(r_c1))
        r_c2 = self.act(r_c2)
        r_c2 = self.c2_r_ex(r_c2) + r_c1

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r_sq(r_c2))
        r_c3 = self.act(r_c3)
        r_c3 = self.c3_r_ex(r_c3) + r_c2

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        if self.shuffle: # channel shuffle
            out_fused = common.channel_shuffle(out_fused, 2)

        if self.add:
            return out_fused + input
        else:
            return out_fused

# efficient channel feauture distillation 
class ECFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, add=False, shuffle=False, att=ESA):
        super(ECFDB, self).__init__()
        self.add = add
        self.shuffle = shuffle
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = CCALayer(in_channels) # no sigmoid
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = CCALayer(in_channels) # no sigmoid
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = CCALayer(in_channels) # no sigmoid
        self.c4 = conv_layer(self.remaining_channels, self.dc, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = att(in_channels, nn.Conv2d)

    def forward(self, input):
        if self.shuffle: # channel shuffle
            input = common.channel_shuffle(input, 2)

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input) + input

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1) + r_c1

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2) + r_c2

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        if self.shuffle: # channel shuffle
            out_fused = common.channel_shuffle(out_fused, 2)

        if self.add:
            return out_fused + input
        else:
            return out_fused

# down sample directly
class E_RFDDB1x1(nn.Module):
    def __init__(self, in_channels, out_channels, distillation_rate=0.25, add=False, shuffle=False, att=ESA):
        super(E_RFDDB1x1, self).__init__()
        self.add = add
        self.shuffle = shuffle
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 1)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 1)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 1)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 1)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, out_channels, 1)
        self.esa = att(out_channels, nn.Conv2d)

    def forward(self, input):
        if self.shuffle: # channel shuffle
            input = common.channel_shuffle(input, 2)

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        if self.shuffle: # channel shuffle
            out_fused = common.channel_shuffle(out_fused, 2)

        if self.add:
            return out_fused + input
        else:
            return out_fused

# E_RFDB Unet
class E_RFDB_U(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, add=False, shuffle=False, att=ESA):
        super(E_RFDB_U, self).__init__()
        self.add = add
        self.shuffle = shuffle

        # distillation channel
        self.dc = in_channels//2
        # remaining channel
        self.rc = in_channels

        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2 = conv_layer(self.rc, self.rc, 3)
        self.c3 = conv_layer(2*self.rc, self.rc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.catlayer = conv_layer(self.dc*3, in_channels, 1)
        self.esa = att(in_channels, nn.Conv2d)

    def forward(self, input):
        if self.shuffle: # channel shuffle
            input = common.channel_shuffle(input, 2)

        distilled_c1 = self.act(self.c1_d(input))
        # c1 srn
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1+input)

        # c2 srn
        r_c2 = (self.c2(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        breakpoint()

        # c3 srn
        tmp_fuse = torch.cat([r_c1, r_c2], dim=1)
        r_c3 = self.c3(tmp_fuse)
        r_c3 = self.act(tmp_fuse + r_c3)

        out = torch.cat([distilled_c1, r_c3], dim=1)
        out_fused = self.esa(self.catlayer(out))

        if self.shuffle: # channel shuffle
            out_fused = common.channel_shuffle(out_fused, 2)

        if self.add:
            return out_fused + input
        else:
            return out_fused

# share weight, like merge sort
class E_RFDB_Share(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, add=False, shuffle=False):
        super(E_RFDB_Share, self).__init__()
        self.add = add
        self.shuffle = shuffle
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*2, in_channels, 1) # share weight
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        if self.shuffle: # channel shuffle
            input = common.channel_shuffle(input, 2)

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        _out1 = self.c5(torch.cat([distilled_c1, distilled_c2], dim=1))
        _out2 = self.c5(torch.cat([distilled_c3, r_c4], dim=1))
        out_fused = self.esa(self.c5(_out1 + _out2))

        if self.shuffle: # channel shuffle
            out_fused = common.channel_shuffle(out_fused, 2)

        if self.add:
            return out_fused + input
        else:
            return out_fused

# share weight, like merge sort
class E_RFDB_ShareV2(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, add=False, shuffle=False, att=ESA):
        super(E_RFDB_ShareV2, self).__init__()
        self.add = add
        self.shuffle = shuffle
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*2, self.dc, 1) # share weight
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        if self.shuffle: # channel shuffle
            input = common.channel_shuffle(input, 2)

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        _out1 = self.c5(torch.cat([distilled_c1, distilled_c2], dim=1))
        _out2 = self.c5(torch.cat([distilled_c3, r_c4], dim=1))
        out_fused = self.esa(torch.cat([_out1, _out2], dim=1))

        if self.shuffle: # channel shuffle
            out_fused = common.channel_shuffle(out_fused, 2)

        if self.add:
            return out_fused + input
        else:
            return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

