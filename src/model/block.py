import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import init

from model.block_rfdn import *


class CropLayer(nn.Module):

    # E.g., (-1, 0) means this layer should crop the first and last rows of the feature map.
    # And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class ACBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        self.padding = kernel_size // 2
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=self.padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:

            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=self.padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)

            center_offset_from_origin_border = self.padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:

            square_outputs = self.square_conv(input)

            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)

            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)

            return square_outputs + vertical_outputs + horizontal_outputs

# TODO: modify
class CFPB(nn.Module):
    def __init__(self, channel, res=False, fuse=True):
        super(CFPB, self).__init__()
        self.res = res
        self.fuse = fuse
        self.atrous_block3 = nn.Conv2d(channel, channel, 3, 1, padding=3, dilation=3)
        self.atrous_block6 = nn.Conv2d(channel, channel, 3, 1, padding=6, dilation=6)
        convs = []
        for i in range(2):
            convs.append(nn.Conv2d(2 * channel, channel, kernel_size=1, padding=0))
        self.convs = nn.ModuleList(convs)
        if self.fuse:
            self.fuse_conv = nn.Conv2d(channel * 3, channel, kernel_size=1, padding=0)

    def forward(self, x):
        ab3 = self.atrous_block3(x)
        ab3 = torch.cat([ab3, x], 1)
        ab3 = self.convs[0](ab3)

        ab6 = self.atrous_block6(ab3)
        ab6 = torch.cat([ab6, ab3], 1)
        ab6 = self.convs[1](ab6)

        out = torch.cat([x, ab3, ab6], 1)
        if self.fuse:
            out = self.fuse_conv(out)
        if self.res and self.fuse:
            out += x
        return out

# fuse
class PPM(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, in_dim, bins=(1, 3, 6, 8)):
        super(PPM, self).__init__()
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
        self.fuse = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return self.fuse(torch.cat(out, 1)) 

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

class RFDBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ver=False, tail=False, add=False, shuffle=False, bone=E_RFDB, att=ESA):
        super(RFDBBlock, self).__init__()
        if ver:
            block = [bone(in_channels, add=add, shuffle=shuffle, att=att)]
            if not tail:
                block.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))
            self.block = nn.Sequential(*block)
        else:
            block = []
            if not tail:
                block.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))
            block.append(bone(out_channels, add=add, shuffle=shuffle, att=att))
            self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

# TODO: modify
class FDPRG(nn.Module):
    def __init__(self, channels, kernel_size=3, bias=True, scale=2, shuffle=False, bone=E_RFDB, att=ESA, cbone=CFPB):  # n_RG=4
        super(FDPRG, self).__init__()
        
        self.scale = scale
        self.w0 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w0.data.fill_(1.0)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)

        self.m1 = bone(channels, shuffle=shuffle, att=att)
        self.w_m1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w_m1.data.fill_(1.0)

        self.m2 = bone(channels, shuffle=shuffle, att=att)
        self.w_m2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w_m2.data.fill_(1.0)

        self.m3 = bone(channels, shuffle=shuffle, att=att)
        self.w_m3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w_m3.data.fill_(1.0)
            
        self.cfpb = cbone(channels)

        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=bias)

    def forward(self, x):
        res1 = self.m1(x)
        res1 = res1 + self.w_m1 * x

        res2 = self.m2(res1)
        res2 = res2 + self.w_m2 * res1
        res2 = res2 + self.w0 * x

        res3 = self.m3(res2)
        res3 = res3 + self.w_m3 * res2
        out = res3 + self.w1 * res1 + self.w2 * x
        out = self.cfpb(out)
        out = self.conv(out)
        out += x
        return out

# Sampling method in ANRB, add softmax
class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2, soft=False):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])
        self.soft = soft

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = []
        for stage in self.stages:
            priors += F.softmax(stage(feats), dim=-1) if self.soft else stage(feats),
        center = torch.cat(priors, -1)
        return center

class ANRB(nn.Module):
    def __init__(self, in_channels, scale=1, psp_size=(1, 3, 6, 8)):
        super(ANRB, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)

        self.psp = PSPModule(psp_size)

        self.W = nn.Conv2d(in_channels=1, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # query: Nx1xHxW -> Nx1xHW
        query = self.f_query(x).view(batch_size, 1, -1)
        # Nx1xHW -> NxHWx1
        query = query.permute(0, 2, 1)

        # key：Nx1xS
        key = self.f_key(x)
        key = self.psp(key)

        # value: Nx1xHW -> Nx1xS （S = 110）
        value = self.psp(self.f_value(x))
        # Nx1xS -> NxSx1 （S = 110）
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        # no normalize?
        sim_map = (1 ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, 1, h, w)
        context = self.W(context)
        context += x
        return context

# sampling method use soft max
class ANRBsoft(nn.Module):
    def __init__(self, in_channels, scale=1, psp_size=(1, 3, 6, 8)):
        super(ANRBsoft, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)

        self.psp = PSPModule(psp_size, soft=True)

        self.W = nn.Conv2d(in_channels=1, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # query: Nx1xHxW -> Nx1xHW
        query = self.f_query(x).view(batch_size, 1, -1)
        # Nx1xHW -> NxHWx1
        query = query.permute(0, 2, 1)

        # key：Nx1xS
        key = self.f_key(x)
        key = self.psp(key)

        # value: Nx1xHW -> Nx1xS （S = 110）
        value = self.psp(self.f_value(x))
        # Nx1xS -> NxSx1 （S = 110）
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, 1, h, w)
        context = self.W(context)
        context += x
        return context

class ANRB_Conv(nn.Module):
    def __init__(self, in_channels, psp_size=(1, 3, 6, 8)):
        super(ANRB_Conv, self).__init__()
        self.in_channels = in_channels
        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=7)
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=7)
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=7)

        self.W = nn.Conv2d(in_channels=1, out_channels=self.in_channels, kernel_size=1)
        self.init_weights()

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        # query: Nx1xHxW -> Nx1xHW
        query = self.f_query(x).view(batch_size, 1, -1)
        # Nx1xHW -> NxHWx1
        query = query.permute(0, 2, 1)

        # key：Nx1xHW
        key = self.f_key(x)

        # value: Nx1xHW -> NxHWx1
        value = self.f_value(x)
        value = value.view(batch_size, 1, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        # no normalize?
        sim_map = (1 ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, 1, h, w)
        context = self.W(context)
        context = F.interpolate(context, (h, w), mode='bilinear', align_corners=False) 
        context += x
        return context
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class ANRB_P(nn.Module):
    def __init__(self, in_channels, scale=1, psp_size=(1, 3, 6, 8)):
        super(ANRB_P, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)

        self.psp = PSPModule(psp_size)
        self.proj = nn.Linear(110, 110)

        self.W = nn.Conv2d(in_channels=1, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # query: Nx1xHxW -> Nx1xHW
        query = self.f_query(x).view(batch_size, 1, -1)
        # Nx1xHW -> NxHWx1
        query = query.permute(0, 2, 1)

        # key：Nx1xS
        key = self.f_key(x)
        key = self.psp(key)

        # value: Nx1xHW -> Nx1xS （S = 110）
        value = self.psp(self.f_value(x))
        value = self.proj(value)
        # Nx1xS -> NxSx1 （S = 110）
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        # no normalize?
        sim_map = (1 ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, 1, h, w)
        context = self.W(context)
        context += x
        return context


# class ACMLP(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding_mode='zeros'):
#         super(ACMLP, self).__init__()
#         self.padding = kernel_size // 2

#         center_offset_from_origin_border = self.padding - kernel_size // 2
#         ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
#         hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
#         if center_offset_from_origin_border >= 0:
#             self.ver_conv_crop_layer = nn.Identity()
#             ver_conv_padding = ver_pad_or_crop
#             self.hor_conv_crop_layer = nn.Identity()
#             hor_conv_padding = hor_pad_or_crop
#         else:
#             self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
#             ver_conv_padding = (0, 0)
#             self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
#             hor_conv_padding = (0, 0)

#         # convert to 1
#         self.reduction = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
#         self.expansion = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=1)

#         self.ver_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_size, 1),
#                                     stride=stride,
#                                     padding='same', dilation=dilation, groups=groups, bias=True,
#                                     padding_mode=padding_mode)

#         self.hor_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, kernel_size),
#                                     stride=stride,
#                                     padding='same', dilation=dilation, groups=groups, bias=True,
#                                     padding_mode=padding_mode)

#         self.fuse = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1,
#                                     stride=stride,
#                                     padding='same', dilation=dilation, groups=groups, bias=True,
#                                     padding_mode=padding_mode)

#     def forward(self, input):
#         x = self.reduction(input)

#         vertical_outputs = self.ver_conv_crop_layer(x)
#         vertical_outputs = self.ver_conv(vertical_outputs)

#         horizontal_outputs = self.hor_conv_crop_layer(x)
#         horizontal_outputs = self.hor_conv(horizontal_outputs)
#         x_fuse = torch.cat([vertical_outputs, horizontal_outputs, x], dim = 1)
#         out = self.fuse(x_fuse)

#         return self.expansion(out)


class ASMLP(nn.Module):
    def __init__(self, in_channels, scale=1, psp_size=(1, 3, 6, 8)):
        super(ASMLP, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        # self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.reduction = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1)
        self.psp = PSPModule(psp_size)
        self.proj_h = nn.Linear(110, 110)
        self.proj_w = nn.Linear(110, 110)
        self.fuse = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.expansion = nn.Conv2d(in_channels=1, out_channels=self.in_channels, kernel_size=1)

    def forward(self, x):
        # b c h w
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        # if self.scale > 1:
        #     x = self.pool(x)
        x = self.reduction(x)

        # convert to low dimension
        w_psp = self.psp(x)
        h_psp = self.psp(x.permute(0,1,3,2))

        # sparse-attention
        x_h = self.proj_h(h_psp)
        x_w = self.proj_w(w_psp)

        x_fuse = torch.cat([x_h.expand(x), x_w.expand(x), x], dim = 1)
        out = self.fuse(x_fuse)

        return self.expansion(x)

# add PSP, channel to 1
class DoubleAttention(nn.Module):
    def __init__(self, in_channels, psp_size=(1, 3, 6, 8)):
        super().__init__()
        self.in_channels = in_channels
        self.convA = nn.Conv2d(in_channels, 1, 1)
        self.convB = nn.Conv2d(in_channels, 1, 1)
        self.convV = nn.Conv2d(in_channels, 1, 1)
        self.psp = PSPModule(psp_size)
        self.conv_reconstruct = nn.Conv2d(1, in_channels, kernel_size = 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        # b 1 h w -> b 1 s
        A = self.psp(A)

        # b 1 h w -> b 1 s
        B = self.psp(B)

        attention_maps = F.softmax(B, dim=-1)
        attention_vectors = F.softmax(V.view(b, 1, -1), dim=-1)

        # step 1: feature gating
        # b 1 s x b s 1 -> b 1 1
        global_descriptors = torch.bmm(A, attention_maps.permute(0, 2, 1))

        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors)

        tmpZ = tmpZ.view(b, 1, h, w) # b, c_m, h, w
        tmpZ = self.conv_reconstruct(tmpZ)

        return x + tmpZ 

