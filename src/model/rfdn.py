import torch
import torch.nn as nn

from model import common
import model.block_rfdn as B

def make_model(args, parent=False):
    model = RFDN(args)
    return model

class RFDN(nn.Module):
    def __init__(self, args, num_modules=6):
        super(RFDN, self).__init__()

        in_nc = 3
        out_nc = 3
        nf = args.n_feats
        scale = args.scale[0]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # head
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        # body
        self.block = nn.ModuleList([B.E_RFDB(in_channels=nf) for _ in range(num_modules)])
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        # tail
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, scale)

        self.scale_idx = 0


    def forward(self, input):
        input = self.sub_mean(input)

        out_fea = self.fea_conv(input)
        out_ = [self.block[0](out_fea)]
        for i, b in enumerate(self.block[1:]):
            out_ += b(out_[-1]), 

        out_B = self.c(torch.cat(out_, dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)
        output = self.add_mean(output)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
    

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