from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class SurfaceNormalDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=3, use_skips=True):
        super(SurfaceNormalDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dconv", s)]  = Conv3x3(self.num_ch_dec[s], 8)
            self.convs[("dispconv", s)] = Conv3x3((4-s)*8, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.dfeats  = []
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.dfeats.append(self.convs[("dconv", i)](x))
        
        self.dfeats.reverse()

        for s in self.scales:
            dcfeats = []
            up = 2 ** (3-s)
            for i in range(3, s-1, -1):
                if i == s:
                    dcfeats.append(self.dfeats[i])
                else:
                    dcfeats.append(upsample(self.dfeats[i], sf=up))
                    up /= 2
            dcfeats = torch.cat((dcfeats), 1)
            self.outputs[("disp_surface_normal", s)] = self.sigmoid(self.convs[("dispconv", s)](dcfeats))

        return self.outputs
