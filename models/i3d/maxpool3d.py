import numpy as np
from torch import nn
from torch.nn import functional as F


class MaxPool3dDynamicPadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        (b, c, t, h, w) = x.size()

        pad_t = int(self.compute_pad(0, t))
        pad_h = int(self.compute_pad(1, h))
        pad_w = int(self.compute_pad(2, w))

        pad_t_low = pad_t // 2
        pad_t_high = pad_t - pad_t_low
        pad_h_low = pad_h // 2
        pad_h_high = pad_h - pad_h_low
        pad_w_low = pad_w // 2
        pad_w_high = pad_w - pad_w_low

        pad = (pad_w_low, pad_w_high, pad_h_low, pad_h_high, pad_t_low, pad_t_high)

        x = F.pad(x, pad)
        return super(MaxPool3dDynamicPadding, self).forward(x)
