import numpy as np
from torch import nn
from torch.nn import functional as F


class Conv3dBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
    ):
        super(Conv3dBlock, self).__init__()

        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_size[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_size[dim] - (s % self._stride[dim]), 0)

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

        x = self.conv3d(x)

        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x
