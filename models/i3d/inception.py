from torch import nn
import torch

from .conv3d import Conv3dBlock
from .maxpool3d import MaxPool3dDynamicPadding


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionBlock, self).__init__()

        self.b0 = Conv3dBlock(
            in_channels=in_channels,
            out_channels=out_channels[0],
            kernel_size=(1, 1, 1),
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Conv3dBlock(
            in_channels=in_channels,
            out_channels=out_channels[1],
            kernel_size=(1, 1, 1),
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Conv3dBlock(
            in_channels=out_channels[1],
            out_channels=out_channels[2],
            kernel_size=(3, 3, 3),
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Conv3dBlock(
            in_channels=in_channels,
            out_channels=out_channels[3],
            kernel_size=(1, 1, 1),
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Conv3dBlock(
            in_channels=out_channels[3],
            out_channels=out_channels[4],
            kernel_size=(3, 3, 3),
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dDynamicPadding(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.b3b = Conv3dBlock(
            in_channels=in_channels,
            out_channels=out_channels[5],
            kernel_size=(1, 1, 1),
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)
