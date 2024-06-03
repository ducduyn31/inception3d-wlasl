from torch import nn


class S3D(nn.Module):
    def __init__(self):
        super(S3D, self).__init__()
        self.base = nn.Sequential(
            SeparableConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Conv3dUnit(64, 64, kernel_size=1, stride=1, padding=0),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

    def forward(self, x):
        return self.base(x)
    
class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv3d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=(0, padding, padding), bias=False),
            nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU()
        )
        self.pointwise = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False),
            nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Conv3dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv3dUnit, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

