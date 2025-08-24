import torch
import math
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from .AKPPA import AKPPA

class RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        # nn.init.zeros_(self.convmap.weight)
        self.bias = None
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups,
                        bias=self.bias)


class Repblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(Repblock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            RepConv(in_channels, in_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channels),
        )
    # 定义前向函数
    def forward(self, x):
        out = self.features(x)
        out = out + self.shortcut(x)
        out = torch.relu(out)
        return out

class CNNblock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.RefLKA = Repblock(in_channels,out_channels)
        self.AKPPA = AKPPA(in_features = out_channels,filters = out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.RefLKA(x)
        out = self.AKPPA(x)
        out = self.maxpool(out)
        return out

class CNNnet_branch(nn.Module):
    def __init__(self, verbose=False, embed_dims=[64, 128, 256, 512]):
        super(CNNnet_branch, self).__init__()
        self.verbose = verbose
        # 特征提取层
        # 这里采用3×3卷积核
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.block1 = CNNblock(3, embed_dims[0])
        self.block2 = CNNblock(embed_dims[0], embed_dims[1])
        self.block3 = CNNblock(embed_dims[1], embed_dims[2])
        self.block4 = CNNblock(embed_dims[2], embed_dims[3])

    def forward(self, x):
        out = self.features(x)
        if self.verbose:
            print('features output: {}'.format(out.shape))
        out = self.block1(out)
        if self.verbose:
            print('block 1 output: {}'.format(out.shape))
        out = self.block2(out)
        if self.verbose:
            print('block 2 output: {}'.format(out.shape))
        out = self.block3(out)
        if self.verbose:
            print('block 3 output: {}'.format(out.shape))
        out = self.block4(out)
        if self.verbose:
            print('block 4 output: {}'.format(out.shape))
        return out


if __name__ == '__main__':
    device = torch.device('cuda')
    block = CNNnet_branch().to(device)
    input = torch.rand(64, 3, 224, 224).cuda()
    output = block(input)
    print(output.size())
