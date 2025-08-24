import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# 2×1 卷积模块
class Conv_2x1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv_2x1, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=(2, 1), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


# 深度可分离卷积模块
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, dilation,
                                        groups=in_channel)
        self.pointwise_conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# Enhanced Channel Attention Module
class EChannel_attn(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(EChannel_attn, self).__init__()
        self.gate_channels = gate_channels
        self.conv2_1 = Conv_2x1(gate_channels, gate_channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1, padding=0, bias=True),
        )

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # (b, c, 1, 1)
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # (b, c, 1, 1)
        x = torch.cat((avg_pool, max_pool), dim=2)
        x = self.conv2_1(x)
        x = self.mlp(x)
        return x


# Enhanced Spatial Attention Module
class ESpatial_attn(nn.Module):
    def __init__(self, dim):
        super(ESpatial_attn, self).__init__()
        self.conv1 = DepthwiseConv2d(dim, dim, kernel_size=3, stride=2, padding=2, dilation=2)
        self.conv2 = DepthwiseConv2d(dim, dim, kernel_size=3, stride=2, padding=4, dilation=4)
        self.conv3 = DepthwiseConv2d(dim * 2, 1, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        x = torch.cat((out1, out2), dim=1)
        x = self.conv3(x)
        new_height = x.size(2) * 2 - 1
        new_width = x.size(3) * 2 - 1
        x = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)
        return x



class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class EMSAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(EMSAFusion, self).__init__()
        self.sa = ESpatial_attn(dim)
        self.ca = EChannel_attn(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
