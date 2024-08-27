from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from timm.models.layers import DropPath

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self,dim,drop_path):
        super(Block, self).__init__()
        self.conv = SeparableConv2d(dim,dim,7,padding=3)
        self.norm = LayerNorm(dim,eps=1e-6)
        self.boltt_conv = nn.Linear(dim,dim*4)
        self.act = nn.Hardswish(True)
        self.unboltt_conv = nn.Linear(dim*4,dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,x):
        input = x
        x = self.conv(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.boltt_conv(x)
        x = self.act(x)
        x = self.unboltt_conv(x)
        x = x.permute(0,3,1,2)
        x = input + self.drop_path(x)
        return x
    
class MyModule(nn.Module):
    def __init__(self,dim=[64,256,512,1024]):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.Ln1 = LayerNorm(32,1e-6)
        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.Ln2 = LayerNorm(64,1e-6)
        self.relu = nn.ReLU(True)

        self.block = nn.ModuleList()
        self.block1 = nn.Sequential(
            Block(dim[0],0.3),
            Block(dim[0], 0.3),
            Block(dim[0], 0.3)
        )
        self.block2 = nn.Sequential(
            Block(dim[1], 0.3),
            Block(dim[1], 0.3),
            Block(dim[1], 0.3)
        )
        self.block3_1 = nn.Sequential(
            Block(dim[2], 0.3),
            Block(dim[2], 0.3),
            Block(dim[2], 0.3),
            Block(dim[2], 0.3),
        )
        self.block3_2 = nn.Sequential(
            Block(dim[2], 0.3),
            Block(dim[2], 0.3),
            Block(dim[2], 0.3),
            Block(dim[2], 0.3),
            Block(dim[2], 0.3),
        )
        self.block4 = nn.Sequential(
            Block(dim[3], 0.3),
            Block(dim[3], 0.3),
            Block(dim[3], 0.3)
        )
        self.block.append(self.block1)
        self.block.append(self.block2)
        self.block.append(self.block3_1)
        self.block.append(self.block3_2)
        self.block.append(self.block4)
        self.down1 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=2,stride=2)
        )
        self.down_list = nn.ModuleList()
        self.down_list.append(self.down1)

        for i in range(3):
            down = nn.Sequential(
                LayerNorm(dim[i],eps=1e-6,data_format='channels_first'),
                nn.Conv2d(dim[i],dim[i+1],kernel_size=2,stride=2)
            )
            self.down_list.append(down)

    #32,111
    def fea_part_1_0(self,x):
        x = self.conv1(x)
        x = x.permute(0,2,3,1)
        x = self.Ln1(x)
        x = x.permute(0,3,1,2)
        x = self.relu(x)
        return x

    #64 109
    def fea_part_1_1(self,x):
        x = self.conv2(x)
        x = x.permute(0,2,3,1)
        x = self.Ln2(x)
        x = x.permute(0,3,1,2)
        x = self.relu(x)
        return x

    #64 54
    def fea_part_2(self,x):
        x = self.down_list[0](x)
        x = self.block[0](x)
        return x

    #256 27
    def fea_part_3(self,x):
        x = self.down_list[1](x)
        x = self.block[1](x)
        return x

    #512 13
    def fea_part_4(self,x):
        x = self.down_list[2](x)
        x = self.block[2](x)
        return x

    #512 13
    def fea_part_5(self,x):
        x = self.block[3](x)
        return x

    #1024 6
    def fea_part_6(self,x):
        x = self.down_list[3](x)
        x = self.block[4](x)
        return x