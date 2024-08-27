from Xception import efficient
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn,einsum
import torchvision
import numpy as np
import torch.nn.functional as F
from myutils.utils import SRMConv2d_simple,SRMConv2d_Separate
from typing import Tuple
from einops import rearrange
from torch import Tensor
from PIL import Image
from dataset.util import GradCAM
import cv2

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def generate_filter_gasuss(start, end, size):
    d0 = end
    n = 2
    template= np.zeros((size,size), dtype=np.float32)
    r, c = size,size
    for i in np.arange(r):
        for j in np.arange(c):
            # if i + j < end / 2:
            #     template[i ,j] = 1
            # else:
            distance = np.sqrt(i ** 2 + j ** 2)
            template[i, j] = 1 / (1 + (distance / d0) ** (2 * n))
            # template[i, j] = np.e ** (-1 * (distance ** 2 / (2 * d0 ** 2)))

    template = 1 - template
    return template

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

class norm_Filter(nn.Module):
    def __init__(self, size,
                 band_start,
                 band_end,
                 use_learnable=True,
                 norm=False):
        super(norm_Filter, self).__init__()
        self.use_learnable = use_learnable
        self.s = band_end
        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class DCT(nn.Module):
    def __init__(self,img_size,m):
        super(DCT, self).__init__()
        self.img_size = img_size
        self.m = m

        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(img_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(img_size)).float(), 0, 1), requires_grad=False)

        high_filter = norm_Filter(img_size, img_size // 4, img_size * 2,use_learnable=True)

        self.filters1 = nn.ModuleList([high_filter])

    def forward(self,x):
        x_dct = self._DCT_patch @ x @ self._DCT_patch_T

        idct_list = []
        for i in range(self.m):
            x = self.filters1[i](x_dct)
            x = self._DCT_patch_T @ x @ self._DCT_patch
            idct_list.append(x)

        idct = torch.cat(idct_list,dim=1)#[b,6,h,w]
        return idct

class block_DCT(nn.Module):
    def __init__(self,img_size,windows_size,m):
        super(block_DCT, self).__init__()
        self.img_size = img_size
        self.m = m
        self.windows_size = windows_size

        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(windows_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(windows_size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(windows_size,windows_size),stride=2)
        self.fold = nn.Fold(output_size=(img_size,img_size),kernel_size=(windows_size,windows_size),stride=2)

        high_filter = norm_Filter(windows_size, windows_size // 3, windows_size * 2,use_learnable=False)

        self.filters1 = nn.ModuleList([high_filter])

    def forward(self,x):
        b,c,h,w = x.shape
        x = self.unfold(x) # [b,w*w*c,l]
        l = x.shape[2]
        x = x.transpose(-2,-1).reshape(b,l,c,self.windows_size,self.windows_size) #b l c s s

        x_dct = self._DCT_patch @ x @ self._DCT_patch_T

        idct_list = []
        for i in range(self.m):
            x = self.filters1[i](x_dct)
            x = self._DCT_patch_T @ x @ self._DCT_patch
            x = x.reshape(b,l,-1).transpose(-2,-1)
            x = self.fold(x)
            idct_list.append(x)

        idct = torch.cat(idct_list,dim=1)#[b,6,h,w]
        return idct

class Global_Attention(nn.Module):
    def __init__(self,dim,head_dim,k):
        super(Global_Attention, self).__init__()
        self.head_num = dim // head_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q = nn.Conv2d(dim,dim,1)
        self.sr = nn.Conv2d(dim,dim,kernel_size=k,stride=k)
        # self.sr = nn.AvgPool2d(kernel_size=k,stride=k)
        self.norm = nn.BatchNorm2d(dim)

        self.kv = nn.Conv2d(dim,dim*2,1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim,dim,3,padding=1,bias=False),
            nn.BatchNorm2d(dim)
            # nn.LeakyReLU(0.2,True)
        )

    def forward(self,x):
        b,c,h,w = x.size()
        q = self.q(x).reshape(b,self.head_num,self.head_dim,-1).permute(0,1,3,2) # b c/d hw d
        x_ = self.norm(self.sr(x))
        kv = self.kv(x_).reshape(b,2,self.head_num,self.head_dim,-1) # b 2 c/d d hw
        kv = kv.permute(1,0,2,4,3)
        k,v = kv[0],kv[1]
        attn = (q * self.scale) @ k.transpose(-2,-1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(-2,-1).reshape(b,c,h,w)
        x = self.out_conv(x)
        return x

class mutil_Attention_rf(nn.Module):
    def __init__(self, dim, patch_size, drop=0.):
        super().__init__()
        self.head_dim = dim // len(patch_size)
        self.patch_size = patch_size
        self.scale = self.head_dim ** -0.5

        self.norm = nn.BatchNorm2d(dim)
        self.qv = nn.Conv2d(dim, dim * 2, 1)
        self.k = nn.Conv2d(dim,dim,1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim,dim,3,padding=1,bias=False),
            # nn.Conv2d(dim,dim,1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2,True)
        )

    def forward(self,rgb,fre):
        b,c,h,w = rgb.shape
        d_k = c // len(self.patch_size)
        attention = []
        qv = self.qv(self.norm(rgb)).reshape(b,2,c,h,w)
        qv = qv.permute(1,0,2,3,4)
        k = self.k(self.norm(fre))
        q,v = qv[0],qv[1]

        for (width, height), query, key, value,x_chunk in zip(
            self.patch_size,
            torch.chunk(q, len(self.patch_size), dim=1),
            torch.chunk(k, len(self.patch_size), dim=1),
            torch.chunk(v, len(self.patch_size), dim=1),
            torch.chunk(rgb, len(self.patch_size), dim=1),
        ):
            grid_h,grid_w = h // height, w // width
            query = query.reshape(b,d_k,grid_h,height,grid_w,width)
            query = query.permute(0,2,4,3,5,1)
            query = query.reshape(b,-1,width*height,d_k)

            key = key.reshape(b,d_k,grid_h,height,grid_w,width)
            key = key.permute(0,2,4,3,5,1)
            key = key.reshape(b,-1,width*height,d_k)

            value = value.reshape(b,d_k,grid_h,height,grid_w,width)
            value = value.permute(0,2,4,3,5,1)
            value = value.reshape(b,-1,width*height,d_k)

            attn = (query * self.scale) @ key.transpose(-2,-1)
            attn = attn.softmax(dim=-1)
            grid_x = (attn @ value).reshape(b,grid_h,grid_w,height,width,d_k)
            grid_x = grid_x.permute(0,5,1,3,2,4).reshape(b,d_k,h,w)
            grid_x = x_chunk + grid_x
            attention.append(grid_x)

        output = torch.cat(attention,dim=1)
        output = self.out_conv(output)
        return output

class mutil_Attention_ff(nn.Module):
    def __init__(self, dim, patch_size, drop=0.):
        super().__init__()
        self.head_dim = dim // len(patch_size)
        self.patch_size = patch_size
        self.scale = self.head_dim ** -0.5

        self.norm = nn.BatchNorm2d(dim)
        self.qk = nn.Conv2d(dim, dim * 2, 1)
        self.v = nn.Conv2d(dim,dim,1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim,dim,3,padding=1,bias=False),
            # nn.Conv2d(dim,dim,1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2,True)
        )

    def forward(self,rgb,fre):
        b,c,h,w = rgb.shape
        d_k = c // len(self.patch_size)
        attention = []
        qk = self.qk(self.norm(fre)).reshape(b,2,c,h,w)
        qk = qk.permute(1,0,2,3,4)
        v = self.v(self.norm(rgb))
        q,k = qk[0],qk[1]

        for (width, height), query, key, value,x_chunk in zip(
            self.patch_size,
            torch.chunk(q, len(self.patch_size), dim=1),
            torch.chunk(k, len(self.patch_size), dim=1),
            torch.chunk(v, len(self.patch_size), dim=1),
            torch.chunk(rgb, len(self.patch_size), dim=1),
        ):
            grid_h,grid_w = h // height, w // width
            query = query.reshape(b,d_k,grid_h,height,grid_w,width)
            query = query.permute(0,2,4,3,5,1)
            query = query.reshape(b,-1,width*height,d_k)

            key = key.reshape(b,d_k,grid_h,height,grid_w,width)
            key = key.permute(0,2,4,3,5,1)
            key = key.reshape(b,-1,width*height,d_k)

            value = value.reshape(b,d_k,grid_h,height,grid_w,width)
            value = value.permute(0,2,4,3,5,1)
            value = value.reshape(b,-1,width*height,d_k)

            attn = (query * self.scale) @ key.transpose(-2,-1)
            attn = attn.softmax(dim=-1)

            grid_x = (attn @ value).reshape(b,grid_h,grid_w,height,width,d_k)
            grid_x = grid_x.permute(0,5,1,3,2,4).reshape(b,d_k,h,w)
            grid_x = x_chunk + grid_x
            attention.append(grid_x)

        output = torch.cat(attention,dim=1)
        output = self.out_conv(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=2, dilation=2
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,dim):
        super(TransformerBlock, self).__init__()
        patch_size = [
            (28,28),
            (14,14),
            (8,8),
            (7,7)
        ]
        self.local_attention1 = mutil_Attention_rf(dim,patch_size)
        self.local_attention2 = mutil_Attention_ff(dim,patch_size)
        self.global_attention = Global_Attention(dim,8,7)
        self.conv = nn.Sequential(
            nn.Conv2d(dim*2,dim,1,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim,dim,3,1,1,bias=False),
            nn.BatchNorm2d(dim)
        )
        self.feedforward1 = FeedForward(dim,dim)
        self.feedforward2 = FeedForward(dim,dim)

    def forward(self,rgb,fre):
        selfatt1 = self.local_attention1(rgb,fre)
        selfatt2 = self.local_attention2(rgb,fre)
        selfatt = self.conv(torch.cat([selfatt1,selfatt2],dim=1))
        out = rgb + selfatt
        out = out + self.feedforward1(out)

        selfatt = self.global_attention(out)
        out = out + selfatt
        out = out + self.feedforward2(out)
        return out

class FreBlock(nn.Module):
    def __init__(self,dim,size):
        super(FreBlock, self).__init__()
        self.dct = block_DCT(size,8,1)
        self.feedforward = FeedForward(dim,dim)

    def forward(self,x):
        x = x + self.feedforward(self.dct(x))
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class HIGHPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(HIGHPixelAttention, self).__init__()
        self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3,padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.pa = SpatialAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.srm(x)
        x = self.conv(x)
        att_map = self.pa(x)

        return att_map

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class FeatureFusionModel(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(FeatureFusionModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,1,1,0,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2,True)
        )
        self.ca = ChannelAttention(1792,16)
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self,rgb,noise):
        x = self.conv(torch.cat((rgb,noise),dim=1))
        x = self.ca(x)*x + x
        return x

class MixBlock(nn.Module):

    def __init__(self, c_in, width, height):
        super(MixBlock, self).__init__()
        self.FAD_query = nn.Conv2d(c_in, c_in, (1, 1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1, 1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1, 1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)  # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)  # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  # [BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))
        return y_FAD, y_LFS

class MSCA(nn.Module):
    def __init__(self,in_channels,ratio):
        super(MSCA, self).__init__()
        self.max_pool_3x3 = nn.AvgPool2d(3)
        self.max_pool_5x5 = nn.AvgPool2d(5)
        self.max_pool_7x7 = nn.AvgPool2d(7)
        self.gobal_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gobal_max_pool = nn.AdaptiveMaxPool2d(1)
        # self.conv = nn.Conv2d(3,1,1,bias=False)
        self.linear1 = nn.Linear(3,1,bias=False)
        self.linear2 = nn.Linear(3,1,bias=False)
        self.sigmoid = nn.Sigmoid()

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self,x):
        m3 = self.max_pool_3x3(x)
        m5 = self.max_pool_5x5(x)
        m7 = self.max_pool_7x7(x)

        p1,p2,p3 = self.gobal_avg_pool(m3),self.gobal_avg_pool(m5),self.gobal_avg_pool(m7)
        p4,p5,p6 = self.gobal_max_pool(m3),self.gobal_max_pool(m5),self.gobal_max_pool(m7)

        a_p = torch.cat([p1,p2,p3],dim=2).transpose(2,3)
        a_p = self.sharedMLP(self.linear1(a_p).transpose(2,3))

        m_p = torch.cat([p4,p5,p6],dim=2).transpose(2,3)
        m_p = self.sharedMLP(self.linear2(m_p).transpose(2,3))

        # p = self.linear(p).transpose(2,3)
        # x = x * p + x
        return self.sigmoid(a_p + m_p)

class doublefusion(nn.Module):
    def __init__(self,channel,size):
        super(doublefusion, self).__init__()
        self.mix = MixBlock(channel,size,size)
        self.ca = MSCA(channel*2,8)
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(channel,channel,1,groups=channel),
            nn.BatchNorm2d(channel)
        )
        self.fre_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1, groups=channel),
            nn.BatchNorm2d(channel)
        )

    def forward(self,rgb,fre):
        rgb,fre = self.mix(rgb,fre)
        r_c,f_c = self.ca(torch.cat([rgb,fre],dim=1)).chunk(2,1)
        rgb = rgb + self.rgb_conv(r_c * rgb)
        fre = fre + self.fre_conv(f_c * fre)
        return rgb,fre

class fu(nn.Module):
    def __init__(self,dim,size):
        super(fu, self).__init__()
        self.rgb_conv = nn.Conv2d(dim,dim,1)
        self.noise_conv = nn.Conv2d(dim,dim,1)
        self.size = size

    def forward(self,rgb,noise):
        out = self.rgb_conv(rgb) + self.noise_conv(noise)
        out = F.interpolate(out,size=(self.size,self.size),mode='bilinear')
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rgb_cnn = efficient.EfficientNet.from_pretrained('efficientnet-b4',num_classes=2)
        self.high_cnn = efficient.EfficientNet.from_pretrained('efficientnet-b4',num_classes=2)

        self.srm_1 = SRMConv2d_simple(3)
        self.srm_2 = SRMConv2d_Separate(48, 48)

        self.act = nn.ReLU()
        # self.att = HIGHPixelAttention(3)
        self.feedward = nn.Sequential(
            nn.BatchNorm2d(48),
            nn.ReLU(True)
        )

        self.fre = FreBlock(32,56)
        self.trans = TransformerBlock(32)

        self.fusion1 = doublefusion(112,14)
        self.fusion2 = doublefusion(160,14)
        self.out_fu = FeatureFusionModel(1792*2,1792)

        # self.rf1 = fu(32,14)
        # self.rf2 = fu(112,14)
        # self.rf3 = fu(160,14)
        self.last_linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1792, 2)
        )

    def classifier(self, x):
        x = self.act(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self,x):
        srm = self.srm_1(x)
        rgb = self.rgb_cnn.start_fea(x)

        noise = self.high_cnn.start_fea(srm) + self.srm_2(rgb)
        noise = self.act(noise)
        # att = self.att(x)
        # rgb = rgb*att + rgb
        # rgb = self.feedward(rgb)

        rgb = self.rgb_cnn.extra_fea_part_1(rgb)
        noise = self.high_cnn.extra_fea_part_1(noise)

        fre = self.fre(rgb)
        rgb = self.trans(rgb,fre)

        # fea1 = self.rf1(rgb,fre)

        rgb = self.rgb_cnn.extra_fea_part_2(rgb)
        noise = self.high_cnn.extra_fea_part_2(noise)
        rgb,noise = self.fusion1(rgb,noise)
        # fea2 = self.rf2(rgb,noise)

        rgb = self.rgb_cnn.extra_fea_part_3(rgb)
        noise = self.high_cnn.extra_fea_part_3(noise)
        rgb,noise = self.fusion2(rgb,noise)
        # fea3 = self.rf3(rgb,noise)

        # fea = torch.cat([fea1,fea2,fea3],dim=1)

        rgb = self.rgb_cnn.extra_fea_part_4(rgb)
        noise = self.high_cnn.extra_fea_part_4(noise)

        out = self.out_fu(rgb,noise)
        out = self.classifier(out)

        return out