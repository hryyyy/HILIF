import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.metrics import average_precision_score,accuracy_score,roc_curve,auc
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray

def gen_cons_conv_weight(shape,in_channel,out_channel):
        center = int(shape / 2)
        accumulation = 0
        for i in range(shape):
            for j in range(shape):
                if i != center or j != center:
                    dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
                    accumulation += 1 / dis

        base = 1 / accumulation
        arr = torch.zeros((shape, shape), requires_grad=False)
        for i in range(shape):
            for j in range(shape):
                if i != center or j != center:
                    dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
                    arr[i][j] = base / dis
        arr[center][center] = -1
        # print(arr.unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1).shape,arr.unsqueeze(0).unsqueeze(0).shape)
        return arr.unsqueeze(0).unsqueeze(0).repeat(out_channel, in_channel, 1, 1)

class BayarConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5,stride=1,padding=0,requires_grad=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels,self.out_channels,1)* - 1.000)

        super(BayarConv2d, self).__init__()
        # self.kernel = nn.Parameter(gen_cons_conv_weight(5,in_channels,out_channels),requires_grad=requires_grad)
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                  requires_grad=requires_grad)

    def bayarConstraint(self):
        # self.kernel.data = set_constrain(self.kernel.data)
        self.kernel.data = self.kernel.permute(2,0,1)
        self.kernel.data = torch.div(self.kernel.data,self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1,2,0)
        ctr = self.kernel_size ** 2 // 2

        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]),
                                dim=2) #[inc,outc,12],[inc,outc,1],[inc,outc,12]
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self,x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

class new_BayarConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5,stride=1,padding=0,requires_grad=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels,self.out_channels,1)* - 1.000)

        super(new_BayarConv2d, self).__init__()
        self.kernel = nn.Parameter(gen_cons_conv_weight(5,in_channels,out_channels),requires_grad=requires_grad)
        # self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
        #                           requires_grad=requires_grad)

    def bayarConstraint(self):
        self.kernel.data = set_constrain(self.kernel.data)
        # self.kernel.data = self.kernel.permute(2,0,1)
        # self.kernel.data = torch.div(self.kernel.data,self.kernel.data.sum(0))
        # self.kernel.data = self.kernel.permute(1,2,0)
        # ctr = self.kernel_size ** 2 // 2
        # real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]),
        #                         dim=2)
        # real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return self.kernel

    def forward(self,x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x

class SRMConv2d_simple(nn.Module):

    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)
        # torchvision.utils.save_image(out,'out.jpg')
        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters

class SRMConv2d_Separate(nn.Module):

    def __init__(self, inc, outc, learnable=False):
        super(SRMConv2d_Separate, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3 * inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        out = self.truc(out)
        out = self.out_conv(out)
        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        # filters = np.repeat(filters, inc, axis=1)
        filters = np.repeat(filters, inc, axis=0)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        # print(filters.size())
        return filters

def set_constrain(weight):
    center = int(weight.shape[2] / 2)

    weight[:, :, center, center] = 0
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            idx_positive = weight[i, j, :, :] >= 0
            idx_negative = weight[i, j, :, :] < 0

            abs_sum = torch.abs(weight[i, j, :, :]).sum()

            if idx_positive.any():
                weight[i, j, idx_positive] = weight[i, j, idx_positive] / abs_sum
                weight[i, j, weight[i, j, :, :] < 0.001] = 0.001

            if idx_negative.any():
                weight[i, j, idx_negative] = 0.001

            weight[i, j, center, center] = -(weight[i, j, :, :].sum())
    return weight

class learn_constrain_simple(nn.Module):
    def __init__(self,inc):
        super(learn_constrain_simple, self).__init__()
        self.truc = nn.Hardtanh(-3,3)
        kernel = self.gen_cons_conv_weight(5)
        self.inc = inc
        self.kernel = nn.Parameter(data=kernel,requires_grad=True)

    def forward(self,x):
        self.kernel.data = set_constrain(self.kernel.data)
        out = F.conv2d(x,self.kernel,stride=1,padding=2,bias=None)
        out = self.truc(out)
        return out

    def gen_cons_conv_weight(self,shape):
        center = int(shape / 2)
        accumulation = 0
        for i in range(shape):
            for j in range(shape):
                if i != center or j != center:
                    dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
                    accumulation += 1 / dis

        base = 1 / accumulation
        arr = torch.zeros((shape, shape), requires_grad=False)
        for i in range(shape):
            for j in range(shape):
                if i != center or j != center:
                    dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
                    arr[i][j] = base / dis
        arr[center][center] = -1
        arr = arr.unsqueeze(0).unsqueeze(0)
        arr = arr.repeat(3, 3, 1, 1)
        return arr

class learn_constrain_Separate(nn.Module):
    def __init__(self, inc,outc):
        super(learn_constrain_Separate, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self.gen_cons_conv_weight(5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*inc,outc,1,1,0,1,1,bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(True)
        )

        for ly in self.out_conv.children():
            if isinstance(ly,nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight,a=1)

    def forward(self, x):
        self.kernel.data = set_constrain(self.kernel.data)
        out = F.conv2d(x, self.kernel, stride=1, padding=2,groups=self.inc)
        out = self.truc(out)
        out = self.out_conv(out)
        return out

    def gen_cons_conv_weight(self, shape):
        center = int(shape / 2)
        accumulation = 0
        for i in range(shape):
            for j in range(shape):
                if i != center or j != center:
                    dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
                    accumulation += 1 / dis

        base = 1 / accumulation
        arr = torch.zeros((shape, shape), requires_grad=False)
        for i in range(shape):
            for j in range(shape):
                if i != center or j != center:
                    dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
                    arr[i][j] = base / dis
        arr[center][center] = -1
        arr = arr.unsqueeze(0).unsqueeze(0)
        arr = arr.repeat(self.inc * 3, 1, 1, 1)
        return arr

class learn_SRMConv2d_simple(nn.Module):

    def __init__(self, inc=3, learnable=True):
        super(learn_SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.inc = inc
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        print(self.kernel.data)
        self.kernel.data = set_constrain(self.kernel.data)
        print(self.kernel.data)
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        torchvision.utils.save_image(out,'out1.jpg')
        out = self.truc(out)
        return out

    def _build_kernel(self, inc):
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters

class learn_SRMConv2d_Separate(nn.Module):
    def __init__(self, inc, outc, learnable=True):
        super(learn_SRMConv2d_Separate, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3 * inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        self.kernel.data = set_constrain(self.kernel.data)
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        out = self.truc(out)
        out = self.out_conv(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        # filters = np.repeat(filters, inc, axis=1)
        filters = np.repeat(filters, inc, axis=0)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        # print(filters.size())
        return filters

class BayarConv2d_simple(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5,stride=1,padding=0,requires_grad=True):
        super(BayarConv2d_simple, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.truc = nn.Hardtanh(-3, 3)
        self.minus1 = (torch.ones(self.in_channels,self.out_channels,1)* - 1.000)
        # self.kernel = nn.Parameter(gen_cons_conv_weight(5,in_channels,out_channels),requires_grad=requires_grad)
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                  requires_grad=requires_grad)

    def bayarConstraint(self):
        # self.kernel.data = set_constrain(self.kernel.data)
        self.kernel.data = self.kernel.permute(2,0,1)
        self.kernel.data = torch.div(self.kernel.data,self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1,2,0)
        ctr = self.kernel_size ** 2 // 2

        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]),
                                dim=2) #[inc,outc,12],[inc,outc,1],[inc,outc,12]
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self,x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        x = self.truc(x)
        return x

class BayarConv2d_Separate(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=5,stride=1,padding=0,requires_grad=True):
        super(BayarConv2d_Separate, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.truc = nn.Hardtanh(-3, 3)
        self.minus1 = (torch.ones(3,1,1)* - 1.000)
        # self.kernel = nn.Parameter(gen_cons_conv_weight(5,in_channels,out_channels),requires_grad=requires_grad)
        self.kernel = nn.Parameter(torch.rand(3, 1, kernel_size ** 2 - 1),
                                  requires_grad=requires_grad)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*self.in_channels, self.out_channels, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def bayarConstraint(self):
        # self.kernel.data = set_constrain(self.kernel.data)
        self.kernel.data = self.kernel.permute(2,0,1)
        self.kernel.data = torch.div(self.kernel.data,self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1,2,0)
        ctr = self.kernel_size ** 2 // 2

        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]),
                                dim=2) #[inc,outc,12],[inc,outc,1],[inc,outc,12]
        real_kernel = real_kernel.reshape((3, 1, self.kernel_size, self.kernel_size))
        real_kernel = real_kernel.repeat(self.in_channels,1,1,1)
        return real_kernel

    def forward(self,x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding,groups=self.in_channels)
        x = self.truc(x)
        x = self.out_conv(x)
        return x

class PEConv(nn.Module):
    def __init__(self,inc=3):
        super(PEConv, self).__init__()
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self,x):
        out = F.conv2d(x, self.kernel, stride=1,padding=1)
        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[ 0, 0, 0],
                   [0, -1, 0],
                   [0, 1, 0]]
        # filter2：KV
        filter2 = [[ 0, 0, 0],
                   [0, -1, 1],
                   [0, 0, 0]]
        # filter3：hor 2rd
        filter3 = [[ 0, 0, 0],
                   [0, -1, 0],
                   [0, 0, 1]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters

class Deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=1):
        super(Deconv, self).__init__()
        self.conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True
        )
        out = self.conv(x)
        out = self.leaky_relu(out)
        return out

def Butterworth(src, d0, n, ftype):
    template = np.zeros((src.shape[2],src.shape[3]), dtype=np.float32)
    b,C,r, c = src.shape
    for i in np.arange(r):
        for j in np.arange(c):
            distance = np.sqrt((i - r/2)**2 + (j - c/2)**2)
            # template[i, j] = 1/(1 + (distance/d0)**(2*n))
            template[i, j] = np.e ** (-1 * (distance**2 / (2 * d0**2)))
    if ftype == 'high':
        template = 1 - template
    return template

def block_img_dct(img_f32):
    b = img_f32.shape[0]
    height,width = img_f32.shape[2:]#[b,c=1,h,w]
    img_f32 = img_f32.squeeze()#[b,h,w]

    block_y = height // 8
    block_x = width // 8
    height_ = block_y * 8
    width_ = block_x * 8
    total_img = []
    if b != 1:
        for i in range(b):
            img_f32_cut = img_f32[i,:height_, :width_]
            img_dct = np.zeros((height_, width_), dtype=np.float32)
            for h in range(block_y):
                for w in range(block_x):
                    img_block = img_f32_cut[8*h: 8*(h+1), 8*w: 8*(w+1)].cpu().numpy()
                    img_dct[8*h: 8*(h+1), 8*w: 8*(w+1)] = cv2.dct(img_block)
            img_dct_log2 = np.log(abs(img_dct)+1e-6)
            total_img.append(img_dct_log2)
    else:
        img_f32_cut = img_f32[:height_, :width_]
        img_dct = np.zeros((height_, width_), dtype=np.float32)
        for h in range(block_y):
            for w in range(block_x):
                img_block = img_f32_cut[8*h: 8*(h+1), 8*w: 8*(w+1)].cpu().numpy()
                img_dct[8*h: 8*(h+1), 8*w: 8*(w+1)] = cv2.dct(img_block)
        img_dct_log2 = np.log(abs(img_dct)+1e-6)
        total_img.append(img_dct_log2)
    return torch.Tensor(np.array(total_img)).unsqueeze(1)

def whole_img_dct(img_f32):
    b = img_f32.shape[0]
    img_f32 = rgb2gray(img_f32)
    dct_list = []
    for i in range(b):
        img = np.array(img_f32[i])
        img_dct = cv2.dct(img.squeeze())
        img_dct_log = np.log(abs(img_dct))

        mask = np.ones(img_dct_log.shape)
        points = np.array([(0,0),(0,int(224/4)),(int(224/4),0)],np.int32)
        mask = cv2.fillPoly(mask,[points],(0,0,0))

        xiangwei = np.angle(img_dct)
        img = cv2.idct(img_dct * mask)
        dct_list.append(img)
    return torch.Tensor(np.array(dct_list)).unsqueeze(1),mask,xiangwei

def whole_img_dft(img_f32):
    b = img_f32.shape[0]
    img_f32 = rgb2gray(img_f32)
    dct_list = []
    for i in range(b):
        img = np.array(img_f32[i])
        img_dct = cv2.dft(img.squeeze())
        img_dct_log = np.log(abs(img_dct))
        xiangwei = np.angle(img_dct)
        print(np.var(xiangwei))

        mask = np.ones(img_dct_log.shape)
        points = np.array([(0,0),(0,int(224/4)),(int(224/4),0)],np.int32)
        mask = cv2.fillPoly(mask,[points],(0,0,0))
        img = cv2.idft(img_dct * mask)
        dct_list.append(img)
    return torch.Tensor(np.array(dct_list)).unsqueeze(1),mask,xiangwei

def caleval(y_true_all,y_pred_all):
    y_true_all,y_pred_all = np.array(y_true_all.cpu()),np.array(y_pred_all.cpu())

    fprs,tprs,ths = roc_curve(y_true_all,y_pred_all,pos_label=1,drop_intermediate=False)

    acc = accuracy_score(y_true_all,np.where(y_pred_all >= 0.5,1,0)) * 100
    ind = 0
    for fpr in fprs:
        if fpr > 1e-2:
            break
        ind += 1
    TPR_2 = tprs[ind - 1]

    ind = 0
    for fpr in fprs:
        if fpr > 1e-3:
            break
        ind += 1
    TPR_3 = tprs[ind - 1]

    ind = 0
    for fpr in fprs:
        if fpr > 1e-4:
            break
        ind += 1
    TPR_4 = tprs[ind - 1]

    ap = average_precision_score(y_true_all, y_pred_all)
    return ap, acc, auc(fprs, tprs), TPR_2, TPR_3, TPR_4

class dist_loss(nn.Module):
    def __init__(self):
        super(dist_loss, self).__init__()

    def forward(self,pred,target):
        batch = target.shape[0]
        t = batch // 2
        real = fake = 0
        for i in range(0,batch):
            if target[i] == 0:
                real = real + pred[i]
            elif target[i] == 1:
                fake = fake + pred[i]
        loss = 1 - (fake/t) + (real/t)
        loss = 1 - (fake / t) + (real / t)
        loss = torch.where(loss > 0, loss, torch.zeros_like(loss))
        return loss