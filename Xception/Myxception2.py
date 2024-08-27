from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
# from blocks.inception_blocks import Inception_A_res, Inception_B_res, Inception_C_res

__all__ = ['xception']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975
            # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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


class Block(nn.Module): # 步长不等于1就用最大池化
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []

        filters = in_filters

        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)
        # self.se = SE_Block(out_filters)

    def forward(self, inp):
        # 此时x2.shape=torch.Size([40, 64, 109, 109])
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        # print("x1.shape={}".format(x.shape)) #x1.shape=torch.Size([40, 128, 55, 55])
        # print("skip.shape={}".format(skip.shape)) skip.shape=torch.Size([40, 128, 55, 55])
        # x = self.se(x)
        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, zero_init_residual=False):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.cuda()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)

        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 1, start_with_relu=True, grow_first=False)
        self.bn12 = nn.BatchNorm2d(1024)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        # self.relu3 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = 2048

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.last_linear = nn.Linear(2048, num_classes)
        self.avgpool_2 = nn.AdaptiveAvgPool2d((2, 2))

        # self.fc_clu = nn.Linear(512 * 4, 31) #4是expansion，31是类数
        # self.fc_loc = nn.Linear(512 * 4, 4)
        # self.scale = 1

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        # self.fc = nn.Linear(512, num_classes)
        for dep in range(2):
            self.head[dep * 3].weight.data.normal_(0, 0.01)
            self.head[dep * 3].bias.data.fill_(0.0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------
    def fea_part1_0(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

    def fea_part1_1(self,x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    def fea_part_1(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

    def fea_part2(self,x):
        x = self.block1(x) #[128,56,56]
        return x

    def fea_part3(self,x):
        x = self.block2(x) #256 28 28
        return x

    def fea_part4(self,x):
        x = self.block3(x) # 728 14 14
        return x

    def fea_part5(self, x):
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x

    def fea_part6(self,x):
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.bn12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x
    # def fea_part3(self,x):
    #     x = self.block4(x)
    #     x = self.block5(x)
    #     x = self.block6(x)
    #     x = self.block7(x)
    #     return x
    #
    # def fea_part4(self, x):
    #     x = self.block8(x)
    #     x = self.block9(x)
    #     x = self.block10(x)
    #     x = self.block11(x)
    #
    #     return x
    #
    # def fea_part5(self, x):
    #     x = self.block12(x)
    #
    #     x = self.conv3(x)
    #     x = self.bn3(x)
    #     x = self.relu(x)
    #
    #     x = self.conv4(x)
    #     x = self.bn4(x)
    #
    #     return x

    def features(self, input):
        x = self.conv1(input) #[b,32,111,111]
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x) #[b,64,109,109]
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.block1(x)#[b,128,55,55]
        x = self.block2(x) #[b,256,28,28]
        low_feature = self.myconv(x)
        x = self.block3(x) #[b,728,14,14]
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        return x

    def interpolate(self, x):
        x = self.features(x)
        x = F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)
        x = self.avgpool_2(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)
        """x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)"""
        # x = self.bottleneck(x)
        # x = self.head(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        low_feature,x = self.features(input) #只用features 2048维
        return low_feature,x


def get_xception(num_classes=2, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        pretrained = 'imagenet'
        settings = pretrained_settings['xception'][pretrained]
        """assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)"""

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']), strict=False)

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
        model.requires_grad_(True)
    # TODO: ugly
    # model.last_linear = model.fc
    # del model.fc
    return model