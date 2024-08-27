import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from typing import Tuple

class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.35,
                 s=30):
        super(AMSoftmax, self).__init__()
        # self.cuda()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0] #batch
        assert x.size()[1] == self.in_feats # feature.shape
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)

        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()

        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)  # weights normed
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)  x.dot(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)  #
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta = cos_theta * xlen.view(-1, 1)

        return cos_theta  # size=(B,Classnum,1)

class AMSoftmax2(nn.Module):
    def __init__(self):
        super(AMSoftmax2, self).__init__()

    def forward(self, input, target, scale=30.0, margin=0.35):
        # self.it += 1
        cos_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= margin
        output = output * scale

        logpt = nn.functional.log_softmax(output)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss

class ICCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = 0.3

    def forward(self, feature, label, **kwargs):  # TODO samples
        C = feature.size(1)
        scale = torch.sqrt(torch.tensor(C).float()) * self.m
        label = label.unsqueeze(1) # b 1
        label = label.repeat(1, C) # b c
        label = label.type(torch.BoolTensor).cuda() 

        # res_label = torch.zeros(label.size(), dtype=label.dtype)
        res_label = torch.where(label == 1, 0, 1)
        res_label = res_label.type(torch.BoolTensor).cuda() 

        pos_feature = torch.masked_select(feature, res_label) 
        neg_feature = torch.masked_select(feature, label)
        pos_feature = pos_feature.view(-1, C)
        neg_feature = neg_feature.view(-1, C)

        pos_center = torch.mean(pos_feature, dim=0, keepdim=True)
        num_p = pos_feature.size(0)
        num_n = neg_feature.size(0)
        pos_center1 = pos_center.repeat(num_p, 1)
        pos_center2 = pos_center.repeat(num_n, 1)
        # dis_pos = nn.functional.cosine_similarity(pos_feature, pos_center1, eps=1e-6)
        # dis_pos = torch.mean(dis_pos, dim=0)
        # dis_neg = nn.functional.cosine_similarity(neg_feature, pos_center2, eps=1e-6)
        # dis_neg = torch.mean(dis_neg, dim=0)
        dis_pos = torch.norm(pos_feature-pos_center1,p=2,dim=1).mean()
        dis_neg = torch.norm(neg_feature-pos_center2,p=2,dim=1).mean()
        max_margin = dis_pos - dis_neg + scale
        if max_margin < 0:
            max_margin = 0

        loss = dis_pos + max_margin
        return loss

def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)
    similarity_matrix = similarity_matrix.view(-1)

    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss