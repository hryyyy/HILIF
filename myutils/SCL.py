import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Tuple

class SingleCenterLoss(nn.Module):
    """
    Single Center Loss

    Reference:
    J Li, Frequency-aware Discriminative Feature Learning Supervised by Single-CenterLoss for Face Forgery Detection, CVPR 2021.

    Parameters:
        m (float): margin parameter.
        D (int): feature dimension.
        C (vector): learnable center.
    """

    def __init__(self, m=0.3, D=1000, use_gpu=True):
        super(SingleCenterLoss, self).__init__()
        self.m = m
        self.D = D
        self.margin = self.m * torch.sqrt(torch.tensor(self.D).float())
        self.use_gpu = use_gpu
        self.l2loss = nn.MSELoss(reduction='none')

        if self.use_gpu:
            self.C = nn.Parameter(torch.randn(self.D).cuda())
        else:
            self.C = nn.Parameter(torch.randn(self.D))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_
            size).
        """
        batch_size = x.size(0)
        eud_mat = torch.sqrt(self.l2loss(x, self.C.expand(batch_size, self.C.size(0))).sum(dim=1, keepdim=True))

        # print(eud_mat)

        labels = labels.unsqueeze(1) # [b,1]

        fake_count = labels.sum()

        dist_fake = (eud_mat * labels.float()).clamp(min=1e-12, max=1e+12).sum()
        dist_real = (eud_mat * (1 - labels.float())).clamp(min=1e-12, max=1e+12).sum()

        if fake_count != 0:
            dist_fake /= fake_count

        if fake_count != batch_size:
            dist_real /= (batch_size - fake_count)

        max_margin = dist_real - dist_fake + self.margin

        if max_margin < 0:
            max_margin = 0

        loss = dist_real + max_margin
        return loss

class SingleCenterLoss2(nn.Module):
    """Single-Center loss.
        Reference:
        Li, J., Xie, H., Li, J., Wang, Z., & Zhang, Y. (2021). Frequency-aware Discriminative Feature Learning Supervised by Single-Center Loss for Face Forgery Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 6458-6467).
        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
            m (float): scale factor that the margin is proportional to the square root of dimension
        """
    def __init__(self, num_classes=2, feat_dim=256, m=0.3):
        super(SingleCenterLoss2, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = m

        self.center = nn.Parameter(torch.randn(1, self.feat_dim).cuda())

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        m_nat = torch.norm(x[labels==0]-self.center, p=2, dim=1).mean()
        m_man = torch.norm(x[labels==1]-self.center, p=2, dim=1).mean()
        loss = m_nat + nn.functional.relu(m_nat - m_man + self.m * math.sqrt(self.feat_dim))
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
        # print(ap)
        # print(an)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        # print(self.soft_plus(torch.logsumexp(logit_n, dim=0)))
        # print(self.soft_plus(torch.logsumexp(logit_p, dim=0)))
        # print(loss)
        return loss