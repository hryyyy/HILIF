from __future__ import print_function,division,absolute_import
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor,Resize
from PIL import Image
import torchvision
from Xception import efficient

class data_prefetcher():
    def __init__(self,loader):
        self.stream = torch.cuda.Stream()
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.next_input,self.next_mask,self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_mask = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_mask = self.next_mask.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        mask = self.next_mask
        target = self.next_target
        self.preload()
        return input,mask,target

class data_prefetcher_2():
    def __init__(self,loader):
        self.stream = torch.cuda.Stream()
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.next_input,self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input,target

class data_prefetcher_two():
    def __init__(self, loader1, loader2):
        self.stream = torch.cuda.Stream()
        self.loader1 = iter(loader1)
        self.loader2 = iter(loader2)
        self.preload()

    def preload(self):
        try:
            tmp_input1, tmp_target1 = next(self.loader1)
            tmp_input2, tmp_target2 = next(self.loader2)
            self.next_input,self.next_target = torch.cat((tmp_input1, tmp_input2)), torch.cat((tmp_target1, tmp_target2))

        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input,target

class data_prefetcher_two_2():
    def __init__(self, loader1, loader2):
        self.stream = torch.cuda.Stream()
        self.loader1 = [iter(loader1),iter(loader1),iter(loader1),iter(loader1)]
        self.loader2 = iter(loader2)
        self.preload()

    def preload(self):
        try:
            tmp_input1, tmp_mask1, tmp_target1 = next(self.loader1[0],['1','1','1'])
            if tmp_input1 == '1':
                tmp_input1, tmp_mask1, tmp_target1 = next(self.loader1[1], ['2', '2', '2'])
            if tmp_input1 == '2':
                tmp_input1, tmp_mask1, tmp_target1 = next(self.loader1[2], ['3', '3', '3'])
            if tmp_input1 == '3':
                tmp_input1, tmp_mask1, tmp_target1 = next(self.loader1[3], ['4', '4', '4'])

            tmp_input2, tmp_mask2, tmp_target2 = next(self.loader2)
            self.next_input, self.next_mask, self.next_target = torch.cat((tmp_input1, tmp_input2)), torch.cat(
                (tmp_mask1, tmp_mask2)), torch.cat((tmp_target1, tmp_target2))

        except StopIteration:
            self.next_input = None
            self.next_mask = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_mask = self.next_mask.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        mask = self.next_mask
        target = self.next_target
        self.preload()
        return input, mask, target

class clg_loss(nn.Module):
    def __init__(self):
        super(clg_loss, self).__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, pred, truth):
        batch = pred.shape[0]
        pred = pred.view(batch, -1) 
        truth = truth.view(batch, -1).requires_grad_(False)
        # print(pred, truth) pred.shape[batch,full-linear]
        pred, truth = self.relu(pred), self.relu(truth)
        if pred.shape != truth.shape:
            raise Exception('pred shape:', pred.shape, 'truth.shape:', truth.shape)
        else:
            loss = F.binary_cross_entropy(pred, truth)
        return loss

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.unfold = nn.Unfold(kernel_size=(5,5),stride=2)

    def forward(self,feature,label):
        f = self.unfold(feature).transpose(-2,-1)
        cos = 1 - torch.cosine_similarity(f.unsqueeze(1),f.unsqueeze(2),eps=1e-6,dim=-1)
        sum = torch.sum(cos,dim=-1)
        std = sum.size(1)
        sum = sum / std

        # max,idx = torch.max(sum,dim=-1)
        topk,idx = torch.topk(sum,k=1,dim=-1)
        #std = topk.size(1)
        #sum = torch.sum(topk,dim=-1) / std
        sum = torch.sum(topk,dim=-1)

        fake_label = label
        real_label = 1 - label

        fake_label = fake_label.type(torch.BoolTensor).cuda()
        real_label = real_label.type(torch.BoolTensor).cuda()

        real_loss = torch.masked_select(sum,real_label)
        fake_loss = torch.masked_select(sum,fake_label)

        real_loss = torch.mean(real_loss,dim=0)
        fake_loss = torch.mean(fake_loss,dim=0)

        loss = 1 - fake_loss + real_loss
        if loss < 0:
            loss = torch.tensor(0).cuda()

        return loss

class MyLoss_max(torch.nn.Module):
    def __init__(self):
        super(MyLoss_max, self).__init__()
        self.unfold = nn.Unfold(kernel_size=(5,5),stride=2)

    def forward(self,feature,label):
        b,c,h,w = feature.shape
        patch = self.unfold(feature).reshape(b,c,5*5,-1).permute(0,3,2,1) # batch block_num patch*patch channel

        cos = 1 - torch.cosine_similarity(patch.unsqueeze(2),patch.unsqueeze(3),eps=1e-6,dim=-1) # b n p p
        sum = torch.mean(cos,dim=-1) # b n p 

        max,idx = torch.max(sum,dim=-1)
        topk,idx = torch.topk(max,k=1,dim=-1)
        sum = torch.mean(topk,dim=-1)

        fake_label = label
        real_label = 1 - label

        fake_label = fake_label.type(torch.BoolTensor).cuda()
        real_label = real_label.type(torch.BoolTensor).cuda()

        real_loss = torch.masked_select(sum,real_label)
        fake_loss = torch.masked_select(sum,fake_label)

        real_loss = torch.mean(real_loss,dim=0)
        fake_loss = torch.mean(fake_loss,dim=0)

        loss = 1 - fake_loss + real_loss
        if loss < 0:
            loss = torch.tensor(0).cuda()

        return loss

class GradCAM():
    '''
    Grad-cam: Visual explanations from deep networks via gradient-based localization
    Selvaraju R R, Cogswell M, Das A, et al.
    https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
    '''

    def __init__(self, model, target_layers, use_cuda=True):
        super(GradCAM).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.target_layers = target_layers

        self.target_layers.register_forward_hook(self.forward_hook)
        self.target_layers.register_full_backward_hook(self.backward_hook)

        self.activations = []
        self.grads = []

    def forward_hook(self, module, input, output):
        self.activations.append(output[0])

    def backward_hook(self, module, grad_input, grad_output):
        self.grads.append(grad_output[0].detach())

    def calculate_cam(self, model_input):
        if self.use_cuda:
            device = torch.device('cuda')
            self.model.to(device)  # Module.to() is in-place method
            model_input = model_input.to(device)  # Tensor.to() is not a in-place method
        self.model.eval()

        # forward
        y_hat = self.model(model_input)
        max_class = np.argmax(y_hat.cpu().data.numpy(), axis=1)

        # backward
        self.model.zero_grad()
        y_c = y_hat[0, max_class]
        y_c.backward()

        # get activations and gradients
        activations = self.activations[0].cpu().data.numpy().squeeze()
        print(self.grads)
        grads = self.grads[0].cpu().data.numpy().squeeze()

        # calculate weights
        weights = np.mean(grads.reshape(grads.shape[0], -1), axis=1)
        weights = weights.reshape(-1, 1, 1)
        cam = (weights * activations).sum(axis=0)
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()
        return cam

    @staticmethod
    def show_cam_on_image(image, cam):
        # image: [H,W,C]
        h, w = image.shape[:2]

        cam = cv2.resize(cam, (h, w))
        cam = cam / cam.max()
        heatmap = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)  # [H,W,C]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        image = image / image.max()
        heatmap = heatmap / heatmap.max()

        result = 0.4 * heatmap + 0.6 * image
        result = result / result.max()

        plt.figure()
        plt.imshow((result * 255).astype(np.uint8))
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def preprocess_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        preprocessing = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        return preprocessing(img.copy()).unsqueeze(0)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def Eval(model,lossfunc,dtloader):
    model.eval()
    sum_cls_loss = 0
    y_true_all = None
    y_pred_all = None

    with torch.no_grad():
        for (j,batch) in enumerate(dtloader):
            x,y_true = batch
            y_pred = model.forward(x.cuda())
            # mask_pred = torchvision.transforms.Resize(224)(mask_pred)

            cls_loss = lossfunc(y_pred,y_true.cuda())
            # mask_loss = imgloss(mask_pred,mask_true.cuda())

            sum_cls_loss += cls_loss.detach()*len(x)
            # sum_mask_loss += mask_loss.detach()*len(x)

            y_pred = torch.nn.functional.softmax(
                y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all,y_true))
                y_pred_all = torch.cat((y_pred_all,y_pred))

    return sum_cls_loss/len(y_true_all),y_true_all.detach(),y_pred_all.detach()

def Eval2(model,lossfunc,dtloader):
    model.eval()
    sum_cls_loss = 0
    y_true_all = None
    y_pred_all = None

    with torch.no_grad():
        for (j,batch) in enumerate(dtloader):
            x,y_true = batch 
            y_fea,y_pred = model.forward(x.cuda())
            # mask_pred = torchvision.transforms.Resize(224)(mask_pred)

            cls_loss = lossfunc(y_pred,y_true.cuda())
            sum_cls_loss += cls_loss.detach()*len(x)

            y_pred = torch.nn.functional.softmax(
                y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all,y_true))
                y_pred_all = torch.cat((y_pred_all,y_pred))

    return sum_cls_loss/len(y_true_all),y_true_all.detach(),y_pred_all.detach()

def Eval3(model,lossfunc1,lossfunc2,dtloader):
    model.eval()
    lossfunc2.eval()
    sum_cls_loss = 0
    sum_scl_loss = 0
    y_true_all = None
    y_pred_all = None

    with torch.no_grad():
        for (j, batch) in enumerate(dtloader):
            x, mid, y_true = batch
            feat,y_pred = model.forward(x.cuda(), mid.cuda())

            cls_loss = lossfunc1(y_pred, y_true.cuda())
            scl_loss = lossfunc2(feat,y_true.cuda())

            sum_cls_loss += cls_loss.detach() * len(x)
            sum_scl_loss += scl_loss.detach() * len(x)

            y_pred = torch.nn.functional.softmax(
                y_pred.detach(), dim=1)[:, 1].flatten()
            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))

    return sum_cls_loss / len(y_true_all),sum_scl_loss / len(y_true_all), y_true_all.detach(), y_pred_all.detach()

def cal_fam(model,inputs):
    model.zero_grad()
    inputs = inputs.detach().clone()
    inputs.requires_grad_()
    output = model(inputs)
    print(output)
    target = output[:,1]-output[:,0]
    target.backward(torch.ones(target.shape))
    fam = torch.abs(inputs.grad)
    fam = torch.max(fam,dim=1,keepdim=True)[0]
    return fam

def cal_normfam(model,inputs):
    fam = cal_fam(model,inputs)
    _,x,y = fam[0].shape
    fam = torch.nn.functional.interpolate(fam, (int(y / 2), int(x / 2)), mode='bilinear', align_corners=False) #用线性插值进行下采样
    fam = torch.nn.functional.interpolate(fam, (y, x), mode='bilinear', align_corners=False) #再上采样？
    for i in range(len(fam)):
        fam[i] -= torch.min(fam[i])
        fam[i] /= torch.max(fam[i])
    return fam