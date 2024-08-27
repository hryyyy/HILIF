import torch
from dataset.util import data_prefetcher_two,clg_loss,setup_seed,Eval,Eval2
from myutils.utils import caleval
from mymodel.HILIF import Model
import dataset.dataset_profile as dp
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torchvision
import time
import warnings
import os
import setproctitle
pro_name = ''
setproctitle.setproctitle(pro_name)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device_ids = [0,1]

warnings.filterwarnings('ignore')
np.set_printoptions(precision=3)
parser = argparse.ArgumentParser()

parser.add_argument('--device',default='cuda:0',type=str)
parser.add_argument('--modelname',default='',type=str)
parser.add_argument('--batch_size',default=32,type=int)
parser.add_argument('--max_batch',default=500000,type=int)
parser.add_argument('--log_batch',default=1,type=int)
parser.add_argument('--save_batch',default=1,type=int)
parser.add_argument('--seed',default=5,type=int)
parser.add_argument('--lr',default=0.0002,type=int)
parser.add_argument('--resume_model',default=None)
parser.add_argument('--save_model',default=True,action='store_true')
parser.add_argument('--distributed',default=False)

args = parser.parse_args()
modelname = args.modelname

def Log(log):
    print(log)
    f = open("./logs/new_log/_" + modelname + ".log", "a")
    f.write(log + "\n")
    f.close()

if __name__ == '__main__':
    torch.cuda.set_device(args.device)
    setup_seed(args.seed)
    Max_AUC = 0
    model = Model().cuda()
    print(model)

    optim = torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=10,eta_min=1e-5)

    if args.distributed:
        model = torch.nn.DataParallel(model.to('cuda:0'),device_ids=device_ids,output_device=device_ids[0])
    if args.resume_model is not None:
        model.load_state_dict(torch.load(args.resume_model),map_location=args.device)
        print('loaded pred state dict success')

    print("="*50)
    print("Loaded model and lossfunc and optim success")
    cls_loss_func = torch.nn.CrossEntropyLoss()

    dataset = dp.ff_c40()
    trainsetR = dataset.getTrainsetR()
    trainsetF = dataset.getTrainsetF()

    validset = dataset.getValidset()

    testsetR = dataset.getTestsetR()
    testsetList,testsetName = dataset.getsetlist(False,2)

    setup_seed(args.seed)

    traindataLoaderR = DataLoader(
        trainsetR,
        batch_size = int(args.batch_size / 2),
        shuffle = True,
        num_workers = 4
    )
    traindataLoaderF = DataLoader(
        trainsetF,
        batch_size = int(args.batch_size / 2),
        shuffle = True,
        num_workers = 4
    )
    validdataLoader = DataLoader(
        validset,
        batch_size=args.batch_size,
        num_workers= 4
    )

    testdataLoaderR = DataLoader(
        testsetR,
        batch_size=args.batch_size,
        num_workers= 4
    )

    testdataLoaderList = []
    for tmptestset in testsetList:
        testdataLoaderList.append(
            DataLoader(
                tmptestset,
                batch_size = args.batch_size,
                num_workers = 4
            )
        )

    print("="*50)
    print("Loaded Dataset")
    batchind = 0
    e = 0
    sumcnt = 0
    sumloss = 0
    sum_cls_loss = 0
    sum_single_loss = 0

    torch.autograd.set_detect_anomaly(True)
    Log('batch_size:%d,lr:%.4f'%(args.batch_size,args.lr))

    while True:
        prefetcher = data_prefetcher_two(traindataLoaderR,traindataLoaderF)
        data,target = prefetcher.next()
        while data is not None and batchind < args.max_batch:
            stime = time.time()
            sumcnt += len(data)
            model.train()
            pred_target = model(data)
            cls_loss = cls_loss_func(pred_target,target)
            loss = cls_loss

            sum_cls_loss += cls_loss.detach() * len(data)
            sumloss += loss.detach() * len(data)
            loss = (loss - 0.04).abs() + 0.04

            data,target = prefetcher.next()
            optim.zero_grad()
            loss.backward()
            optim.step()

            batchind += 1
            print("\riter:%06d,cls_loss:%.5f,total_loss:%.5f,avg_cls_loss:%.5f,avg_loss:%.5f,cls_lr:%.6f,time:%.4f"%
                  (batchind,cls_loss,loss,sum_cls_loss/sumcnt,sumloss/sumcnt,optim.param_groups[0]['lr'],time.time()-stime),end='')

            if data is None:
                e += 1
                print()
                Log("batch:%03d loss:%.6f,avg_cls_loss:%.5f,avg_loss:%.6f"%(e,loss,sum_cls_loss/sumcnt,sumloss/sumcnt))

                cls_loss_valid,y_true_valid,y_pred_valid = Eval(model,cls_loss_func,validdataLoader)
                ap,acc,auc,tpr2,tpr3,tpr4 = caleval(y_true_valid,y_pred_valid)
                Log("ACC:%.6f AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f %s loss:%.2f" % (acc,auc, tpr2, tpr3, tpr4, "validset",cls_loss_valid))

                sumtloss = sumacc = sumauc = sumtpr2 = sumtpr3 = sumtpr4 = 0
                cls_loss_r,y_true_r,y_pred_r = Eval(model,cls_loss_func,testdataLoaderR)
                y_true = y_true_r
                y_pred = y_pred_r

                for i, tmptestdataloader in enumerate(testdataLoaderList):
                    cls_loss_f,y_true_f, y_pred_f = Eval(model, cls_loss_func, tmptestdataloader)
                    ap, acc, auc, tpr2, tpr3, tpr4 = caleval(torch.cat((y_true_r, y_true_f)),
                                                               torch.cat((y_pred_r, y_pred_f)))
                    y_true = torch.cat((y_true,y_true_f))
                    y_pred = torch.cat((y_pred,y_pred_f))

                    sumacc += acc
                    sumauc += auc
                    sumtpr2 += tpr2
                    sumtpr3 += tpr3
                    sumtpr4 += tpr4
                    sumtloss += cls_loss_f
                    Log("ACC:%.6f,AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f %s loss:%.2f" % (acc,auc,tpr2,tpr3,tpr4, testsetName[i],cls_loss_f))

                ap,acc,auc,tpr2,tpr3,tpr4 = caleval(y_true,y_pred)
                Log("ACC:%.6f,AUC:%.6f TPR_2:%.6f TPR_3:%.6f TPR_4:%.6f Test r_loss:%.2f f_loss:%.2f test:%.6f" %
                    (acc, auc, tpr2, tpr3, tpr4,cls_loss_r,sumtloss/len(testdataLoaderList),sumauc/len(testdataLoaderList)))

                if batchind % args.save_batch == 0 or auc > Max_AUC:
                    Max_AUC = auc
                    if args.save_model:
                        torch.save(model.state_dict(),
                                   "./models/" + modelname + "_model_" + str(batchind))
        scheduler.step()