import torchvision
import os
import dataset.DataTools as dt
import torch

aug_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(256),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    #lambda x: x + torch.randn_like(x) * 0.01,
    # lambda x: add_salt_and_pepper_noise(x, salt_prob=0.05, pepper_prob=0.05),
    #torchvision.transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25])
     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

aug_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    #lambda x: x + torch.randn_like(x) * 0.01,
    # lambda x: add_salt_and_pepper_noise(x, salt_prob=0.05, pepper_prob=0.05),
    #torchvision.transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25])
     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def add_salt_and_pepper_noise(image_tensor, salt_prob=0.05, pepper_prob=0.05):

    noisy_image_tensor = image_tensor.clone()
    channels, height, width = noisy_image_tensor.shape
    num_salt = int(salt_prob * height * width)
    num_pepper = int(pepper_prob * height * width)

    for i in range(num_salt):
        x = torch.randint(0, width, (1,))
        y = torch.randint(0, height, (1,))
        noisy_image_tensor[:, y, x] = 1.0

    for i in range(num_pepper):
        x = torch.randint(0, width, (1,))
        y = torch.randint(0, height, (1,))
        noisy_image_tensor[:, y, x] = 0.0
    return noisy_image_tensor


class selfdataset():
    def getDatasets(self,pathfunc,infolist,transform,process=None,datasetfunc=None):
        datalist = []
        for info in infolist:
            discribe = info[0]
            dirlist = info[1]
            label = info[2]
            cnt = 0
            for dirname in dirlist:
                path = pathfunc(self.folder_path,dirname)
                cnt += len(os.listdir(path))
                datalist.append((path,label)) 
            print(discribe,cnt)

        if datasetfunc is not None:
            dataset = datasetfunc(datalist,transform=transform,process=process)
        else:
            dataset = dt.Imgdataset(datalist,transform=transform,process=process)

        return dataset

    def getDatasets2(self,pathfunc,infolist,transform,process=None,datasetfunc=None):
        datalist = []
        for info in infolist:
            discribe = info[0]
            dirlist = info[1]
            label = info[2]
            cnt = 0
            for dirname in dirlist:
                path = pathfunc(self.folder_path,dirname)
                cnt += len(os.listdir(path))
                datalist.append((path,label))
            print(discribe,cnt)

        if datasetfunc is not None:
            dataset = datasetfunc(datalist,transform=transform,process=process)
        else:
            dataset = dt.Imgdataset2(datalist,transform=transform,process=process)

        return dataset

    def getsetlist(self,real,setType,process=None,datasetfunc=None):
        setdir = self.R_dir if real is True else self.F_dir
        label = 0 if real is True else 1
        aug = aug_train if setType == 0 else aug_test
        pathfunc = self.trainpath if setType == 0 else self.validpath if setType == 1 else self.testpath
        setlist = []
        for setname in setdir:
            datalist = [(pathfunc(self.folder_path,setname),label)]
            if datasetfunc is not None:
                tmptestset = datasetfunc(datalist,transform=aug,porcess=process)
            else:
                tmptestset = dt.Imgdataset(datalist,transform=aug,process=process)
            setlist.append(tmptestset)
        return setlist,setdir

    def getsetlist2(self,real,setType,process=None,datasetfunc=None):
        setdir = self.R_dir if real is True else self.F_dir
        label = 0 if real is True else 1
        aug = aug_train if setType == 0 else aug_test
        pathfunc = self.trainpath if setType == 0 else self.validpath if setType == 1 else self.testpath
        setlist = []
        for setname in setdir:
            datalist = [(pathfunc(self.folder_path,setname),label)]
            if datasetfunc is not None:
                tmptestset = datasetfunc(datalist,transform=aug,porcess=process)
            else:
                tmptestset = dt.Imgdataset2(datalist,transform=aug,process=process)
            setlist.append(tmptestset)
        return setlist,setdir

    def getTrainsetR(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath,[[self.__class__.__name__+" TrainsetR", self.R_dir, 0]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainsetF(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath,[[self.__class__.__name__+" TrainsetF", self.F_dir, 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainsetDF(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath,[[self.__class__.__name__+" TrainsetF", ['deepfakes'], 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainsetFF(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath,[[self.__class__.__name__+" TrainsetF", ['face2face'], 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainsetFS(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath,[[self.__class__.__name__+" TrainsetF", ['faceswap'], 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainsetNT(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath,[[self.__class__.__name__+" TrainsetF", ['neuraltextures'], 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainset(self, process=None, datasetfunc=None):
        return self.getDatasets(self.trainpath,[[self.__class__.__name__+" TrainsetR", self.R_dir, 0], [self.__class__.__name__+" TrainsetF", self.F_dir, 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getValidsetR(self, process=None, datasetfunc=None):
        return self.getDatasets(self.validpath,[[self.__class__.__name__+" ValidsetR", self.R_dir, 0]], aug_test, process=process, datasetfunc=datasetfunc)

    def getValidsetF(self, process=None, datasetfunc=None):
        return self.getDatasets(self.validpath,[[self.__class__.__name__+" ValidsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getValidset(self, process=None, datasetfunc=None):
        return self.getDatasets(self.validpath,[[self.__class__.__name__+" ValidsetR", self.R_dir, 0], [self.__class__.__name__+" ValidsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestsetR(self, process=None, datasetfunc=None):
        return self.getDatasets(self.testpath,[[self.__class__.__name__+" TestsetR", self.R_dir, 0]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestsetF(self, process=None, datasetfunc=None):
        return self.getDatasets(self.testpath,[[self.__class__.__name__+" TestsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestset(self, process=None, datasetfunc=None):
        return self.getDatasets(self.testpath,[[self.__class__.__name__+" TestsetR", self.R_dir, 0], [self.__class__.__name__+" TestsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTrainsetR2(self, process=None, datasetfunc=None):
        return self.getDatasets2(self.trainpath,[[self.__class__.__name__+" TrainsetR", self.R_dir, 0]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainsetF2(self, process=None, datasetfunc=None):
        return self.getDatasets2(self.trainpath,[[self.__class__.__name__+" TrainsetF", self.F_dir, 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getTrainset2(self, process=None, datasetfunc=None):
        return self.getDatasets2(self.trainpath,[[self.__class__.__name__+" TrainsetR", self.R_dir, 0], [self.__class__.__name__+" TrainsetF", self.F_dir, 1]], aug_train, process=process, datasetfunc=datasetfunc)

    def getValidsetR2(self, process=None, datasetfunc=None):
        return self.getDatasets2(self.validpath,[[self.__class__.__name__+" ValidsetR", self.R_dir, 0]], aug_test, process=process, datasetfunc=datasetfunc)

    def getValidsetF2(self, process=None, datasetfunc=None):
        return self.getDatasets2(self.validpath,[[self.__class__.__name__+" ValidsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getValidset2(self, process=None, datasetfunc=None):
        return self.getDatasets2(self.validpath,[[self.__class__.__name__+" ValidsetR", self.R_dir, 0], [self.__class__.__name__+" ValidsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestsetR2(self, process=None, datasetfunc=None):
        return self.getDatasets2(self.testpath,[[self.__class__.__name__+" TestsetR", self.R_dir, 0]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestsetF2(self, process=None, datasetfunc=None):
        return self.getDatasets2(self.testpath,[[self.__class__.__name__+" TestsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

    def getTestset2(self, process=None, datasetfunc=None):
        return self.getDatasets2(self.testpath,[[self.__class__.__name__+" TestsetR", self.R_dir, 0], [self.__class__.__name__+" TestsetF", self.F_dir, 1]], aug_test, process=process, datasetfunc=datasetfunc)

class DFFD(selfdataset):
    def __init__(self,folder_path=""):
        super(selfdataset,self).__init__()
        self.folder_path = folder_path
        # self.R_dir = ["ffhq","CelebAMask-HQ"]
        # self.F_dir = ["faceapp","pggan_v2","stargan","stylegan_celeba","stylegan_ffhq"]
        self.R_dir = ["ffhq","CelebAMask-HQ"]
        self.F_dir = ["stargan","faceapp","stylegan_celeba","pggan_v2","stylegan_ffhq"]

        self.trainpath = lambda path,file:os.path.join(self.folder_path,file,"train")
        self.validpath = lambda path, file: os.path.join(self.folder_path, file, "valid")
        self.testpath = lambda path,file:os.path.join(self.folder_path,file,"test")
        self.train_mask_path = lambda path, file: os.path.join(self.folder_path, file, "train_mask")
        self.valid_mask_path = lambda path, file: os.path.join(self.folder_path, file, "valid_mask")
        self.test_mask_path = lambda path, file: os.path.join(self.folder_path, file, "test_mask")

class ff_c23(selfdataset):
    def __init__(self,folder_path=""):
        super(selfdataset,self).__init__()
        self.folder_path = folder_path
        self.R_dir = ["original"]
        self.F_dir = ["deepfakes","face2face","faceswap","neuraltextures"]
        self.trainpath = lambda path,file:os.path.join(self.folder_path,file,"train_face")
        self.validpath = lambda path, file: os.path.join(self.folder_path, file, "val_face")
        self.testpath = lambda path,file:os.path.join(self.folder_path,file,"test_face")

class ff_c40(selfdataset):
    def __init__(self,folder_path=""):
        super(selfdataset,self).__init__()
        self.folder_path = folder_path
        self.R_dir = ["original"]
        self.F_dir = ["deepfakes","face2face","faceswap","neuraltextures"]
        self.trainpath = lambda path,file:os.path.join(self.folder_path,file,"train_face")
        self.validpath = lambda path, file: os.path.join(self.folder_path, file, "val_face")
        self.testpath = lambda path,file:os.path.join(self.folder_path,file,"test_face")

class fsh_c23(selfdataset):
    def __init__(self,folder_path=""):
        super(selfdataset,self).__init__()
        self.folder_path = folder_path
        self.R_dir = ["original"]
        self.F_dir = ["faceshifter"]
        self.trainpath = lambda path,file:os.path.join(self.folder_path,file,"c23","train")
        self.validpath = lambda path, file: os.path.join(self.folder_path, file,"c23","val")
        self.testpath = lambda path,file:os.path.join(self.folder_path,file,"c23","test")

class fsh_c40(selfdataset):
    def __init__(self,folder_path=""):
        super(selfdataset,self).__init__()
        self.folder_path = folder_path
        self.R_dir = ["original"]
        self.F_dir = ["faceshifter"]
        self.trainpath = lambda path,file:os.path.join(self.folder_path,file,"c40","train")
        self.validpath = lambda path, file: os.path.join(self.folder_path, file,"c40","val")
        self.testpath = lambda path,file:os.path.join(self.folder_path,file,"c40","test")

class Celeb_DF(selfdataset):
    def __init__(self,folder_path=""):
        super(selfdataset,self).__init__()
        self.folder_path = folder_path
        self.R_dir = ["Real"]
        self.F_dir = ["Fake"]
        self.trainpath = lambda path,file:os.path.join(self.folder_path,file,"train","face")
        # self.validpath = lambda path, file: os.path.join(self.folder_path, file, "val_Img_face")
        self.testpath = lambda path,file:os.path.join(self.folder_path,file,"test","face")

class DFDC(selfdataset):
    def __init__(self,folder_path=""):
        super(selfdataset,self).__init__()
        self.folder_path = folder_path
        self.R_dir = ["real_new"]
        self.F_dir = ["fake_new"]

        self.trainpath = lambda path,file:os.path.join(self.folder_path,file)
        self.validpath = lambda path, file: os.path.join(self.folder_path, file)
        self.testpath = lambda path,file:os.path.join(self.folder_path,file)

class wildedeepfake(selfdataset):
    def __init__(self,folder_path=""):
        super(selfdataset,self).__init__()
        self.folder_path = folder_path
        self.R_dir = ["real"]
        self.F_dir = ["fake"]

        self.trainpath = lambda path,file:os.path.join(self.folder_path,file,"train")
        # self.validpath = lambda path, file: os.path.join(self.folder_path, file, "val_face")
        self.testpath = lambda path,file:os.path.join(self.folder_path,file,"test")