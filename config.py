from utils.dataset import MyDataset, MyDataset_P2
import torch
from torchvision import transforms
import os, time

'''
input
'''
H, W=128,128 
message_length=64 
encoder_channels=256
transform = transforms.Compose([
    transforms.RandomCrop((H, W), pad_if_needed=True, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
transform_test = transforms.Compose([
    transforms.CenterCrop((H, W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


'''
train
'''
lr=1e-4
batch_size=8 
epochs=100
epochs_ft=100

'''
save
'''

save_path=os.path.join('runs',time.strftime('HiWL_train' + "_%m_%d__%H_%M_%S", time.localtime()))

train_noise_layer = ["Identity()","RandomJpegMask(40,100, padding=True)", "RandomJpeg(40,100,padding=True)", 
                    "RandomJpegSS(40,100,padding=True)", "RandomGN(3,10)", "RandomGF(3,8)", "RandomColor(0.5,1.5)", 
                    "RandomDropout(0.7,1)", 
                    "RandomMF(0,7)", 
                    "RandomBright(0,0.5)","RandomSaturation(0.2,1.8)", "RandomHue(-0.3,0.3)","RandomContrast(0.2,1.8)",
                    "RandomResize(target_size=128)", 
                    "RandomCrop(0.7, 1, target_size=128)", "RandomPIP(1,2, target_size=128)", 
                    "RandomPadding(0,50,target_size=128)","RandomOcclusion(0.25,0.5)", 
                    "RandomAffine_Diff(angle=(-180,180),shear=(0,0))", 
                    "RandomAffine_Diff(angle=(0,0),shear=(0,30))",
                    "RandomAffine_Diff(angle=(-180,180),shear=(0,30))",
                    "RandomRotate(180)", ##
                    "RandomAffine(angle=(0,0),shear=(0,30))",
                    "RandomAffine(angle=(-180,180),shear=(0,30))"] 

val_noise_layer = ["Identity()", "RandomJpeg(40,100,padding=True)", "RandomJpegSS(40,100,padding=True)", 
                    "RandomGN(3,10)", "RandomGF(3,8)", "RandomColor(0.5,1.5)", "RandomDropout(0.7,1)",
                    "RandomMF(0,7)", 
                    "RandomBright(0,0.5)","RandomSaturation(0.2,1.8)", "RandomHue(-0.3,0.3)","RandomContrast(0.2,1.8)",
                    "RandomResize(target_size=128)",
                    "RandomCrop(0.7, 1, target_size=128)", 
                    "RandomPIP(1,2, target_size=128)", "RandomPadding(0,50,target_size=128)",
                    "RandomOcclusion(0.25,0.5)",
                    "RandomRotate(180)",
                    "RandomAffine(angle=(0,0),shear=(0,30))",
                    "RandomAffine(angle=(-180,180),shear=(0,30))",
                    "RandomAffine_Diff(angle=(-180,180),shear=(0,0))", 
                    "RandomAffine_Diff(angle=(0,0),shear=(0,30))",
                    "RandomAffine_Diff(angle=(-180,180),shear=(0,30))",]

import pytorch_lightning as pl

class WMDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, batch_size, transform_train, transform_test):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.transform_train = transform_train
        self.transform_test = transform_test

    def setup(self, stage=None):
        self.dataset_train=MyDataset(self.train_dir,transform=self.transform_train)
        self.dataset_test=MyDataset(self.test_dir,transform=self.transform_test)
        self.dataset_train_sub2=torch.utils.data.Subset(self.dataset_train,range(50000,100000))
        self.dataset_test_sub2=torch.utils.data.Subset(self.dataset_test,range(1000,2000))

    def train_dataloader(self):
        dataloader_train=torch.utils.data.DataLoader(self.dataset_train_sub2,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True)
        return dataloader_train

    def val_dataloader(self):
        dataloader_test=torch.utils.data.DataLoader(self.dataset_test_sub2,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
        return dataloader_test

class WMDataModule_P2(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, batch_size, transform_train, transform_test):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.transform_train = transform_train
        self.transform_test = transform_test

    def setup(self, stage=None):
        self.dataset_train=MyDataset_P2(self.train_dir,transform=self.transform_train)
        self.dataset_test=MyDataset_P2(self.test_dir,transform=self.transform_test)
        self.dataset_train_sub2=torch.utils.data.Subset(self.dataset_train,range(0,50000))
        self.dataset_test_sub2=torch.utils.data.Subset(self.dataset_test,range(0,1000))

    def train_dataloader(self):
        dataloader_train=torch.utils.data.DataLoader(self.dataset_train_sub2,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True)
        return dataloader_train

    def val_dataloader(self):
        dataloader_test=torch.utils.data.DataLoader(self.dataset_test_sub2,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
        return dataloader_test

