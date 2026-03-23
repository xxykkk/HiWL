import os
import time
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from utils.dataset import MyDataset

# --- Configuration ---
H, W = 128, 128
message_length = 64
lr = 1e-4
batch_size = 64
epochs = 100
encoder_channels = 256

# --- Data Transformation ---
transform_test = transforms.Compose([
    transforms.CenterCrop((H, W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Dataset and Dataloader ---
dataset_test = MyDataset('YOUR_DATA_PATH', transform = transform_test)

dataset_test_sub1 = Subset(dataset_test, range(1000))
dataset_test_sub2 = Subset(dataset_test, range(1000, 2000))

dataloader_test = DataLoader(
    dataset_test_sub1, 
    batch_size = batch_size, 
    shuffle = False, 
    num_workers = 4, 
    pin_memory = True
)

dataloader_test2 = DataLoader(
    dataset_test_sub2, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

# --- Save Path ---
save_path = os.path.join('My_exp', time.strftime('UEW_test_%m_%d__%H_%M_%S', time.localtime()))

# --- Noise Layers Definition ---

val_noise_layer_standard = ["Identity()", "RandomJpeg(40,100,padding=True)", "RandomJpegSS(40,100,padding=True)", 
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

val_noise_layer_easy = ["Identity()", "RandomJpeg(90,91,padding=True)", "RandomJpegSS(90,91,padding=True)", 
					"RandomGN(3,4)", "RandomGF(3,4)", "RandomColor(1.1,1.11)", "RandomDropout(0.9,0.91)",
					"RandomMF(2,3)",
					"RandomBright(0.2,0.21)","RandomSaturation(0.8,0.81)", "RandomHue(-0.1,0.1)","RandomContrast(0.8,0.81)",
					"RandomResize(min_ratio=0.8, max_ratio=1.2,target_size=128)",
					"RandomCrop(0.9, 0.91, target_size=128)", 
					"RandomPIP(1.19,1.2, target_size=128)", "RandomPadding(10,11,target_size=128)",
					"RandomOcclusion(0.25,0.251)",
					"RandomRotate(10)",
					"RandomAffine(angle=(0,0),shear=(10,11))",
					"RandomAffine(angle=(10,11),shear=(10,11))",
					"RandomAffine_Diff(angle=(10,11),shear=(0,0))", 
					"RandomAffine_Diff(angle=(0,0),shear=(10,11))",
					"RandomAffine_Diff(angle=(10,11),shear=(10,11))",]

val_noise_layer_extreme = ["Identity()","RandomJpeg(40,41,padding=True)", 
					"RandomGN(9,10)", "RandomGF(7,8)", "RandomColor(1.49,1.5)", "RandomDropout(0.7,0.71)",
					"RandomMF(6,7)", 
					"RandomBright(0.49,0.5)","RandomSaturation(0.2,0.21)", "RandomHue(-0.3,0.3)","RandomContrast(0.2,0.21)",
					"RandomResize(target_size=128)",
					"RandomCrop(0.7, 0.71, target_size=128)", 
					"RandomPIP(1.9,2, target_size=128)", "RandomPadding(49,50,target_size=128)",
					"RandomOcclusion(0.49,0.5)",
					"RandomRotate(180)",
					"RandomAffine(angle=(0,0),shear=(29,30))",
					"RandomAffine(angle=(179,180),shear=(29,30))",
					"RandomAffine_Diff(angle=(179,180),shear=(0,0))", 
					"RandomAffine_Diff(angle=(0,0),shear=(29,30))",
					"RandomAffine_Diff(angle=(179,180),shear=(29,30))",]

val_noise_layer_paper = ["Identity()","RandomJpeg(40,41,padding=True)", 
					"RandomGN(9.99,10)", "RandomGF(7.99,8)", "RandomColor(1.49,1.5)", "RandomDropout(0.7,0.701)",
					"RandomMF(6,7)", 
					"RandomBright(0.49,0.5)","RandomSaturation(0.2,0.201)", "RandomHue(0.299,0.3)","RandomContrast(0.2,0.201)",
					"RandomResize(target_size=128)",
					"RandomCrop(0.7, 0.701, target_size=128)", 
					"RandomPIP(1.99,2, target_size=128)", "RandomPadding(49,50,target_size=128)",
					"RandomOcclusion(0.49,0.5)",
					"RandomRotate(180)",
					"RandomAffine(angle=(0,0),shear=(29.9,30))",
					"RandomAffine(angle=(179.9,180),shear=(29.9,30))",
					"RandomAffine_Diff(angle=(179.9,180),shear=(0,0))", 
					"RandomAffine_Diff(angle=(0,0),shear=(29.9,30))",
					"RandomAffine_Diff(angle=(179.9,180),shear=(29.9,30))",]

val_noise_layer = val_noise_layer_paper

