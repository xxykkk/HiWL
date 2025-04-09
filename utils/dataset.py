from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, path, transform):
        self.path=path
        self.trans=transform
        self.img=os.listdir(path) 

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image=Image.open(os.path.join(self.path, self.img[idx])).convert('RGB')
        image=self.trans(image)
        return image

class MyDataset_P2(Dataset):
    def __init__(self, path, transform):
        self.path=path
        self.trans=transform
        self.img=os.listdir(path) 
        
        mid_point = len(self.img) // 2
        self.img1 = self.img[:mid_point]
        self.img2 = self.img[mid_point:mid_point*2]

    def __len__(self):
        return len(self.img1)

    def __getitem__(self, idx):
        image1=Image.open(os.path.join(self.path, self.img1[idx])).convert('RGB')
        image2=Image.open(os.path.join(self.path, self.img2[idx])).convert('RGB')
        image1=self.trans(image1)
        image2=self.trans(image2)
        return image1,image2
