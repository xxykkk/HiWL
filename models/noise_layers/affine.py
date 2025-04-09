from torchvision.transforms import v2
import torch
import torch.nn as nn
import kornia
import random


class RandomAffine(nn.Module):
    def __init__(self, translate=(0, 0), angle=(0, 0), shear=(0, 0)):
        super().__init__()
        self.translate_range = translate
        self.angle_range = angle
        self.shear_range = shear

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        
        affine_transfomer = v2.RandomAffine(degrees=self.angle_range, 
            translate=self.translate_range, shear=self.shear_range) #scale=(0,0)
        output = affine_transfomer(image)

        return output


class RandomAffine_Diff(nn.Module):
    def __init__(self, translate=(0, 0), angle=(0, 0), shear=(0, 0)):
        super().__init__()
        self.translate_range = translate
        self.angle_range = angle
        self.shear_range = shear

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        degree=random.uniform(*self.angle_range)
        shear=random.uniform(*self.shear_range)/100

        _, _, height, width = image_and_cover[0].shape
        translations = torch.zeros(1, 2, device=image.device)  
        center = torch.tensor([[width / 2, height / 2]], device=image.device)  
        scale = torch.tensor([[1.0, 1.0]], device=image.device)  
        angle = torch.tensor([degree], device=image.device) 
        sx = torch.tensor([shear], device=image.device) 
        sy = torch.tensor([shear], device=image.device) 
        affine_matrix = kornia.geometry.transform.get_affine_matrix2d(
            translations=translations, center=center, scale=scale, angle=angle, sx=sx, sy=sy
        )
        transformed_img_tensor = kornia.geometry.transform.affine(
            image, affine_matrix[:,:2,:], 
        )

        return transformed_img_tensor