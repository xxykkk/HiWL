
import numpy as np
from torch import nn
import torchvision.transforms.functional as F2
import kornia, random

class RandomColor(nn.Module):
	
	def __init__(self, min_r=0.5, max_r=1.5):
		super(RandomColor, self).__init__()
		self.min_r = min_r
		self.max_r = max_r
		
	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		choice = np.random.randint(0, 3)
		r = np.random.uniform(self.min_r, self.max_r)
		output = (image+1)/2
		if choice == 0:
			output = F2.adjust_brightness(output, r)
		elif choice == 1:
			output = F2.adjust_contrast(output, r)
		elif choice == 2:
			output = F2.adjust_saturation(output, r)

		output = (output-0.5)/0.5
		return output


class RandomBright(nn.Module):
	def __init__(self, min_r, max_r):
		super().__init__()
		self.range=(min_r, max_r)
	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		rate=random.uniform(*self.range)
		output=kornia.enhance.adjust_brightness(image, rate) 
		return output

class RandomSaturation(nn.Module):
	def __init__(self, min_r, max_r):
		super().__init__()
		self.range=(min_r, max_r)
	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		rate=random.uniform(*self.range)
		output=kornia.enhance.adjust_saturation(image, rate) 
		return output

class RandomHue(nn.Module):
	def __init__(self, min_r, max_r):
		super().__init__()
		self.range=(min_r, max_r)
	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		rate=random.uniform(*self.range)
		output=kornia.enhance.adjust_hue(image, rate) 
		return output

class RandomContrast(nn.Module):
	def __init__(self, min_r, max_r):
		super().__init__()
		self.range=(min_r, max_r)
	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		rate=random.uniform(*self.range)
		output=kornia.enhance.adjust_contrast(image, rate) 
		return output

