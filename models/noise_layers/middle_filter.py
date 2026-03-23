import torch.nn as nn
from kornia.filters import MedianBlur
import numpy as np

class RandomMF(nn.Module):

	def __init__(self, k1,k2):
		super().__init__()
		self.kernel=(k1,k2)
		self.kernels = [x for x in range(k1, k2+1) if x % 2 != 0]

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		kernel_size=np.random.choice(self.kernels)
		middle_filter = MedianBlur((kernel_size, kernel_size))
		output=middle_filter(image)
		return output

