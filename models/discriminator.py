# *************************************************************************
# Code Modified from https://github.com/bytedance/DWSF/blob/main/networks/models/Discriminator.py
# *************************************************************************
from .basic_blocks.ConvNet import ConvBNRelu, ConvNet
from torch import nn
import torch


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.layers = nn.Sequential(
			ConvNet(3, 64, blocks=1,),
			ConvBNRelu(64, 64, kernel_size=3, stride=2, padding=0),
			ConvNet(64, 64, blocks=1,),
			ConvBNRelu(64, 64, kernel_size=3, stride=2, padding=0),
			ConvNet(64, 64, blocks=1,),
			ConvBNRelu(64, 64, kernel_size=3, stride=2, padding=0),
			ConvNet(64, 64, blocks=1,),
			ConvBNRelu(64, 64, kernel_size=3, stride=2, padding=0),
		)

	def forward(self, image):
		x = self.layers(image)
		x = torch.mean(x, dim=(2, 3))
		x = torch.mean(x, dim=(1), keepdim=True)

		return x

