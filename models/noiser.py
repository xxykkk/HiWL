

from .noise_layers.crop import RandomCrop
from .noise_layers.dropout import RandomDropout
from .noise_layers.gaussian_filter import RandomGF
from .noise_layers.gaussian_noise import RandomGN
from .noise_layers.identity import Identity
from .noise_layers.jpeg import RandomJpegMask, RandomJpegSS, RandomJpeg, MyJpegTest 
from .noise_layers.resize import RandomResize
from .noise_layers.rotate import RandomRotate
from .noise_layers.pip import RandomPIP
from .noise_layers.occlusion import RandomOcclusion
from .noise_layers.color import RandomColor,RandomBright,RandomSaturation,RandomHue,RandomContrast
from .noise_layers.padding import RandomPadding

from .noise_layers.occlusion import RandomOcclusion 
from .noise_layers.padding import RandomPadding

from .noise_layers.salt_pepper import RandomSP
from .noise_layers.middle_filter import RandomMF
from .noise_layers.affine import RandomAffine,RandomAffine_Diff

from torch import nn
import numpy as np


class Noise(nn.Module):
    """
    Noise Network, randomly or intentionally selected a kind of noise
    """
    def __init__(self, layers):
        super().__init__()
        layers = [eval(layer_str) for layer_str in layers]
        self.noise = nn.ModuleList(layers)

    def forward(self, image_and_cover, k=None):
        if k==None: k=np.random.randint(0, len(self.noise))
        noised_image = self.noise[k](image_and_cover)
        return noised_image, k
