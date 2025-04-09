
import torch
from torch import nn
import numpy as np


class RandomDropout(nn.Module):
    """
    replace watermarked image pixels with original ones randomly
    """
    def __init__(self, minprob=0.7, maxprob=1.0):
        super(RandomDropout, self).__init__()
        self.minprob = minprob
        self.maxprob = maxprob

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        prob = np.random.rand() * (self.maxprob - self.minprob) + self.minprob
        rdn = torch.rand(image.shape).to(image.device)
        output = torch.where(rdn > prob, cover_image, image)
        return output
