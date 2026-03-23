
from .basic_blocks.ConvNet import ConvBNRelu
from .basic_blocks.SENet import SENet, SENet_decoder
from torch import nn
import torch
import torch.nn.functional as F
from .swin import PatchMerging,SwinTransformerBlock as SwinTransformerBlock
from .dino import DINOHead, dinoloss

class Decoder_swin(nn.Module):
    def __init__(self, message_length=30, decoder_channels=96, in_channel=3):

        super().__init__()
        self.channels = decoder_channels

        self.layers = nn.Sequential(
            ConvBNRelu(in_channel, self.channels), # 96,128,128
            SENet(self.channels, self.channels, blocks=4), # 96, 128, 128
            SENet_decoder(self.channels, self.channels//2, blocks=2, drop_rate2=2), # 96, 64,64
        )

        self.att1=nn.Sequential(
            SwinTransformerBlock(dim=96, input_resolution=(64,64), num_heads = 3, window_size=8, ),
            SwinTransformerBlock(dim=96, input_resolution=(64,64), num_heads = 3, window_size=8, ),
        )
        self.drop1=SENet_decoder(96, 96, blocks=2, drop_rate2=2) 
        self.att2=nn.Sequential(
            SwinTransformerBlock(dim=192, input_resolution=(32,32), num_heads = 6, window_size=8, ),
            SwinTransformerBlock(dim=192, input_resolution=(32,32), num_heads = 6, window_size=8, ),
        )
        self.drop2=SENet_decoder(192, 192, blocks=2, drop_rate2=2) 
        self.att3=nn.Sequential(
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=8, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=8, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=8, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=8, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=8, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=8, ),
        )
        self.drop3=SENet_decoder(384, 384, blocks=2, drop_rate2=2) 
        self.att4=nn.Sequential(
            SwinTransformerBlock(dim=768, input_resolution=(8,8), num_heads = 24, window_size=8, ),
            SwinTransformerBlock(dim=768, input_resolution=(8,8), num_heads = 24, window_size=8, ),
        )
        self.drop4=SENet_decoder(768, 384, blocks=2, drop_rate2=2) # 768,4,4
        
        self.att_layers=nn.ModuleList([self.att1,self.att2,self.att3,self.att4])
        self.drop_layers=nn.ModuleList([self.drop1,self.drop2,self.drop3,self.drop4])

        '''
        linear
        '''
        self.linear = nn.Linear(768*2*2, message_length) 
        self.activation = nn.ReLU(True)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, image_with_wm):
        x = image_with_wm

        x = self.layers(x)

        for i in range(4):

            B,C,H,W = x.shape # b,c,h,w
            x = (x.reshape(B,C,H*W)).permute(0, 2, 1)
            x = self.att_layers[i](x)
            x = (x.permute(0, 2, 1)).reshape(B,C,H,W)
            x = self.drop_layers[i](x)
        
        x = self.pool(x)

        x = x.view(x.shape[0], -1)

        x = self.linear(x)
        x = self.activation(x)
        return x

class PatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding
    """
    def __init__(self, img_size=128, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x


class Decoder_Swin_v2(nn.Module):
    def __init__(self, message_length=64, decoder_channels=96, in_channel=3):

        super().__init__()
        self.channels = decoder_channels

        self.layers = nn.Sequential(
            ConvBNRelu(in_channel, self.channels), # 96,128,128
            SENet(self.channels, self.channels, blocks=4), # 96, 128, 128
            SENet_decoder(self.channels, self.channels//2, blocks=2, drop_rate2=2), # 96, 64,64
        )

        self.att1=nn.Sequential(
            SwinTransformerBlock(dim=96, input_resolution=(64,64), num_heads = 3, window_size=4, ),
            SwinTransformerBlock(dim=96, input_resolution=(64,64), num_heads = 3, window_size=4, ),
        )
        self.drop1=SENet_decoder(96, 96, blocks=2, drop_rate2=2) 
        self.att2=nn.Sequential(
            SwinTransformerBlock(dim=192, input_resolution=(32,32), num_heads = 6, window_size=4, ),
            SwinTransformerBlock(dim=192, input_resolution=(32,32), num_heads = 6, window_size=4, ),
        )
        self.drop2=SENet_decoder(192, 192, blocks=2, drop_rate2=2) 
        self.att3=nn.Sequential(
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=4, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=4, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=4, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=4, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=4, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=4, ),
        )
        self.drop3=SENet_decoder(384, 384, blocks=2, drop_rate2=2) 
        self.att4=nn.Sequential(
            SwinTransformerBlock(dim=768, input_resolution=(8,8), num_heads = 24, window_size=4, ),
            SwinTransformerBlock(dim=768, input_resolution=(8,8), num_heads = 24, window_size=4, ),
        )
        self.drop4=SENet_decoder(768, 384, blocks=2, drop_rate2=2) # 768,4,4
        
        self.att_layers=nn.ModuleList([self.att1,self.att2,self.att3,self.att4])
        self.drop_layers=nn.ModuleList([self.drop1,self.drop2,self.drop3,self.drop4])

        '''
        linear
        '''
        self.linear = nn.Linear(768*2*2, message_length) 
        self.activation = nn.ReLU(True)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, image_with_wm):
        x = image_with_wm

        x = self.layers(x)

        for i in range(4):

            B,C,H,W = x.shape # b,c,h,w
            x = (x.view(B,C,H*W)).permute(0, 2, 1)
            x = self.att_layers[i](x)
            x = (x.permute(0, 2, 1)).view(B,C,H,W)
            x = self.drop_layers[i](x)
        
        x = self.pool(x)

        x = x.view(x.shape[0], -1)

        x = self.linear(x)
        x = self.activation(x)
        return x

class Decoder_Swin_v3(nn.Module):
    def __init__(self, message_length=64, decoder_channels=96, in_channel=3, window_size=8):

        super().__init__()
        self.channels = decoder_channels

        self.layers = nn.Sequential(
            ConvBNRelu(in_channel, self.channels), 
            SENet(self.channels, self.channels, blocks=4),
            SENet_decoder(self.channels, self.channels//2, blocks=2, drop_rate2=2), 
        )

        self.att1=nn.Sequential(
            SwinTransformerBlock(dim=96, input_resolution=(64,64), num_heads = 3, window_size=window_size, ),
            SwinTransformerBlock(dim=96, input_resolution=(64,64), num_heads = 3, window_size=window_size, ),
        )
        self.drop1=SENet_decoder(96, 96, blocks=2, drop_rate2=2) 
        self.att2=nn.Sequential(
            SwinTransformerBlock(dim=192, input_resolution=(32,32), num_heads = 6, window_size=window_size, ),
            SwinTransformerBlock(dim=192, input_resolution=(32,32), num_heads = 6, window_size=window_size, ),
        )
        self.drop2=SENet_decoder(192, 192, blocks=2, drop_rate2=2) 
        self.att3=nn.Sequential(
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
            SwinTransformerBlock(dim=384, input_resolution=(16,16), num_heads = 12, window_size=window_size, ),
        )
        self.drop3=SENet_decoder(384, 384, blocks=2, drop_rate2=2) 
        self.att4=nn.Sequential(
            SwinTransformerBlock(dim=768, input_resolution=(8,8), num_heads = 24, window_size=window_size//2, ),
            SwinTransformerBlock(dim=768, input_resolution=(8,8), num_heads = 24, window_size=window_size//2, ),
        )
        self.drop4=SENet_decoder(768, 384, blocks=2, drop_rate2=2) 
        
        self.att_layers=nn.ModuleList([self.att1,self.att2,self.att3,self.att4])
        self.drop_layers=nn.ModuleList([self.drop1,self.drop2,self.drop3,self.drop4])

        '''
        linear
        '''
        self.linear = nn.Linear(768 * 4 * 4, message_length) 
        self.activation = nn.ReLU(True)

    def forward(self, image_with_wm):
        x = image_with_wm

        x = self.layers(x)

        for i in range(4):

            B,C,H,W = x.shape # b,c,h,w
            x = (x.view(B,C,H*W)).permute(0, 2, 1)
            x = self.att_layers[i](x)
            x = (x.permute(0, 2, 1)).view(B,C,H,W)
            x = self.drop_layers[i](x)
        
        #x = self.pool(x)

        x = x.flatten(start_dim=1)

        x = self.linear(x)
        x = self.activation(x)
        return x
