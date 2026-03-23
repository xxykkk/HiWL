import csv, sys, os, time
# os.environ['CUDA_VISIBLE_DEVICES']='2' 
import numpy as np
import torch, argparse, kornia
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets, transforms, utils
from torch.utils import data
from lpips import LPIPS
import shutil
import config_test as cf

from models.noiser import Noise
from models.encoder import Encoder_v2 as Encoder
from models.decoder import Decoder_Swin_v3 as Decoder 
from models.discriminator import Discriminator
from utils.util import save_preprocess, make_grid_save_image, write_csv_test
from utils.dataset import MyDataset

device = torch.device('cuda')

import piq
fid_metric = piq.FID()
from torchvision.models import inception_v3
import torch.nn.functional as F
model_Icpv3 = inception_v3(pretrained=True, transform_input=False)
model_Icpv3.fc = torch.nn.Identity()
model_Icpv3.eval().to(device)

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=False, default=r'YOUR_CKPT_PATH', help='Path to the model checkpoint')
parser.add_argument('--epoch', type=int, required=False, help='The k-th epoch .pth file in the model path')
parser.add_argument('--type', type=str, required=False, default='diff', help='Test model difference effect')
args = parser.parse_args()



class EncoderDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder(cf.H, cf.W, cf.message_length, encoder_channels=cf.encoder_channels)
        self.decoder = Decoder(cf.message_length)
    def forward(self, images):
        return images

# --- Model Loading ---
lpips_vgg = LPIPS(net='vgg').to(device)
val_noiser = Noise(cf.val_noise_layer)
ed = EncoderDecoder().to(device)
discriminator = Discriminator().to(device)

# --- Checkpoint Loading ---
load_path = args.path
number = args.epoch
print('Model path and selected epoch file:')
print(load_path, number)

state_dict = torch.load(load_path, map_location=device)['state_dict']

# Filter state dict for sub-modules
encoder_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
decoder_state_dict = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")}
discriminator_state_dict = {k.replace("discriminator.", ""): v for k, v in state_dict.items() if k.startswith("discriminator.")}

# Load weights
ed.encoder.load_state_dict(encoder_state_dict)
ed.decoder.load_state_dict(decoder_state_dict)
discriminator.load_state_dict(discriminator_state_dict)

# --- Loss and Hyperparameters ---
criterion_BCE = nn.BCEWithLogitsLoss().to(device)
criterion_MSE = nn.MSELoss().to(device)
ssim_loss = kornia.losses.MS_SSIMLoss(data_range=2, alpha=0.5).to(device)

encoder_weight = 0.2
decoder_weight = 1
discriminator_weight = 1e-3 

label_cover = torch.ones((cf.batch_size, 1), device=device)
label_encoded = torch.zeros((cf.batch_size, 1), device=device)

# --- Helper Functions ---
def data_generator(dataloader):
    return iter(dataloader)

def custom_loss(img1, img2):
    larger = img1 > 1
    smaller = img1 < -1
    loss = 0
    if len(img1[larger]) > 0: 
        loss += 10 * (criterion_MSE(img1[larger], torch.ones_like(img1[larger])))
    if len(img1[smaller]) > 0: 
        loss += 10 * (criterion_MSE(img1[smaller], torch.ones_like(img1[smaller]) - 2))
    return criterion_MSE(img1, img2) + loss

def decoded_message_acc_rate_bit(mess1, mess2):
    mess2 = mess2.detach().cpu().numpy().round().clip(0, 1)
    error_rate = np.sum(np.abs(mess1.detach().cpu().numpy() - mess2)) / (
            mess2.shape[0] * mess2.shape[1])
    return 1 - error_rate

def decoded_message_acc_rate_batch(mess1, mess2, tolerance=1e-4):
    x1 = mess2.detach().round().clip(0, 1)
    x2 = mess1.detach()
    # Check if elements are equal within tolerance
    match = torch.isclose(x1, x2, atol=tolerance)  
    # Check if samples match completely
    match_per_sample = match.view(x1.shape[0], -1).all(dim=1)  
    num_correct = match_per_sample.sum().item()
    accuracy = num_correct / x1.shape[0]
    return accuracy

# --- Initialization of Directories ---
if not os.path.exists(cf.save_path): 
    os.mkdir(cf.save_path)
if not os.path.exists(cf.save_path + '/images'): 
    os.mkdir(cf.save_path + '/images')
if not os.path.exists(cf.save_path + '/pth'): 
    os.mkdir(cf.save_path + '/pth')

# Create empty log file
log_filename = f'{args.type, args.epoch}.txt'
with open(os.path.join(cf.save_path, log_filename), 'a') as file: 
    pass

def copy_code(new_directory):
    current_script_path = os.path.abspath(__file__)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    script_name = os.path.basename(current_script_path)
    target_path = os.path.join(new_directory, script_name)
    shutil.copy2(current_script_path, target_path)

# Backup current scripts and configs
copy_code(cf.save_path)
shutil.copy2('config_test.py', os.path.join(cf.save_path, 'config_test.py'))


import torch.nn.functional as F
def compute_js_divergence(imgs1, imgs2, bins=8, eps=1e-10):
    """
    Compute JS divergence between two batches of images.
    
    Args:
        imgs1 (torch.Tensor): Shape (bs, 3, H, W), range [-1, 1]
        imgs2 (torch.Tensor): Shape (bs, 3, H, W), range [-1, 1]
        bins (int): Number of bins for histogram
        eps (float): Small value to avoid log(0)
    
    Returns:
        torch.Tensor: JS divergence (scalar, averaged over batch)
    """
    assert imgs1.shape == imgs2.shape, "Input shapes must match!"
    bs = imgs1.shape[0]

    imgs1_flat = ((imgs1 / 2 + 0.5) * 255).view(bs, -1)  # (bs, 3*H*W)
    imgs2_flat = ((imgs2 / 2 + 0.5) * 255).view(bs, -1)

    hist1 = torch.stack([
        torch.histc(img, bins=bins, min=0, max=255) 
        for img in imgs1_flat
    ])
    hist2 = torch.stack([
        torch.histc(img, bins=bins, min=0, max=255) 
        for img in imgs2_flat
    ])

    hist1 = (hist1 / hist1.sum(dim=1, keepdim=True) + eps)
    hist2 = (hist2 / hist2.sum(dim=1, keepdim=True) + eps)

    m = 0.5 * (hist1 + hist2)
    kl_pm = F.kl_div(torch.log(m), hist1, reduction='none').sum(dim=1)  # KL(M || P)
    kl_qm = F.kl_div(torch.log(m), hist2, reduction='none').sum(dim=1)  # KL(M || Q)
    js = 0.5 * (kl_pm + kl_qm)  # JS(P || Q)

    return js.mean() 

def compute_wasserstein(imgs1, imgs2, p=2):
    """
    Compute Wasserstein distance (EMD) between two batches of images.
    
    Args:
        imgs1 (torch.Tensor): Shape (bs, C, H, W), range [-1, 1]
        imgs2 (torch.Tensor): Shape (bs, C, H, W), range [-1, 1]
        p (int): Power for Wasserstein distance (1 or 2)
    
    Returns:
        torch.Tensor: Wasserstein distance (scalar, averaged over batch)
    """
    assert imgs1.shape == imgs2.shape, "Input shapes must match!"
    bs = imgs1.shape[0]
    
    flat1 = imgs1.view(bs, -1)  # (bs, C*H*W)
    flat2 = imgs2.view(bs, -1)
    
    sorted1, _ = torch.sort(flat1, dim=1)  # (bs, C*H*W)
    sorted2, _ = torch.sort(flat2, dim=1)
    
    emd = torch.mean(torch.abs(sorted1 - sorted2)**p, dim=1) ** (1/p)  # (bs,)
    
    return emd.mean() 


def compute_fid(images1, images2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate the FID (Fréchet Inception Distance) score between two batches of images.
    
    Args:
        images1 (Tensor): [bs, 3, 128, 128], real images or the first set of images.
        images2 (Tensor): [bs, 3, 128, 128], generated images or the second set of images.
        device (str): Execution device, e.g., 'cuda' or 'cpu'.

    Returns:
        float: The calculated FID score.
    """
    # images1 = images1.float() / 255.0 if images1.max() > 1 else images1
    # images2 = images2.float() / 255.0 if images2.max() > 1 else images2

    images1 = F.interpolate(images1, size=(299, 299), mode='bicubic', align_corners=False)
    images2 = F.interpolate(images2, size=(299, 299), mode='bicubic', align_corners=False)

    with torch.no_grad():
        feats1 = model_Icpv3(images1)
        feats2 = model_Icpv3(images2)

    return fid_metric(feats1, feats2)



'''
Individual testing for each noise type
'''
if args.type == 'diff':
    # generalized test
    for k_noise in range(len(cf.val_noise_layer)):
        gen_test = data_generator(cf.dataloader_test2) # next gen get the images
        cnt = 0
        # save_all = np.zeros(12) # Replaced by the 25-element version below
        save_all = np.zeros(25)
        first_flag = True
        with torch.no_grad():
            for images in tqdm(cf.dataloader_test):
                images = images.to(device)
                messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], cf.message_length))).to(device)
                wm_images = ed.encoder(images, messages)
                diff = wm_images - images
                
                images = next(gen_test).to(device)
                wm_images = images + diff
                wm_images = torch.clamp(wm_images, -1, 1)
        
                noise_images, kind = val_noiser([wm_images, images], k_noise)
                assert kind == k_noise
                out_messages = ed.decoder(noise_images)

                d_label_cover = discriminator(images)
                d_cover_loss = criterion_BCE(d_label_cover, label_cover[:d_label_cover.shape[0]])
                d_label_encoded = discriminator(wm_images.detach())
                d_encoded_loss = criterion_BCE(d_label_encoded, label_encoded[:d_label_encoded.shape[0]])
                
                g_label_decoded = discriminator(wm_images)
                g_loss_on_discriminator = criterion_BCE(g_label_decoded, label_cover[:g_label_decoded.shape[0]])
                train_encoder_mseloss = custom_loss(wm_images, images)
                train_msssim_loss = ssim_loss(images, wm_images)
                g_loss_on_encoder = 0.005 * train_msssim_loss + train_encoder_mseloss
                g_loss_on_decoder = criterion_MSE(out_messages, messages)
                g_loss = discriminator_weight * g_loss_on_discriminator + encoder_weight * g_loss_on_encoder + decoder_weight * g_loss_on_decoder

                psnr = kornia.losses.psnr_loss(wm_images.detach(), images, max_val=2)
                ssim = 1 - 2 * kornia.losses.ssim_loss(wm_images.detach(), images, max_val=1.0, window_size=5, reduction="mean")
                dist = lpips_vgg(wm_images.detach(), images).mean()
                acc_rate = decoded_message_acc_rate_bit(messages, out_messages)
                acc_batch = decoded_message_acc_rate_batch(messages, out_messages)
                diffH = ((wm_images * 0.5 + 0.5) - (images * 0.5 + 0.5)).abs().mean() * 255
                
                js_div = compute_js_divergence(wm_images.detach(), images)
                wass_dist = compute_wasserstein(wm_images.detach(), images)
                fid = compute_fid(wm_images.detach(), images)
                psnr_piq = piq.psnr(wm_images.detach() * 0.5 + 0.5, images * 0.5 + 0.5, data_range=1.0)
                ssim_piq = piq.ssim(wm_images.detach() * 0.5 + 0.5, images * 0.5 + 0.5, data_range=1.0, reduction='mean')

                psnr_wm = kornia.losses.psnr_loss(diff.detach(), images, max_val=2)
                ssim_wm = 1 - 2 * kornia.losses.ssim_loss(diff.detach(), images, max_val=1.0, window_size=5, reduction="mean")
                dist_wm = lpips_vgg(diff.detach(), images).mean()
                js_div_wm = compute_js_divergence(diff.detach(), images)
                wass_dist_wm = compute_wasserstein(diff.detach(), images)
                fid_wm = compute_fid(diff.detach(), images)
                psnr_piq_wm = piq.psnr((diff.detach() * 0.25 + 0.5).clamp(0, 1), images * 0.5 + 0.5, data_range=1.0)
                ssim_piq_wm = piq.ssim((diff.detach() * 0.25 + 0.5).clamp(0, 1), images * 0.5 + 0.5, data_range=1.0, reduction='mean')
                
                res = {
                    "acc_rate": acc_rate,
                    "acc_batch": acc_batch,
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                    "lpips": dist.item(),
                    "g_loss": g_loss.item(),
                    "g_loss_on_discriminator": g_loss_on_discriminator.item(),
                    "g_loss_on_encoder": g_loss_on_encoder.item(),
                    "g_loss_on_decoder": g_loss_on_decoder.item(),
                    "d_cover_loss": d_cover_loss.item(),
                    "d_encoded_loss": d_encoded_loss.item(),
                    "diffH": diffH.item(),

                    "js_div": js_div.item(),
                    "wass_dist": wass_dist.item(),
                    "fid": fid.item(),
                    "psnr_piq": psnr_piq.item(),
                    "ssim_piq": ssim_piq.item(),

                    "psnr_wm": psnr_wm.item(),
                    "ssim_wm": ssim_wm.item(),
                    "lpips_wm": dist_wm.item(),
                    "js_div_wm": js_div_wm.item(),
                    "wass_dist_wm": wass_dist_wm.item(),
                    "fid_wm": fid_wm.item(),
                    "psnr_piq_wm": psnr_piq_wm.item(),
                    "ssim_piq_wm": ssim_piq_wm.item(),
                }
                cnt += 1
                save_all += np.array([res['d_cover_loss'], res['d_encoded_loss'],
                    res['g_loss_on_discriminator'], res['g_loss_on_encoder'], res['g_loss_on_decoder'],
                    res['g_loss'], res['psnr'], res['ssim'], res['lpips'], res["diffH"], res['acc_rate'], res['acc_batch'],
                    res['js_div'], res['wass_dist'], res['fid'], res['psnr_piq'], res['ssim_piq'],
                    res['psnr_wm'], res['ssim_wm'], res['lpips_wm'], res['js_div_wm'], res['wass_dist_wm'], res['fid_wm'], res['psnr_piq_wm'], res['ssim_piq_wm']
                    ])

                if first_flag:
                    first_flag = False
                    make_grid_save_image(save_preprocess(images, wm_images, noise_images, min(6, cf.batch_size), norm=True), os.path.join(cf.save_path + '/images', f'noise-{cf.val_noise_layer[k_noise]}.png'), min(6, cf.batch_size))
        
        para_show = np.round(save_all / cnt, 6).tolist()
        write_csv_test(os.path.join(cf.save_path, 'validation.csv'), cf.val_noise_layer[k_noise], para_show)

elif args.type == 'normal':
    # normal test
    for k_noise in range(len(cf.val_noise_layer)):
        cnt = 0
        save_all = np.zeros(12)
        first_flag = True
        with torch.no_grad():
            for images in tqdm(cf.dataloader_test):
                images = images.to(device)
                messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], cf.message_length))).to(device)
                wm_images = ed.encoder(images, messages)
                wm_images = torch.clamp(wm_images, -1, 1)

                noise_images, kind = val_noiser([wm_images, images], k_noise)
                assert kind == k_noise
                out_messages = ed.decoder(noise_images)
                
                d_label_cover = discriminator(images)
                d_cover_loss = criterion_BCE(d_label_cover, label_cover[:d_label_cover.shape[0]])
                d_label_encoded = discriminator(wm_images.detach())
                d_encoded_loss = criterion_BCE(d_label_encoded, label_encoded[:d_label_encoded.shape[0]])
                
                g_label_decoded = discriminator(wm_images)
                g_loss_on_discriminator = criterion_BCE(g_label_decoded, label_cover[:g_label_decoded.shape[0]])
                train_encoder_mseloss = criterion_MSE(wm_images, images)
                train_msssim_loss = ssim_loss(images, wm_images)
                g_loss_on_encoder = 0.005 * train_msssim_loss + train_encoder_mseloss
                g_loss_on_decoder = criterion_MSE(out_messages, messages)
                g_loss = discriminator_weight * g_loss_on_discriminator + encoder_weight * g_loss_on_encoder + decoder_weight * g_loss_on_decoder

                psnr = kornia.losses.psnr_loss(wm_images.detach(), images, max_val=2)
                ssim = 1 - 2 * kornia.losses.ssim_loss(wm_images.detach(), images, max_val=1.0, window_size=5, reduction="mean")
                dist = lpips_vgg(wm_images.detach(), images).mean()
                acc_rate = decoded_message_acc_rate_bit(messages, out_messages)
                acc_batch = decoded_message_acc_rate_batch(messages, out_messages)
                diffH = ((wm_images * 0.5 + 0.5) - (images * 0.5 + 0.5)).abs().mean() * 255
                
                res = {
                    "acc_rate": acc_rate,
                    "acc_batch": acc_batch,
                    "psnr": psnr.item(),
                    "ssim": ssim.item(),
                    "lpips": dist.item(),
                    "g_loss": g_loss.item(),
                    "g_loss_on_discriminator": g_loss_on_discriminator.item(),
                    "g_loss_on_encoder": g_loss_on_encoder.item(),
                    "g_loss_on_decoder": g_loss_on_decoder.item(),
                    "d_cover_loss": d_cover_loss.item(),
                    "d_encoded_loss": d_encoded_loss.item(),
                    "diffH": diffH.item()
                }
                cnt += 1
                save_all += np.array([res['d_cover_loss'], res['d_encoded_loss'],
                    res['g_loss_on_discriminator'], res['g_loss_on_encoder'], res['g_loss_on_decoder'],
                    res['g_loss'], res['psnr'], res['ssim'], res['lpips'], res["diffH"], res['acc_rate'], res['acc_batch']])

                if first_flag:
                    first_flag = False
                    make_grid_save_image(save_preprocess(images, wm_images, noise_images, min(6, cf.batch_size), norm=True), os.path.join(cf.save_path + '/images', f'noise-{cf.val_noise_layer[k_noise]}.png'), min(6, cf.batch_size))
        
        para_show = np.round(save_all / cnt, 6).tolist()
        write_csv_test(os.path.join(cf.save_path, 'validation.csv'), cf.val_noise_layer[k_noise], para_show)
else:
    input("Error")