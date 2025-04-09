import csv, torch
from torchvision import utils as vutils
import os, shutil
import numpy as np
from torchvision import transforms

def write_csv(filename, epoch, perrow, save=True):
    """
    write result to csv file
    """
    row_to_write = ['epoch','loss_d_cov_bce','loss_d_enc_bce','loss_g_adv_bce',
							'loss_g_enc_mse','loss_g_dec_mse','loss_g','psnr','ssim','ipips','acc_rate']
    if save==False:
        if epoch==0:
            print(row_to_write)
        row_to_write = [epoch] + perrow
        print(row_to_write)
        return
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 0:
            writer.writerow(row_to_write)
        row_to_write = [epoch] + perrow
        writer.writerow(row_to_write)

def write_csv_test(filename, noise, perrow, save=True):
    """
    write result to csv file
    """
    row_to_write = ['loss_d_cov_bce','loss_d_enc_bce','loss_g_adv_bce',
                    'loss_g_enc_mse','loss_g_dec_mse','loss_g','psnr','ssim','ipips','APD_H','acc_rate','acc_batch',
                    'js_div','wass_dist','fid',
                    'noise']
    if save==False:
        if noise=='Identity()':
            print(row_to_write)
        row_to_write =  perrow + [noise]
        print(row_to_write)
        return
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if noise == 'Identity()':
            writer.writerow(row_to_write)
        row_to_write = perrow + [noise]
        writer.writerow(row_to_write)


def setup_seed(seed):
    """
    set random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def make_grid_save_image(images_all, path, save_img_num=6, save=True):
    images, encoded_images, noised_images, diff_images, diff_images_linear=images_all
    plotimg=vutils.make_grid(torch.cat([images,encoded_images,
									  noised_images,diff_images,diff_images_linear], dim=0),nrow=save_img_num,padding=1)
    if(save): vutils.save_image(plotimg, path)
    else: return plotimg

def save_preprocess(images, encoded_images, noised_images, save_img_num=6, norm=True):
    idx=min(len(images), save_img_num)
    images=images[:idx].detach()
    encoded_images=encoded_images[:idx].detach()
    noised_images=noised_images[:idx].detach()

    diff_images=encoded_images-images
    if norm:
        images=(images+1)/2
        encoded_images=(encoded_images+1)/2
        noised_images=(noised_images+1)/2
        diff_images = (encoded_images - images + 1) / 2

    diff_images_linear = diff_images.clone()
    R = diff_images_linear[:, 0, :, :]
    G = diff_images_linear[:, 1, :, :]
    B = diff_images_linear[:, 2, :, :]
    diff_images_linear[:, 0, :, :] = 0.299 * R + 0.587 * G + 0.114 * B
    diff_images_linear[:, 1, :, :] = diff_images_linear[:, 0, :, :]
    diff_images_linear[:, 2, :, :] = diff_images_linear[:, 0, :, :]
    diff_images_linear = torch.abs(diff_images_linear * 2 - 1)

    # maximize diff
    for id in range(diff_images_linear.shape[0]):
        diff_images_linear[id] = (diff_images_linear[id] - diff_images_linear[id].min()) / (
                diff_images_linear[id].max() - diff_images_linear[id].min())
        
    return [images, encoded_images, noised_images, diff_images, diff_images_linear]


def is_intersect(x1, y1, x2, y2, w, h):
    if abs(x1-x2) < w and abs(y1-y2) < h:
        return True
    else:
        return False






def padding_save(img, mask, size=512, padding_mode='constant', fill=0, pad_if_needed=True):
    H_orig, W_orig= img.shape[2], img.shape[3]
    H_target, W_target = size, size

    with torch.no_grad():
        if pad_if_needed and (H_orig < H_target or W_orig < W_target):
            left=(W_target-W_orig) // 2 
            right=(W_target - W_orig + 1) // 2 
            top=(H_target - H_orig) // 2
            down=(H_target - H_orig + 1) // 2
            img = F.pad(img, pad=(left, right, top, down), mode='constant', value=fill)
            mask = F.pad(mask, pad=(left, right, top, down), mode='constant', value=fill)

            W_orig, H_orig = img.shape[3], img.shape[2]  
            
        x_left = np.random.randint(0, W_orig - W_target + 1)
        y_top = np.random.randint(0, H_orig - H_target + 1)
        

        img_cropped = img[:,:, y_top:y_top+H_target, x_left:x_left+W_target]
        mask_cropped = mask[:,:, y_top:y_top+H_target, x_left:x_left+W_target]

        return img_cropped, mask_cropped


def save_image_PIL(images_all, path, save=True):
    output_dirs = {
        "images": f"{path}/images",
        "encoded_images": f"{path}/encoded_images",
        "noised_images": f"{path}/noised_images",
        "diff_images": f"{path}/diff_images",
        "diff_images_linear": f"{path}/diff_images_linear",
    }

    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    def save_images(image_batch, output_dir, prefix="image"):
        bs, _, height, width = image_batch.shape
        for i in range(bs):
            transforms.ToPILImage()(image_batch[i]).save(os.path.join(output_dir, f"{prefix}_{i}.png"))

    batch_names = ["images", "encoded_images", "noised_images", "diff_images", "diff_images_linear"]
    for name, batch in zip(batch_names, images_all):
        save_images(batch, output_dirs[name], prefix=name)