from lpips import LPIPS
import torch
from PIL import Image
from torchvision import transforms
import kornia

lpips_vgg=LPIPS(net='vgg')#.cuda()

for i in range(0,100,10):
    dir1="../other_models/DWSF/output_train/image/val_epoch-"+str(i)+"-ori.png"
    dir2="../other_models/DWSF/output_train/image/val_epoch-"+str(i)+"-en.png"

    row1=Image.open(dir1).convert("RGB")
    row2=Image.open(dir2).convert("RGB")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img1=transform(row1).unsqueeze(0)
    img2=transform(row2).unsqueeze(0)

    ssim=1 - 2*kornia.losses.ssim_loss(img1, img2, max_val=1.0, window_size=5, reduction="mean")
    psnr=kornia.losses.psnr_loss(img1, img2, max_val=2.0)
    lpips=lpips_vgg(img1, img2)

    transform2=transforms.Compose([
        transforms.ToTensor(),

    ])

    img11=transform2(row1).unsqueeze(0)
    img22=transform2(row2).unsqueeze(0)
    ssim2=1 - 2*kornia.losses.ssim_loss(img11, img22, max_val=1.0, window_size=5, reduction="mean")
    psnr2=kornia.losses.psnr_loss(img11, img22, max_val=2.0)
    lpips2=lpips_vgg(img11, img22)


    print(ssim, psnr, lpips.item(), ssim2, psnr2, lpips2.item())




