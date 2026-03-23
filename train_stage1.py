import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

import torch
import numpy as np
import shutil
import argparse
from torch.utils.data import DataLoader
import kornia
from lpips import LPIPS
import pytorch_lightning as pl

from models.noiser import Noise
from models.encoder import Encoder_v2 as Encoder
from models.decoder import Decoder_Swin_v3 as Decoder
from models.discriminator import Discriminator
from utils.util import write_csv, save_preprocess, make_grid_save_image
import config as cf

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str,default='My_exp/HiWL_Train_Stage1',  help='path')
args = parser.parse_args()

cf.save_path=args.save_path

class WatermarkModel(pl.LightningModule):
    def __init__(self, ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.encoder = Encoder(cf.H, cf.W, cf.message_length, encoder_channels = cf.encoder_channels)
        self.decoder = Decoder(cf.message_length)

        self.discriminator = Discriminator()
        self.train_noiser = Noise(cf.train_noise_layer)

        self.val_noiser = Noise(cf.val_noise_layer)
        self.val_noiser.eval()
        for param in self.val_noiser.parameters():
            param.requires_grad = False

        self.criterion_BCE = torch.nn.BCEWithLogitsLoss()
        self.criterion_MSE = torch.nn.MSELoss()
        self.ssim_loss = kornia.losses.MS_SSIMLoss(data_range = 2, alpha = 0.5)

        self.lpips_vgg = LPIPS(net='vgg')
        self.lpips_vgg.eval()
        for param in self.lpips_vgg.parameters():
            param.requires_grad = False

        self.encoder_weight = 0.2
        self.decoder_weight = 1.0
        self.discriminator_weight = 1e-3

        self.register_buffer('label_cover', torch.ones((cf.batch_size, 1)))
        self.register_buffer('label_encoded', torch.zeros((cf.batch_size, 1)))

    def forward(self, images, messages):
        wm_images = self.encoder(images, messages)
        noise_images, _ = self.train_noiser([wm_images, images])
        out_messages = self.decoder(noise_images)
        return wm_images, noise_images, out_messages

    def configure_optimizers(self):
        opt_ed = torch.optim.AdamW([
            {'params': self.encoder.parameters(), 'lr': cf.lr}, 
            {'params': self.decoder.parameters(), 'lr': cf.lr}, 
        ])
        opt_discriminator = torch.optim.AdamW(self.discriminator.parameters(), lr=cf.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ed, T_max=cf.epochs, eta_min=1e-6)
        return [opt_ed, opt_discriminator], [scheduler]

    def decoded_message_acc_rate_bit(self, mess1, mess2):
        mess2 = mess2.detach().round().clamp(0, 1)
        error_rate = torch.sum(torch.abs(mess1.detach() - mess2)) / (mess2.shape[0] * mess2.shape[1])
        return 1 - error_rate

    def training_step(self, batch, batch_idx):
        images = batch
        opt_ed, opt_d = self.optimizers()
        messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], cf.message_length))).to(images.device)
        
        wm_images, noise_images, out_messages = self(images, messages)

        '''
        discriminator
        '''
        opt_d.zero_grad()
        d_label_cover = self.discriminator(images)
        d_cover_loss =self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
        d_label_encoded = self.discriminator(wm_images.detach())
        d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
        d_loss = d_cover_loss + d_encoded_loss

        self.manual_backward(d_loss)
        opt_d.step()

        '''
        generator
        '''
        opt_ed.zero_grad()
        with torch.no_grad():
            g_label_decoded = self.discriminator(wm_images)
        g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
        train_encoder_mseloss=self.criterion_MSE(wm_images, images)
        train_msssim_loss = self.ssim_loss(images, wm_images)
        g_loss_on_encoder = 0.005*train_msssim_loss + train_encoder_mseloss
        g_loss_on_decoder = self.criterion_MSE(out_messages, messages)
        g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + self.decoder_weight * g_loss_on_decoder

        self.manual_backward(g_loss)
        opt_ed.step()

    def training_step_update_dividly (self, batch, batch_idx):
            images = batch
            opt_ed, opt_d = self.optimizers()
            messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], cf.message_length))).to(images.device)
            
            wm_images, noise_images, out_messages = self(images, messages)
            
            '''
            discriminator
            '''
            opt_d.zero_grad()
            d_label_cover = self.discriminator(images)
            d_cover_loss =self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
            d_label_encoded = self.discriminator(wm_images.detach())
            d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
            d_loss=d_cover_loss+d_encoded_loss
            self.manual_backward(d_loss)
            opt_d.step()

            '''
            generator
            '''
            opt_ed.zero_grad()
            with torch.no_grad():
                g_label_decoded = self.discriminator(wm_images)
            g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
            train_encoder_mseloss=self.criterion_MSE(wm_images, images)
            train_msssim_loss = self.ssim_loss(images, wm_images)
            g_loss_on_encoder = 0.005*train_msssim_loss + train_encoder_mseloss
            g_loss_on_decoder = self.criterion_MSE(out_messages, messages)
            g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + self.decoder_weight * g_loss_on_decoder
            self.manual_backward(g_loss)
            opt_ed.step()


    def validation_step(self, batch, batch_idx):
        images = batch
        messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], cf.message_length))).to(images.device)
        
        wm_images, noise_images, out_messages = self(images, messages)
        
        '''
        discriminator
        '''
        d_label_cover = self.discriminator(images)
        d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
        d_label_encoded = self.discriminator(wm_images.detach())
        d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
        d_loss = d_cover_loss + d_encoded_loss

        '''
        generator
        '''
        with torch.no_grad():
            g_label_decoded = self.discriminator(wm_images)
        g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])
        train_encoder_mseloss = self.criterion_MSE(wm_images, images)
        train_msssim_loss = self.ssim_loss(images, wm_images)
        g_loss_on_encoder = 0.005*train_msssim_loss + train_encoder_mseloss
        g_loss_on_decoder = self.criterion_MSE(out_messages, messages)
        g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + self.decoder_weight * g_loss_on_decoder

        loss = d_loss + g_loss

        with torch.no_grad():
            psnr = kornia.losses.psnr_loss(wm_images.detach(), images, max_val=2)
            ssim = 1 - 2 * kornia.losses.ssim_loss(wm_images.detach(), images, max_val = 1.0, window_size = 5, reduction = "mean")
            dist = self.lpips_vgg(wm_images.detach(), images).mean()
            acc_rate = self.decoded_message_acc_rate_bit(messages, out_messages)

            res = {
                "acc_rate": acc_rate, 
                "psnr": psnr,
                "ssim": ssim,
                "lpips": dist,
                "g_loss": g_loss,
                "g_loss_on_discriminator": g_loss_on_discriminator,
                "g_loss_on_encoder": g_loss_on_encoder,
                "g_loss_on_decoder": g_loss_on_decoder,
                "d_cover_loss": d_cover_loss,
                "d_encoded_loss": d_encoded_loss
            }
        
            if self.trainer.global_rank == 0 and batch_idx == 0:
                make_grid_save_image(save_preprocess(images, wm_images, noise_images, min(6,cf.batch_size), norm=True), os.path.join(cf.save_path+f'/images', f'epoch-{self.current_epoch}.png'), min(6,cf.batch_size))
        
        #self.log_dict(res, prog_bar=True, on_step= False, on_epoch=True, logger=True, sync_dist=True)
        self.log_dict(res, prog_bar=True, on_step=True, on_epoch=False, logger=True, sync_dist=True)


@pl.utilities.rank_zero_only  
def backup():
    os.makedirs(cf.save_path, exist_ok=True)
    os.makedirs(os.path.join(cf.save_path, 'pth'), exist_ok=True)
    os.makedirs(os.path.join(cf.save_path, 'pth2'), exist_ok=True)
    os.makedirs(os.path.join(cf.save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(cf.save_path, 'images2'), exist_ok=True)
    def copy_code(new_directory):
        current_script_path = os.path.abspath(__file__)
        os.makedirs(new_directory, exist_ok=True)
        script_name = os.path.basename(current_script_path)
        target_path = os.path.join(new_directory, script_name)
        shutil.copy2(current_script_path, target_path)
    copy_code(cf.save_path)
    shutil.copy2('config.py', os.path.join(cf.save_path, 'config.py'))
    shutil.copytree('models', os.path.join(cf.save_path, 'models'), dirs_exist_ok=True)
    with open(os.path.join(cf.save_path, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print("backup finish!!!")
backup()

loader=cf.WMDataModule(
    'YOUR_TRAIN_DIR',
    'YOUR_VALID_DIR',
    cf.batch_size, 
    cf.transform,
    cf.transform_test
)




model = WatermarkModel()


trainer = pl.Trainer(
    logger = pl.loggers.CSVLogger(save_dir=cf.save_path, name="logs"),
    max_epochs = cf.epochs, 
    accelerator = 'gpu',
    strategy = 'ddp_find_unused_parameters_true',
    devices = -1, 
    val_check_interval = 500,
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_top_k = 200,       
            monitor = "epoch",     
            mode = "max",          
            every_n_epochs = 1,    
            # filename = "epoch_{epoch:02d}",  
        ),
        # pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
    ]
)

trainer.fit(model, loader)
