import os
import sys
import argparse
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.generator import UNet3DGenerator
from models.discriminator import Discriminator3D
from data.dataset import MIPDataset
from utils.losses import GANLoss, MAELoss, DiscriminatorLoss


class TeacherTrainer:
    def __init__(self, config, direction='axial'):
        self.config = config
        self.direction = direction
        self.device = config.device
        self.mip_key_3t = f'mip_3t_{direction}'
        self.mip_key_7t = f'mip_7t_{direction}'
        
        self.generator = UNet3DGenerator(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_features=config.base_features,
            num_layers=config.num_layers
        ).to(self.device)
        
        self.discriminator = Discriminator3D(
            in_channels=config.in_channels * 2,
            base_features=config.base_features,
            num_layers=config.disc_num_layers
        ).to(self.device)
        
        if config.num_gpus > 1:
            self.generator = nn.DataParallel(self.generator, device_ids=config.gpu_ids)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=config.gpu_ids)
        
        self.gan_loss = GANLoss().to(self.device)
        self.mae_loss = MAELoss().to(self.device)
        self.disc_loss = DiscriminatorLoss().to(self.device)
        
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        self.schedulers = {
            'g': optim.lr_scheduler.LambdaLR(
                self.g_optimizer,
                lr_lambda=self._get_lr_lambda(config)
            ),
            'd': optim.lr_scheduler.LambdaLR(
                self.d_optimizer,
                lr_lambda=self._get_lr_lambda(config)
            )
        }
        
        self.checkpoint_dir = os.path.join(config.checkpoint_dir, f'teacher_{direction}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.log_dir = os.path.join(config.log_dir, f'teacher_{direction}')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _get_lr_lambda(self, config):
        def lr_lambda(epoch):
            if epoch < config.lr_decay_start:
                return 1.0
            else:
                progress = (epoch - config.lr_decay_start) / (config.num_epochs - config.lr_decay_start)
                return max(0.0, 1.0 - progress)
        return lr_lambda
    
    def train_epoch(self, train_loader, epoch):
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_gan_loss = 0.0
        total_mae_loss = 0.0
        
        if self.config.time_verbose:
            import time
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            if self.config.time_verbose:
                step_start = time.time()
                data_load_time = step_start
            
            mip_3t = batch[self.mip_key_3t].to(self.device)
            mip_7t = batch[self.mip_key_7t].to(self.device)
            
            mip_3t = mip_3t.squeeze(0)
            mip_7t = mip_7t.squeeze(0)
            
            if self.config.time_verbose:
                forward_start = time.time()
            
            fake_7t = self.generator(mip_3t)
            
            d_input_real = torch.cat([mip_3t, mip_7t], dim=1)
            d_input_fake = torch.cat([mip_3t, fake_7t.detach()], dim=1)
            
            d_loss, d_real_loss, d_fake_loss = self.disc_loss(
                self.discriminator, d_input_real, d_input_fake
            )
            
            if self.config.time_verbose:
                forward_end = time.time()
                forward_time = forward_end - forward_start
                backward_start = time.time()
            
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            if self.config.time_verbose:
                d_backward_end = time.time()
                d_backward_time = d_backward_end - backward_start
                g_forward_start = time.time()
            
            fake_7t = self.generator(mip_3t)
            g_input_fake = torch.cat([mip_3t, fake_7t], dim=1)
            
            gan_loss = self.gan_loss(self.discriminator(g_input_fake), True)
            mae_loss = self.mae_loss(fake_7t, mip_7t)
            
            g_loss = gan_loss + self.config.alpha * mae_loss
            
            if self.config.time_verbose:
                g_forward_end = time.time()
                g_forward_time = g_forward_end - g_forward_start
                g_backward_start = time.time()
            
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()
            
            if self.config.time_verbose:
                g_backward_end = time.time()
                g_backward_time = g_backward_end - g_backward_start
                step_end = time.time()
                total_step_time = step_end - step_start
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_gan_loss += gan_loss.item()
            total_mae_loss += mae_loss.item()
            
            self.global_step += 1
            
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('Loss/generator', g_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/discriminator', d_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/gan', gan_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/mae', mae_loss.item(), self.global_step)
            
            if self.config.time_verbose:
                postfix_dict = {
                    'G_loss': g_loss.item(),
                    'D_loss': d_loss.item(),
                    'GAN': gan_loss.item(),
                    'MAE': mae_loss.item(),
                    'data_time': f'{data_load_time*1000:.1f}ms',
                    'forward_time': f'{forward_time*1000:.1f}ms',
                    'd_back': f'{d_backward_time*1000:.1f}ms',
                    'g_forward': f'{g_forward_time*1000:.1f}ms',
                    'g_back': f'{g_backward_time*1000:.1f}ms',
                    'total': f'{total_step_time*1000:.1f}ms'
                }
            else:
                postfix_dict = {
                    'G_loss': g_loss.item(),
                    'D_loss': d_loss.item(),
                    'GAN': gan_loss.item(),
                    'MAE': mae_loss.item()
                }
            
            pbar.set_postfix(postfix_dict)
        
        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
        avg_gan_loss = total_gan_loss / len(train_loader)
        avg_mae_loss = total_mae_loss / len(train_loader)
        
        return avg_g_loss, avg_d_loss, avg_gan_loss, avg_mae_loss
    
    def validate(self, val_loader, epoch):
        self.generator.eval()
        
        total_g_loss = 0.0
        total_mae_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                mip_3t = batch[self.mip_key_3t].to(self.device)
                mip_7t = batch[self.mip_key_7t].to(self.device)
                
                mip_3t = mip_3t.squeeze(0)
                mip_7t = mip_7t.squeeze(0)        
                
                fake_7t = self.generator(mip_3t)
                
                mae_loss = self.mae_loss(fake_7t, mip_7t)
                
                total_g_loss += mae_loss.item()
                total_mae_loss += mae_loss.item()
        
        avg_g_loss = total_g_loss / len(val_loader)
        avg_mae_loss = total_mae_loss / len(val_loader)
        
        self.writer.add_scalar('Val/generator_loss', avg_g_loss, epoch)
        self.writer.add_scalar('Val/mae_loss', avg_mae_loss, epoch)
        
        return avg_g_loss, avg_mae_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader):
        print(f"Training Teacher Model for {self.direction} direction")
        print(f"Device: {self.device}")
        print(f"Number of GPUs: {self.config.num_gpus}")
        if self.config.gpu_ids:
            print(f"GPU IDs: {self.config.gpu_ids}")
        
        for epoch in range(self.config.num_epochs):
            avg_g_loss, avg_d_loss, avg_gan_loss, avg_mae_loss = self.train_epoch(train_loader, epoch)
            
            self.schedulers['g'].step()
            self.schedulers['d'].step()
            
            current_lr = self.g_optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train - G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
            print(f"  Train - GAN: {avg_gan_loss:.4f}, MAE: {avg_mae_loss:.4f}")
            print(f"  Learning rate: {current_lr:.6f}")
            
            if (epoch + 1) % self.config.val_interval == 0:
                avg_val_loss, avg_val_mae = self.validate(val_loader, epoch)
                print(f"  Val - Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}")
                
                is_best = avg_val_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_val_loss
                    print(f"  New best model! Loss: {self.best_loss:.4f}")
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, is_best=is_best if (epoch + 1) % self.config.val_interval == 0 else False)
                print(f"  Checkpoint saved at epoch {epoch+1}")
        
        self.save_checkpoint(self.config.num_epochs - 1, is_best=False)
        print("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Teacher Model for AU-MIPGAN')
    parser.add_argument('--direction', type=str, default='axial', 
                        choices=['axial', 'coronal', 'sagittal'],
                        help='MIP direction for training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use (e.g., 0)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Preprocessed data directory (default: project_root/preprocessed)')
    args = parser.parse_args()              
    
    config = Config()
    config.set_gpu(args.gpu)
    
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        for key, value in vars(custom_config).items():
            if not key.startswith('_'):
                setattr(config, key, value)
    
    print(f"Configuration:")
    print(config)
    
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(config.project_root, 'preprocessed')
    
    train_data_dir = os.path.join(data_dir, 'train', args.direction)
    val_data_dir = os.path.join(data_dir, 'val', args.direction)
    
    num_train_samples = len(glob.glob(os.path.join(train_data_dir, '*_3t.nii.gz')))
    num_val_samples = len(glob.glob(os.path.join(val_data_dir, '*_3t.nii.gz')))
    
    print(f"\nLoading data from: {data_dir}")
    print(f"  Train samples: {num_train_samples}")
    print(f"  Val samples: {num_val_samples}")
    
    train_dataset = MIPDataset(
        data_dir=train_data_dir,
        num_samples=num_train_samples,
        patch_size=config.patch_size,
        mode='train',
        batch_size=config.batch_size,
        batch_per_sample=config.batch_per_sample,
        dataset_type=args.direction
    )
    
    val_dataset = MIPDataset(
        data_dir=val_data_dir,
        num_samples=num_val_samples,
        patch_size=config.patch_size,
        mode='val',
        batch_size=config.batch_size,
        batch_per_sample=config.batch_per_sample,
        dataset_type=args.direction
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=min(config.num_workers, os.cpu_count()),
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(config.num_workers, os.cpu_count()),
        pin_memory=config.pin_memory
    )
    
    trainer = TeacherTrainer(config, direction=args.direction)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
