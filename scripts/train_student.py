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
from models.amalgamation import MultiDirectionAmalgamation, AdaptiveWeightModule
from data.dataset import MIPDataset, MultiDirDataset
from utils.losses import GANLoss, MAELoss, KDLoss, ReconstructionLoss, AUAELoss


class StudentTrainer:
    def __init__(self, config, teacher_checkpoints=None):
        self.config = config
        self.device = config.device
        
        self.teachers = nn.ModuleList()
        self.directions = ['axial', 'coronal', 'sagittal']
        
        if teacher_checkpoints:
            for i, direction in enumerate(self.directions):
                teacher = UNet3DGenerator(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    base_features=config.base_features,
                    num_layers=config.num_layers,
                    return_features=True
                ).to(self.device)
                
                checkpoint = torch.load(teacher_checkpoints[i], map_location='cpu')
                if 'generator_state_dict' in checkpoint:
                    teacher.load_state_dict(checkpoint['generator_state_dict'])
                else:
                    teacher.load_state_dict(checkpoint)
                teacher = teacher.to(self.device)
                
                for param in teacher.parameters():
                    param.requires_grad = False
                teacher.eval()
                self.teachers.append(teacher)
        
        self.student = UNet3DGenerator(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_features=config.base_features,
            num_layers=config.num_layers,
            output_uncertainty=True,
            return_features=True
        ).to(self.device)
        
        self.discriminator = Discriminator3D(
            in_channels=config.in_channels * 2,
            base_features=config.base_features,
            num_layers=config.disc_num_layers
        ).to(self.device)
        
        feature_channels = self._get_feature_channels()
        self.amalgamation = MultiDirectionAmalgamation(
            feature_channels, num_directions=3
        ).to(self.device)
        
        self.adaptive_weights = AdaptiveWeightModule(
            num_directions=3, num_layers=len(feature_channels)
        ).to(self.device)
        
        if config.num_gpus > 1:
            self.student = nn.DataParallel(self.student, device_ids=config.gpu_ids)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=config.gpu_ids)
            self.amalgamation = nn.DataParallel(self.amalgamation, device_ids=config.gpu_ids)
            self.adaptive_weights = nn.DataParallel(self.adaptive_weights, device_ids=config.gpu_ids)
        
        self.gan_loss = GANLoss().to(self.device)
        self.mae_loss = MAELoss().to(self.device)
        self.kd_loss = KDLoss().to(self.device)
        self.recon_loss = ReconstructionLoss().to(self.device)
        self.au_ae_loss = AUAELoss().to(self.device)
        
        params = list(self.student.parameters()) + \
                 list(self.amalgamation.parameters()) + \
                 list(self.adaptive_weights.parameters())
        
        self.g_optimizer = optim.Adam(
            params,
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
        
        self.checkpoint_dir = os.path.join(config.checkpoint_dir, 'student')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.log_dir = os.path.join(config.log_dir, 'student')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _get_feature_channels(self):
        base = self.config.base_features
        num_layers = self.config.num_layers
        channels = []
        for i in range(num_layers):
            channels.append(base * (2 ** i))
        channels.append(base * (2 ** num_layers))
        return channels
    
    def _get_lr_lambda(self, config):
        def lr_lambda(epoch):
            if epoch < config.lr_decay_start:
                return 1.0
            else:
                progress = (epoch - config.lr_decay_start) / (config.num_epochs - config.lr_decay_start)
                return max(0.0, 1.0 - progress)
        return lr_lambda
    
    def _get_teacher_features(self, batch):
        teacher_features = []
        
        with torch.no_grad():
            for i, direction in enumerate(self.directions):
                mip_key = f'mip_3t_{direction}'
                mip_input = batch[mip_key].to(self.device)
                
                mip_input = mip_input.squeeze(0)

                _, features = self.teachers[i](mip_input)
                teacher_features.append(features)
        
        return teacher_features
    
    def train_epoch(self, train_loader, epoch):
        self.student.train()
        self.discriminator.train()
        self.amalgamation.train()
        self.adaptive_weights.train()
        
        for teacher in self.teachers:
            teacher.eval()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_gan_loss = 0.0
        total_mae_loss = 0.0
        total_kd_loss = 0.0
        total_recon_loss = 0.0
        
        if self.config.time_verbose:
            import time
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            if self.config.time_verbose:
                step_start = time.time()
                data_load_time = step_start
            
            patch_3t = batch['patch_3t'].to(self.device)
            patch_7t = batch['patch_7t'].to(self.device)
            
            if self.config.time_verbose:
                data_load_end = time.time()
                data_load_time = data_load_end - data_load_time
                teacher_start = time.time()
            
            patch_3t = patch_3t.squeeze(0)
            patch_7t = patch_7t.squeeze(0)  
            
            teacher_features = self._get_teacher_features(batch)
            
            if self.config.time_verbose:
                teacher_end = time.time()
                teacher_time = teacher_end - teacher_start
                amalgamation_start = time.time()
            
            amalgamated_features, reconstructed_features = self.amalgamation(teacher_features)
            
            if self.config.time_verbose:
                amalgamation_end = time.time()
                amalgamation_time = amalgamation_end - amalgamation_start
                weights_start = time.time()
            
            phi, omega = self.adaptive_weights()
            
            if self.config.time_verbose:
                weights_end = time.time()
                weights_time = weights_end - weights_start
                student_forward_start = time.time()
            
            fake_7t, uncertainty, student_features = self.student(patch_3t)
            
            if self.config.time_verbose:
                student_forward_end = time.time()
                student_forward_time = student_forward_end - student_forward_start
                disc_forward_start = time.time()
            
            d_input_real = torch.cat([patch_3t, patch_7t], dim=1)
            d_input_fake = torch.cat([patch_3t, fake_7t.detach()], dim=1)
            
            d_real_pred = self.discriminator(d_input_real)
            d_fake_pred = self.discriminator(d_input_fake)
            
            if self.config.time_verbose:
                disc_forward_end = time.time()
                disc_forward_time = disc_forward_end - disc_forward_start
                disc_backward_start = time.time()
            
            d_loss_real = self.gan_loss(d_real_pred, True)
            d_loss_fake = self.gan_loss(d_fake_pred, False)
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()
            
            if self.config.time_verbose:
                disc_backward_end = time.time()
                disc_backward_time = disc_backward_end - disc_backward_start
                student_re_forward_start = time.time()
            
            fake_7t, uncertainty, student_features = self.student(patch_3t)
            
            if self.config.time_verbose:
                student_re_forward_end = time.time()
                student_re_forward_time = student_re_forward_end - student_re_forward_start
                loss_compute_start = time.time()
            
            g_input_fake = torch.cat([patch_3t, fake_7t], dim=1)
            gan_loss = self.gan_loss(self.discriminator(g_input_fake), True)
            
            mae_loss = self.au_ae_loss(fake_7t, patch_7t, uncertainty)
            
            kd_loss = self.kd_loss(student_features, amalgamated_features, omega)
            
            recon_loss = self.recon_loss(reconstructed_features, teacher_features, phi)
            
            g_loss = gan_loss + self.config.alpha * mae_loss + \
                     self.config.beta * recon_loss + self.config.gamma * kd_loss
            
            if self.config.time_verbose:
                loss_compute_end = time.time()
                loss_compute_time = loss_compute_end - loss_compute_start
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
            total_kd_loss += kd_loss.item()
            total_recon_loss += recon_loss.item()
            
            self.global_step += 1
            
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar('Loss/generator', g_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/discriminator', d_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/gan', gan_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/mae', mae_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/kd', kd_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/recon', recon_loss.item(), self.global_step)
                
                for l in range(len(omega)):
                    self.writer.add_scalar(f'Omega/layer_{l}', omega[l].item(), self.global_step)
                
                for m in range(phi.shape[0]):
                    for l in range(phi.shape[1]):
                        self.writer.add_scalar(f'Phi/dir_{m}_layer_{l}', phi[m, l].item(), self.global_step)
            
            if self.config.time_verbose:
                postfix_dict = {
                    'G': g_loss.item(),
                    'D': d_loss.item(),
                    'MAE': mae_loss.item(),
                    'KD': kd_loss.item(),
                    'data': f'{data_load_time*1000:.1f}ms',
                    'teacher': f'{teacher_time*1000:.1f}ms',
                    'amalg': f'{amalgamation_time*1000:.1f}ms',
                    'weights': f'{weights_time*1000:.1f}ms',
                    's_fwd': f'{student_forward_time*1000:.1f}ms',
                    'disc_fwd': f'{disc_forward_time*1000:.1f}ms',
                    'disc_back': f'{disc_backward_time*1000:.1f}ms',
                    's_refwd': f'{student_re_forward_time*1000:.1f}ms',
                    'loss': f'{loss_compute_time*1000:.1f}ms',
                    'g_back': f'{g_backward_time*1000:.1f}ms',
                    'total': f'{total_step_time*1000:.1f}ms'
                }
            else:
                postfix_dict = {
                    'G': g_loss.item(),
                    'D': d_loss.item(),
                    'MAE': mae_loss.item(),
                    'KD': kd_loss.item()
                }
            
            pbar.set_postfix(postfix_dict)
        
        n = len(train_loader)
        return {
            'g_loss': total_g_loss / n,
            'd_loss': total_d_loss / n,
            'gan_loss': total_gan_loss / n,
            'mae_loss': total_mae_loss / n,
            'kd_loss': total_kd_loss / n,
            'recon_loss': total_recon_loss / n
        }
    
    def validate(self, val_loader, epoch):
        self.student.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                patch_3t = batch['patch_3t'].to(self.device)
                patch_7t = batch['patch_7t'].to(self.device)
                
                patch_3t = patch_3t.squeeze(0)
                patch_7t = patch_7t.squeeze(0)      
                
                fake_7t, uncertainty, _ = self.student(patch_3t)
                
                mae = self.mae_loss(fake_7t, patch_7t)
                
                total_loss += mae.item()
                total_mae += mae.item()
        
        n = len(val_loader)
        avg_loss = total_loss / n
        avg_mae = total_mae / n
        
        self.writer.add_scalar('Val/loss', avg_loss, epoch)
        self.writer.add_scalar('Val/mae', avg_mae, epoch)
        
        return avg_loss, avg_mae
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'student_state_dict': self.student.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'amalgamation_state_dict': self.amalgamation.state_dict(),
            'adaptive_weights_state_dict': self.adaptive_weights.state_dict(),
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
        
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.amalgamation.load_state_dict(checkpoint['amalgamation_state_dict'])
        self.adaptive_weights.load_state_dict(checkpoint['adaptive_weights_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader):
        print("Training Student Model (AU-MIPGAN)")
        print(f"Device: {self.device}")
        print(f"Number of GPUs: {self.config.num_gpus}")
        if self.config.gpu_ids:
            print(f"GPU IDs: {self.config.gpu_ids}")
        
        for epoch in range(self.config.num_epochs):
            losses = self.train_epoch(train_loader, epoch)
            
            self.schedulers['g'].step()
            self.schedulers['d'].step()
            
            current_lr = self.g_optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train - G_loss: {losses['g_loss']:.4f}, D_loss: {losses['d_loss']:.4f}")
            print(f"  Train - GAN: {losses['gan_loss']:.4f}, MAE: {losses['mae_loss']:.4f}")
            print(f"  Train - KD: {losses['kd_loss']:.4f}, Recon: {losses['recon_loss']:.4f}")
            print(f"  Learning rate: {current_lr:.6f}")
            
            is_best = False
            if (epoch + 1) % self.config.val_interval == 0:
                avg_val_loss, avg_val_mae = self.validate(val_loader, epoch)
                print(f"  Val - Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}")
                
                is_best = avg_val_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_val_loss
                    print(f"  New best model! Loss: {self.best_loss:.4f}")
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, is_best=is_best)
                print(f"  Checkpoint saved at epoch {epoch+1}")
        
        self.save_checkpoint(self.config.num_epochs - 1, is_best=False)
        print("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Student Model for AU-MIPGAN')
    parser.add_argument('--teacher_axial', type=str, default=None,
                        help='Path to axial teacher checkpoint')
    parser.add_argument('--teacher_coronal', type=str, default=None,
                        help='Path to coronal teacher checkpoint')
    parser.add_argument('--teacher_sagittal', type=str, default=None,
                        help='Path to sagittal teacher checkpoint')
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
    
    teacher_checkpoints = [
        args.teacher_axial,
        args.teacher_coronal,
        args.teacher_sagittal
    ]
    
    if all(ckp is None for ckp in teacher_checkpoints):
        teacher_checkpoints = [
            os.path.join(config.checkpoint_dir, 'teacher_axial', 'best_model.pth'),
            os.path.join(config.checkpoint_dir, 'teacher_coronal', 'best_model.pth'),
            os.path.join(config.checkpoint_dir, 'teacher_sagittal', 'best_model.pth')
        ]
    
    print(f"Configuration:")
    print(config)
    print(f"\nTeacher checkpoints:")
    for i, ckp in enumerate(teacher_checkpoints):
        print(f"  {['Axial', 'Coronal', 'Sagittal'][i]}: {ckp}")
    
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(config.project_root, 'preprocessed')
    
    train_data_dirs = {
        'volume': os.path.join(data_dir, 'train', 'volume'),
        'axial': os.path.join(data_dir, 'train', 'axial'),
        'coronal': os.path.join(data_dir, 'train', 'coronal'),
        'sagittal': os.path.join(data_dir, 'train', 'sagittal'),
    }
    
    val_data_dirs = {
        'volume': os.path.join(data_dir, 'val', 'volume'),
        'axial': os.path.join(data_dir, 'val', 'axial'),
        'coronal': os.path.join(data_dir, 'val', 'coronal'),
        'sagittal': os.path.join(data_dir, 'val', 'sagittal'),
    }
    
    num_train_samples = len(glob.glob(os.path.join(train_data_dirs['volume'], '*_3t.nii.gz')))
    num_val_samples = len(glob.glob(os.path.join(val_data_dirs['volume'], '*_3t.nii.gz')))
    
    print(f"\nLoading data from: {data_dir}")
    print(f"  Train samples: {num_train_samples}")
    print(f"  Val samples: {num_val_samples}")
    
    train_dataset = MultiDirDataset(
        data_dirs=train_data_dirs,
        num_samples=num_train_samples,
        patch_size=config.patch_size,
        mode='train',
        batch_size=config.batch_size,
        batch_per_sample=config.batch_per_sample
    )
    
    val_dataset = MultiDirDataset(
        data_dirs=val_data_dirs,
        num_samples=num_val_samples,
        patch_size=config.patch_size,
        mode='val',
        batch_size=config.batch_size,
        batch_per_sample=config.batch_per_sample
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=min(config.num_workers*4, os.cpu_count()),
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(config.num_workers*4, os.cpu_count()),
        pin_memory=config.pin_memory
    )
    
    trainer = StudentTrainer(config, teacher_checkpoints)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
