import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.loss(prediction, target)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
    
    def forward(self, discriminator, real_data, fake_data):
        real_pred = discriminator(real_data)
        fake_pred = discriminator(fake_data)
        
        loss_real = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )
        
        return loss_real + loss_fake


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
    
    def forward(self, prediction, target):
        return torch.mean(torch.abs(prediction - target))


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
    
    def forward(self, reconstructed_features, original_features, phi=None):
        """
        Compute reconstruction loss with adaptive weighting (Equation 7).
        
        L_R = sum_{m,l} [1/(2*phi_{m,l}^2) * L_R^{(m)(l)} + log(phi_{m,l})]
        
        Args:
            reconstructed_features: List of reconstructed features for each direction
            original_features: List of original features for each direction
            phi: Adaptive weights of shape (num_directions, num_layers), directly used as phi
        
        Returns:
            Total reconstruction loss
        """
        num_directions = len(reconstructed_features)
        num_layers = len(reconstructed_features[0])
        
        total_loss = 0.0
        
        for m in range(num_directions):
            for l in range(num_layers):
                recon = reconstructed_features[m][l]
                orig = original_features[m][l]
                
                loss = F.mse_loss(recon, orig, reduction='mean')
                
                if phi is not None:
                    phi_m_l = phi[m, l]
                    total_loss = total_loss + 1.0 / (2.0 * phi_m_l ** 2) * loss + torch.log(phi_m_l)
                else:
                    total_loss = total_loss + loss
        
        return total_loss


class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
    
    def forward(self, student_features, amalgamated_features, omega=None):
        """
        Compute knowledge distillation loss (Equation 9).
        
        L_KD = sum_{l=1}^{L} [1/(2*omega_l^2) * L_KD^{(l)} + log(omega_l)]
        
        Args:
            student_features: List of student features at each layer
            amalgamated_features: List of amalgamated features at each layer
            omega: Adaptive weights of shape (num_layers,), directly used as omega
        
        Returns:
            Total KD loss
        """
        num_layers = len(student_features)
        
        total_loss = 0.0
        
        for l in range(num_layers):
            student_feat = student_features[l]
            amalgamated_feat = amalgamated_features[l]
            
            loss = F.mse_loss(student_feat, amalgamated_feat, reduction='mean')
            
            if omega is not None:
                omega_l = omega[l]
                total_loss = total_loss + 1.0 / (2.0 * omega_l ** 2) * loss + torch.log(omega_l)
            else:
                total_loss = total_loss + loss
        
        return total_loss


class AUAELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(AUAELoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, prediction, target, uncertainty):
        """
        Compute Aleatoric Uncertainty-weighted Absolute Error loss.
        
        Args:
            prediction: Predicted 7T-like image
            target: Ground truth 7T image
            uncertainty: Predicted uncertainty map
        
        Returns:
            AU-AE loss
        """
        abs_error = torch.abs(prediction - target)
        
        weighted_error = abs_error / (uncertainty + self.epsilon)
        
        log_uncertainty = torch.log(uncertainty + self.epsilon)
        
        loss = torch.mean(weighted_error + log_uncertainty)
        
        return loss


class TeacherLoss(nn.Module):
    def __init__(self, alpha=10.0):
        super(TeacherLoss, self).__init__()
        self.alpha = alpha
        self.gan_loss = GANLoss()
        self.mae_loss = MAELoss()
    
    def forward(self, discriminator, generator, real_7t, fake_7t):
        gan_loss = self.gan_loss(discriminator(fake_7t), True)
        mae_loss = self.mae_loss(fake_7t, real_7t)
        
        total_loss = gan_loss + self.alpha * mae_loss
        
        return total_loss, gan_loss, mae_loss


class StudentLoss(nn.Module):
    def __init__(self, alpha=10.0, beta=1.0, gamma=1.0):
        super(StudentLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.gan_loss = GANLoss()
        self.mae_loss = MAELoss()
        self.recon_loss = ReconstructionLoss()
        self.kd_loss = KDLoss()
        self.au_ae_loss = AUAELoss()
    
    def forward(self, discriminator, generator, real_7t, fake_7t, uncertainty,
                student_features, amalgamated_features, reconstructed_features, 
                teacher_features, recon_weights=None, kd_weights=None,
                use_au=True):
        gan_loss = self.gan_loss(discriminator(fake_7t), True)
        
        if use_au and uncertainty is not None:
            mae_loss = self.au_ae_loss(fake_7t, real_7t, uncertainty)
        else:
            mae_loss = self.mae_loss(fake_7t, real_7t)
        
        recon_loss = self.recon_loss(reconstructed_features, teacher_features, recon_weights)
        
        kd_loss = self.kd_loss(student_features, amalgamated_features, kd_weights)
        
        total_loss = gan_loss + self.alpha * mae_loss + self.beta * recon_loss + self.gamma * kd_loss
        
        return total_loss, gan_loss, mae_loss, recon_loss, kd_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.gan_loss = GANLoss()
    
    def forward(self, discriminator, real_data, fake_data):
        real_loss = self.gan_loss(discriminator(real_data), True)
        fake_loss = self.gan_loss(discriminator(fake_data.detach()), False)
        
        total_loss = (real_loss + fake_loss) * 0.5
        
        return total_loss, real_loss, fake_loss


if __name__ == '__main__':
    batch_size = 2
    channels = 1
    depth, height, width = 64, 64, 64
    
    real_7t = torch.randn(batch_size, channels, depth, height, width)
    fake_7t = torch.randn(batch_size, channels, depth, height, width)
    uncertainty = torch.abs(torch.randn(batch_size, 1, depth, height, width)) + 0.1
    
    gan_loss = GANLoss()
    mae_loss = MAELoss()
    au_ae_loss = AUAELoss()
    
    print(f"MAE Loss: {mae_loss(fake_7t, real_7t).item():.4f}")
    print(f"AU-AE Loss: {au_ae_loss(fake_7t, real_7t, uncertainty).item():.4f}")
    
    num_layers = 5
    feature_channels = [32, 64, 128, 256, 512]
    spatial_size = [64, 32, 16, 8, 4]
    
    student_features = [torch.randn(batch_size, c, s, s, s) for c, s in zip(feature_channels, spatial_size)]
    amalgamated_features = [torch.randn(batch_size, c, s, s, s) for c, s in zip(feature_channels, spatial_size)]
    
    kd_loss = KDLoss()
    print(f"\nKD Loss (no weights): {kd_loss(student_features, amalgamated_features).item():.4f}")
    
    omega = torch.ones(num_layers)
    print(f"KD Loss (with omega=1): {kd_loss(student_features, amalgamated_features, omega).item():.4f}")
    
    num_directions = 3
    reconstructed_features = [[torch.randn(batch_size, c, s, s, s) for c, s in zip(feature_channels, spatial_size)] 
                              for _ in range(num_directions)]
    teacher_features = [[torch.randn(batch_size, c, s, s, s) for c, s in zip(feature_channels, spatial_size)] 
                        for _ in range(num_directions)]
    
    recon_loss = ReconstructionLoss()
    print(f"\nReconstruction Loss (no weights): {recon_loss(reconstructed_features, teacher_features).item():.4f}")
    
    phi = torch.ones(num_directions, num_layers)
    print(f"Reconstruction Loss (with phi=1): {recon_loss(reconstructed_features, teacher_features, phi).item():.4f}")
