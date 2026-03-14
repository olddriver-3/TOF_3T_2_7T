import torch
import torch.nn as nn
import torch.nn.functional as F


class AmalgamationModule(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super(AmalgamationModule, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = in_channels // 2
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(in_channels)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class MultiDirectionAmalgamation(nn.Module):
    def __init__(self, feature_channels_list, num_directions=3):
        super(MultiDirectionAmalgamation, self).__init__()
        
        self.num_directions = num_directions
        self.num_layers = len(feature_channels_list)
        
        self.amalgamation_modules = nn.ModuleList()
        for channels in feature_channels_list:
            self.amalgamation_modules.append(
                AmalgamationModule(channels * num_directions, channels)
            )
        
        self.phi = nn.Parameter(torch.ones(num_directions, self.num_layers))
    
    def forward(self, teacher_features_list):
        """
        Args:
            teacher_features_list: list of lists
                teacher_features_list[m][l] is the feature map from teacher m at layer l
                m = 0, 1, 2 for axial, coronal, sagittal
                l = 0, 1, ..., L-1 for each layer
        
        Returns:
            amalgamated_features: list of amalgamated features at each layer
            reconstructed_features: list of lists of reconstructed features
        """
        batch_size = teacher_features_list[0][0].size(0)
        
        amalgamated_features = []
        reconstructed_features = [[[] for _ in range(self.num_layers)] for _ in range(self.num_directions)]
        
        for l in range(self.num_layers):
            concat_features = torch.cat([teacher_features_list[m][l] for m in range(self.num_directions)], dim=1)
            
            reconstructed, encoded = self.amalgamation_modules[l](concat_features)
            
            amalgamated_features.append(encoded)
            
            channels = teacher_features_list[0][l].size(1)
            for m in range(self.num_directions):
                start_idx = m * channels
                end_idx = (m + 1) * channels
                reconstructed_features[m][l] = reconstructed[:, start_idx:end_idx, ...]
        
        return amalgamated_features, reconstructed_features
    
    def get_adaptive_weights(self):
        weights = F.softmax(self.phi, dim=0)
        return weights


class AdaptiveWeightModule(nn.Module):
    def __init__(self, num_directions=3, num_layers=5):
        super(AdaptiveWeightModule, self).__init__()
        
        self.num_directions = num_directions
        self.num_layers = num_layers
        
        self.phi = nn.Parameter(torch.ones(num_directions, num_layers))
        
        self.omega = nn.Parameter(torch.ones(num_layers))
    
    def get_reconstruction_weights(self):
        return self.phi
    
    def get_kd_weights(self):
        return self.omega
    
    def forward(self):
        return self.phi, self.omega


if __name__ == '__main__':
    batch_size = 2
    num_directions = 3
    num_layers = 5
    feature_channels = [32, 64, 128, 256, 512]
    spatial_size = [64, 32, 16, 8, 4]
    
    teacher_features = []
    for m in range(num_directions):
        direction_features = []
        for l in range(num_layers):
            feat = torch.randn(batch_size, feature_channels[l], spatial_size[l], spatial_size[l], spatial_size[l])
            direction_features.append(feat)
        teacher_features.append(direction_features)
    
    amalgamation = MultiDirectionAmalgamation(feature_channels, num_directions)
    
    amalgamated, reconstructed = amalgamation(teacher_features)
    
    print("Amalgamated features:")
    for i, feat in enumerate(amalgamated):
        print(f"  Layer {i}: {feat.shape}")
    
    print("\nReconstructed features:")
    for m in range(num_directions):
        print(f"  Direction {m}:")
        for l in range(num_layers):
            print(f"    Layer {l}: {reconstructed[m][l].shape}")
    
    adaptive_weight = AdaptiveWeightModule(num_directions, num_layers)
    recon_weights, kd_weights = adaptive_weight()
    print(f"\nReconstruction weights shape: {recon_weights.shape}")
    print(f"KD weights shape: {kd_weights.shape}")
