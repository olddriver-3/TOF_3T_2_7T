import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bn=True, use_leaky=True):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm3d(out_channels))
        if use_leaky:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)


class Discriminator3D(nn.Module):
    def __init__(self, in_channels=1, base_features=32, num_layers=4):
        super(Discriminator3D, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Sequential(
            nn.Conv3d(in_channels, base_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        in_ch = base_features
        for i in range(1, num_layers):
            out_ch = min(base_features * (2 ** i), 512)
            self.layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        
        self.final_conv = nn.Conv3d(in_ch, 1, kernel_size=4, stride=1, padding=0)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return x


class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_channels=1, base_features=32, num_layers=4):
        super(PatchDiscriminator3D, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Sequential(
            nn.Conv3d(in_channels, base_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        in_ch = base_features
        for i in range(1, num_layers):
            out_ch = min(base_features * (2 ** i), 512)
            self.layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        
        self.final_conv = nn.Conv3d(in_ch, 1, kernel_size=4, stride=1, padding=1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_conv(x)
        return x


class MultiScaleDiscriminator3D(nn.Module):
    def __init__(self, in_channels=1, base_features=32, num_scales=3, num_layers=4):
        super(MultiScaleDiscriminator3D, self).__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for i in range(num_scales):
            self.discriminators.append(PatchDiscriminator3D(in_channels, base_features, num_layers))
        
        self.downsample = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < self.num_scales - 1:
                x = self.downsample(x)
        return outputs


if __name__ == '__main__':
    x = torch.randn(2, 1, 64, 64, 64)
    
    print("=" * 60)
    print("Discriminator3D (num_layers=4)")
    print("=" * 60)
    disc = Discriminator3D(in_channels=1, base_features=32, num_layers=4)
    output = disc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    num_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    print("\n" + "=" * 60)
    print("PatchDiscriminator3D (num_layers=4)")
    print("=" * 60)
    patch_disc = PatchDiscriminator3D(in_channels=1, base_features=32, num_layers=4)
    output = patch_disc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    num_params = sum(p.numel() for p in patch_disc.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    print("\n" + "=" * 60)
    print("MultiScaleDiscriminator3D (num_scales=3, num_layers=3)")
    print("=" * 60)
    multi_disc = MultiScaleDiscriminator3D(in_channels=1, base_features=32, num_scales=3, num_layers=3)
    outputs = multi_disc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shapes:")
    for i, out in enumerate(outputs):
        print(f"  Scale {i+1}: {out.shape}")
    num_params = sum(p.numel() for p in multi_disc.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
