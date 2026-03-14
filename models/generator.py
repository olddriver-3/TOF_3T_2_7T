import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, use_relu=True):
        super(ConvBlock3D, self).__init__()
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm3d(out_channels))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)


class UpConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
        super(UpConvBlock3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv = ConvBlock3D(out_channels, out_channels, kernel_size, stride, padding, use_bn)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(EncoderBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(ConvBlock3D(in_channels if i == 0 else out_channels, out_channels))
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.convs(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None, num_convs=2):
        super(DecoderBlock, self).__init__()
        self.up = UpConvBlock3D(in_channels, out_channels)
        
        conv_in_channels = out_channels
        if skip_channels is not None:
            conv_in_channels = out_channels + skip_channels
        
        layers = []
        for i in range(num_convs):
            layers.append(ConvBlock3D(conv_in_channels if i == 0 else out_channels, out_channels))
        self.convs = nn.Sequential(*layers)
    
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.convs(x)
        return x


class UNet3DGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=32, num_layers=5, 
                 output_uncertainty=False, return_features=False):
        super(UNet3DGenerator, self).__init__()
        self.num_layers = num_layers
        self.output_uncertainty = output_uncertainty
        self.return_features = return_features
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        in_ch = in_channels
        out_ch = base_features
        self.encoder_channels = []
        
        for i in range(num_layers):
            self.encoders.append(EncoderBlock(in_ch, out_ch))
            self.encoder_channels.append(out_ch)
            in_ch = out_ch
            out_ch = out_ch * 2 if i < num_layers - 1 else out_ch
        
        self.bottleneck = nn.Sequential(
            ConvBlock3D(self.encoder_channels[-1], self.encoder_channels[-1] * 2),
            ConvBlock3D(self.encoder_channels[-1] * 2, self.encoder_channels[-1] * 2)
        )
        bottleneck_channels = self.encoder_channels[-1] * 2
        
        decoder_in_ch = bottleneck_channels
        for i in range(num_layers - 1, -1, -1):
            skip_ch = self.encoder_channels[i]
            out_ch = self.encoder_channels[i] if i > 0 else base_features
            self.decoders.append(DecoderBlock(decoder_in_ch, out_ch, skip_ch))
            decoder_in_ch = out_ch
        
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)
        
        if output_uncertainty:
            self.uncertainty_conv = nn.Conv3d(base_features, 1, kernel_size=1)
    
    def forward(self, x):
        features = []
        skips = []
        
        for encoder in self.encoders:
            x, skip = encoder(x)
            features.append(x)
            skips.append(skip)
        
        x = self.bottleneck(x)
        features.append(x)
        
        skips = skips[::-1]
        
        for i, decoder in enumerate(self.decoders):
            skip = skips[i] if i < len(skips) else None
            x = decoder(x, skip)
        
        output = self.final_conv(x)
        
        if self.output_uncertainty:
            uncertainty = self.uncertainty_conv(x)
            uncertainty = F.softplus(uncertainty) + 1e-6
            if self.return_features:
                return output, uncertainty, features
            return output, uncertainty
        
        if self.return_features:
            return output, features
        return output


class UNet3DGeneratorWithFeatures(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=32, num_layers=5):
        super(UNet3DGeneratorWithFeatures, self).__init__()
        self.num_layers = num_layers
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        in_ch = in_channels
        out_ch = base_features
        self.encoder_channels = []
        
        for i in range(num_layers):
            self.encoders.append(EncoderBlock(in_ch, out_ch))
            self.encoder_channels.append(out_ch)
            in_ch = out_ch
            out_ch = out_ch * 2 if i < num_layers - 1 else out_ch
        
        self.bottleneck = nn.Sequential(
            ConvBlock3D(self.encoder_channels[-1], self.encoder_channels[-1] * 2),
            ConvBlock3D(self.encoder_channels[-1] * 2, self.encoder_channels[-1] * 2)
        )
        bottleneck_channels = self.encoder_channels[-1] * 2
        
        decoder_in_ch = bottleneck_channels
        for i in range(num_layers - 1, -1, -1):
            skip_ch = self.encoder_channels[i]
            out_ch = self.encoder_channels[i] if i > 0 else base_features
            self.decoders.append(DecoderBlock(decoder_in_ch, out_ch, skip_ch))
            decoder_in_ch = out_ch
        
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        encoder_features = []
        skips = []
        
        for encoder in self.encoders:
            x, skip = encoder(x)
            encoder_features.append(x)
            skips.append(skip)
        
        x = self.bottleneck(x)
        encoder_features.append(x)
        
        decoder_features = []
        skips = skips[::-1]
        
        for i, decoder in enumerate(self.decoders):
            skip = skips[i] if i < len(skips) else None
            x = decoder(x, skip)
            decoder_features.append(x)
        
        output = self.final_conv(x)
        
        all_features = encoder_features + decoder_features
        
        return output, all_features


if __name__ == '__main__':
    model = UNet3DGenerator(in_channels=1, out_channels=1, base_features=32, num_layers=5, output_uncertainty=True)
    x = torch.randn(2, 1, 64, 64, 64)
    output, uncertainty = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
