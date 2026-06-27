import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    """Standard ResNet Residual Block with Conv, BatchNorm, and PReLU."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.prelu(out)
        out = self.bn2(self.conv2(out))
        return out + residual

class SRResNet(nn.Module):
    """Deep residual network for 4x geospatial super-resolution."""
    def __init__(self, in_channels: int = 2, scale: int = 4, num_res_blocks: int = 8):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Upsampling block using PixelShuffle (sub-pixel convolution)
        upsample_blocks = []
        num_upsamples = int(np.log2(scale)) if scale > 1 else 0
        for _ in range(num_upsamples):
            upsample_blocks.append(nn.Conv2d(64, 256, kernel_size=3, padding=1))
            upsample_blocks.append(nn.PixelShuffle(upscale_factor=2))
            upsample_blocks.append(nn.PReLU())
            
        self.upsample = nn.Sequential(*upsample_blocks)
        self.conv_out = nn.Conv2d(64, in_channels, kernel_size=9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.prelu(self.conv1(x))
        out = self.res_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out + out1
        out = self.upsample(out)
        return self.conv_out(out)
        
    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs forward pass and extracts features for cross-attention/skip conditioning."""
        out1 = self.prelu(self.conv1(x))
        features = self.res_blocks(out1)
        out = self.bn2(self.conv2(features))
        out = out + out1
        out = self.upsample(out)
        hr_ir = self.conv_out(out)
        return hr_ir, features

class UNetColorizer(nn.Module):
    """Deep U-Net for translating co-registered IR/Thermal bands to natural RGB."""
    def __init__(self, in_channels: int = 2, out_channels: int = 3):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2) # 256x256
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2) # 128x128
        
        # Bridge features concatenated at 128x128 level: 64 (enc2) + 64 (bridge) = 128 channels
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2) # 64x64
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2) # 32x32
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2) # 64x64
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2) # 128x128
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) # 256x256
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2) # 512x512
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_conditioned(x, None)
        
    def forward_conditioned(self, x: torch.Tensor, bridge_features: torch.Tensor | None = None) -> torch.Tensor:
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        
        # Concat bridge features if present at the 128x128 bottleneck inlet
        if bridge_features is not None:
            if bridge_features.shape[-2:] != p2.shape[-2:]:
                bridge_features = nn.functional.interpolate(bridge_features, size=p2.shape[-2:], mode="bilinear", align_corners=False)
            p2_cond = torch.cat([p2, bridge_features], dim=1)
        else:
            zeros = torch.zeros(p2.size(0), 64, p2.size(2), p2.size(3), device=p2.device, dtype=p2.dtype)
            p2_cond = torch.cat([p2, zeros], dim=1)
            
        x3 = self.enc3(p2_cond)
        p3 = self.pool3(x3)
        
        x4 = self.enc4(p3)
        p4 = self.pool4(x4)
        
        b = self.bottleneck(p4)
        
        d4 = self.up4(b)
        if d4.shape[-2:] != x4.shape[-2:]:
            d4 = nn.functional.interpolate(d4, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, x4], dim=1)
        x_dec4 = self.dec4(d4)
        
        d3 = self.up3(x_dec4)
        if d3.shape[-2:] != x3.shape[-2:]:
            d3 = nn.functional.interpolate(d3, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, x3], dim=1)
        x_dec3 = self.dec3(d3)
        
        d2 = self.up2(x_dec3)
        if d2.shape[-2:] != x2.shape[-2:]:
            d2 = nn.functional.interpolate(d2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, x2], dim=1)
        x_dec2 = self.dec2(d2)
        
        d1 = self.up1(x_dec2)
        if d1.shape[-2:] != x1.shape[-2:]:
            d1 = nn.functional.interpolate(d1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, x1], dim=1)
        x_dec1 = self.dec1(d1)
        
        return self.final_conv(x_dec1)
