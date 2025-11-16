import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard 2D convolutional block (Conv2D -> BatchNorm -> ReLU).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    """
    Down-sampling block (MaxPool -> ConvBlock).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x_pooled = self.pool(x)
        x_conv = self.conv(x_pooled)
        return x_conv, x_pooled # Return pooled for skip (this is slightly different, just pass x_conv)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x # Return convoluted block, we'll store skip in main model

class UpBlock(nn.Module):
    """
    Up-sampling block (ConvTranspose -> Concat -> ConvBlock).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels is for the skip connection + upsampled feature map

        # --- BUG ---
        # self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # --- FIX ---
        # The input tensor (x_up) has 2/3 of the total combined channels (e.g., 1024 in a 1536 total).
        # The output of this layer should also have 2/3 of the channels, to be concatenated
        # with the skip connection (which has 1/3 of the channels).
        up_channels = (in_channels * 2) // 3
        
        # Check if in_channels is divisible by 3, otherwise something is wrong
        if (in_channels * 2) % 3 != 0:
            raise ValueError(f"UpBlock in_channels ({in_channels}) is not divisible by 3. Check U-Net architecture.")

        self.up = nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)
        
        # Handle potential size mismatch from pooling
        if x_up.shape != x_skip.shape:
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """
    The main 2D U-Net model.
    """
    def __init__(self, in_channels, out_channels, n_filters=64):
        super().__init__()
        
        # Encoder
        self.in_conv = ConvBlock(in_channels, n_filters)
        self.down1 = DownBlock(n_filters, n_filters * 2)
        self.down2 = DownBlock(n_filters * 2, n_filters * 4)
        self.down3 = DownBlock(n_filters * 4, n_filters * 8)
        
        # Bottleneck
        self.bottle = DownBlock(n_filters * 8, n_filters * 16)

        # Decoder
        self.up3 = UpBlock(n_filters * 16 + n_filters * 8, n_filters * 8)
        self.up2 = UpBlock(n_filters * 8 + n_filters * 4, n_filters * 4)
        self.up1 = UpBlock(n_filters * 4 + n_filters * 2, n_filters * 2)
        self.out_conv = UpBlock(n_filters * 2 + n_filters, n_filters)

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.in_conv(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        
        # Bottleneck
        b = self.bottle(s4)

        # Decoder
        x = self.up3(b, s4)
        x = self.up2(x, s3)
        x = self.up1(x, s2)
        x = self.out_conv(x, s1)
        
        return self.final_conv(x)