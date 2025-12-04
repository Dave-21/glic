import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset


class ConvBlock(nn.Module):
    """
    Standard 2D convolutional block with Dropout for Uncertainty Quantification.
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, up_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)

        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(
                x_up, size=x_skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class PhysicsGate(nn.Module):
    """
    Gating mechanism that scales bottleneck features based on global physical context (CFDD).
    Acts as a 'thermostat' to adjust model behavior based on the season.
    """
    def __init__(self, bottleneck_channels):
        super().__init__()
        # Input to MLP is scalar (mean CFDD)
        self.mlp = nn.Sequential(
            nn.Linear(1, bottleneck_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels // 2, bottleneck_channels),
            nn.Sigmoid()
        )

    def forward(self, x_input, x_bottleneck):
        # x_input: (B, C_in, H, W). Channel 1 is CFDD.
        # x_bottleneck: (B, C_bottle, H_b, W_b)
        
        # Extract CFDD (Channel 1)
        # Note: We assume channel 1 is CFDD based on dataset.py
        cfdd = x_input[:, 1:2, :, :] # (B, 1, H, W)
        
        # Global Average Pooling of CFDD
        cfdd_avg = F.adaptive_avg_pool2d(cfdd, 1).flatten(1) # (B, 1)
        
        # Compute Gate
        scale = self.mlp(cfdd_avg) # (B, C_bottle)
        
        # Reshape for broadcasting: (B, C_bottle, 1, 1)
        scale = scale.unsqueeze(2).unsqueeze(3)
        
        return x_bottleneck * scale


class UNet(nn.Module):
    """
    Configurable U-Net with variable depth, Attention, and Physics Gating.
    """

    def __init__(
        self,
        in_channels=dataset.N_INPUT_CHANNELS,
        out_channels=dataset.N_OUTPUT_CHANNELS,
        n_filters=64,
        dropout_rate=0.2,
        depth=4,
        use_attention=False,
        use_physics_gate=False,
        num_classes=1,
    ):
        super().__init__()
        self.depth = depth
        self.use_attention = use_attention
        self.use_physics_gate = use_physics_gate
        self.num_classes = num_classes


class ConvBlock(nn.Module):
    """
    Standard 2D convolutional block with Dropout for Uncertainty Quantification.
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, up_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)

        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(
                x_up, size=x_skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class PhysicsGate(nn.Module):
    """
    Gating mechanism that scales bottleneck features based on global physical context (CFDD).
    Acts as a 'thermostat' to adjust model behavior based on the season.
    """
    def __init__(self, bottleneck_channels):
        super().__init__()
        # Input to MLP is scalar (mean CFDD)
        self.mlp = nn.Sequential(
            nn.Linear(1, bottleneck_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels // 2, bottleneck_channels),
            nn.Sigmoid()
        )

    def forward(self, x_input, x_bottleneck):
        # x_input: (B, C_in, H, W). Channel 1 is CFDD.
        # x_bottleneck: (B, C_bottle, H_b, W_b)
        
        # Extract CFDD (Channel 1)
        # Note: We assume channel 1 is CFDD based on dataset.py
        cfdd = x_input[:, 1:2, :, :] # (B, 1, H, W)
        
        # Global Average Pooling of CFDD
        cfdd_avg = F.adaptive_avg_pool2d(cfdd, 1).flatten(1) # (B, 1)
        
        # Compute Gate
        scale = self.mlp(cfdd_avg) # (B, C_bottle)
        
        # Reshape for broadcasting: (B, C_bottle, 1, 1)
        scale = scale.unsqueeze(2).unsqueeze(3)
        
        return x_bottleneck * scale


class UNet(nn.Module):
    """
    Configurable U-Net with variable depth, Attention, and Physics Gating.
    """

    def __init__(
        self,
        in_channels=dataset.N_INPUT_CHANNELS,
        out_channels=dataset.N_OUTPUT_CHANNELS,
        n_filters=64,
        dropout_rate=0.2,
        depth=4,
        use_attention=False,
        use_physics_gate=False,
        num_classes=1, # 1 for regression, >1 for classification
    ):
        super().__init__()
        self.depth = depth
        self.use_attention = use_attention
        self.use_physics_gate = use_physics_gate
        self.num_classes = num_classes
        self.out_channels = out_channels

        # --- Encoders ---
        self.enc1 = ConvBlock(in_channels, n_filters, dropout_rate)
        self.enc2 = DownBlock(n_filters, n_filters * 2, dropout_rate)
        self.enc3 = DownBlock(n_filters * 2, n_filters * 4, dropout_rate)
        
        if depth >= 4:
            self.enc4 = DownBlock(n_filters * 4, n_filters * 8, dropout_rate)
        if depth >= 5:
            self.enc5 = DownBlock(n_filters * 8, n_filters * 16, dropout_rate)

        # --- Center (Bottleneck) ---
        if depth == 3:
            self.center = DownBlock(n_filters * 4, n_filters * 8, dropout_rate)
            center_channels = n_filters * 8
        elif depth == 4:
            self.center = DownBlock(n_filters * 8, n_filters * 16, dropout_rate)
            center_channels = n_filters * 16
        elif depth == 5:
            self.center = DownBlock(n_filters * 16, n_filters * 32, dropout_rate)
            center_channels = n_filters * 32
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        # --- Physics Gate ---
        if use_physics_gate:
            self.physics_gate = PhysicsGate(center_channels)

        # --- Decoders & Attention ---
        
        if depth >= 5:
            # dec5: up(center) + enc5
            self.dec5 = UpBlock(n_filters * 32 + n_filters * 16, n_filters * 32, n_filters * 16, dropout_rate)
            if use_attention:
                self.att5 = AttentionBlock(F_g=n_filters*32, F_l=n_filters*16, F_int=n_filters*16)

        if depth >= 4:
            # dec4: up(dec5 or center) + enc4
            # input to dec4 is (nf*16) from below. enc4 is (nf*8).
            self.dec4 = UpBlock(n_filters * 16 + n_filters * 8, n_filters * 16, n_filters * 8, dropout_rate)
            if use_attention:
                self.att4 = AttentionBlock(F_g=n_filters*16, F_l=n_filters*8, F_int=n_filters*8)

        # dec3: up(dec4 or center) + enc3
        # input is (nf*8). enc3 is (nf*4).
        self.dec3 = UpBlock(n_filters * 8 + n_filters * 4, n_filters * 8, n_filters * 4, dropout_rate)
        if use_attention:
            self.att3 = AttentionBlock(F_g=n_filters*8, F_l=n_filters*4, F_int=n_filters*4)

        # dec2: up(dec3) + enc2
        # input is (nf*4). enc2 is (nf*2).
        self.dec2 = UpBlock(n_filters * 4 + n_filters * 2, n_filters * 4, n_filters * 2, dropout_rate)
        if use_attention:
            self.att2 = AttentionBlock(F_g=n_filters*4, F_l=n_filters*2, F_int=n_filters*2)

        # dec1: up(dec2) + enc1
        # input is (nf*2). enc1 is (nf).
        self.dec1 = UpBlock(n_filters * 2 + n_filters, n_filters * 2, n_filters, dropout_rate)
        if use_attention:
            self.att1 = AttentionBlock(F_g=n_filters*2, F_l=n_filters, F_int=n_filters//2)

        # --- Output Heads ---
        self.out_conc = nn.Conv2d(n_filters, out_channels, kernel_size=1)
        
        # Thickness head: If classification, output (T * Classes) channels
        if num_classes > 1:
            self.out_thick = nn.Conv2d(n_filters, out_channels * num_classes, kernel_size=1)
        else:
            self.out_thick = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoders
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        if self.depth == 3:
            c = self.center(e3)
            if self.use_physics_gate:
                c = self.physics_gate(x, c)
            
            # Decoder 3
            up_c = self.dec3.up(c)
            if self.use_attention:
                if up_c.shape[2:] != e3.shape[2:]:
                    up_c = F.interpolate(up_c, size=e3.shape[2:], mode="bilinear", align_corners=True)
                e3 = self.att3(g=up_c, x=e3)
            d3 = self.dec3(c, e3)
            
        elif self.depth == 4:
            e4 = self.enc4(e3)
            c = self.center(e4)
            if self.use_physics_gate:
                c = self.physics_gate(x, c)
                
            # Decoder 4
            up_c = self.dec4.up(c)
            if self.use_attention:
                if up_c.shape[2:] != e4.shape[2:]:
                    up_c = F.interpolate(up_c, size=e4.shape[2:], mode="bilinear", align_corners=True)
                e4 = self.att4(g=up_c, x=e4)
            d4 = self.dec4(c, e4)
            
            # Decoder 3
            up_d4 = self.dec3.up(d4)
            if self.use_attention:
                if up_d4.shape[2:] != e3.shape[2:]:
                    up_d4 = F.interpolate(up_d4, size=e3.shape[2:], mode="bilinear", align_corners=True)
                e3 = self.att3(g=up_d4, x=e3)
            d3 = self.dec3(d4, e3)

        elif self.depth == 5:
            e4 = self.enc4(e3)
            e5 = self.enc5(e4)
            c = self.center(e5)
            if self.use_physics_gate:
                c = self.physics_gate(x, c)
                
            # Decoder 5
            up_c = self.dec5.up(c)
            if self.use_attention:
                if up_c.shape[2:] != e5.shape[2:]:
                    up_c = F.interpolate(up_c, size=e5.shape[2:], mode="bilinear", align_corners=True)
                e5 = self.att5(g=up_c, x=e5)
            d5 = self.dec5(c, e5)
            
            # Decoder 4
            up_d5 = self.dec4.up(d5)
            if self.use_attention:
                if up_d5.shape[2:] != e4.shape[2:]:
                    up_d5 = F.interpolate(up_d5, size=e4.shape[2:], mode="bilinear", align_corners=True)
                e4 = self.att4(g=up_d5, x=e4)
            d4 = self.dec4(d5, e4)
            
            # Decoder 3
            up_d4 = self.dec3.up(d4)
            if self.use_attention:
                if up_d4.shape[2:] != e3.shape[2:]:
                    up_d4 = F.interpolate(up_d4, size=e3.shape[2:], mode="bilinear", align_corners=True)
                e3 = self.att3(g=up_d4, x=e3)
            d3 = self.dec3(d4, e3)

        # Decoder 2
        up_d3 = self.dec2.up(d3)
        if self.use_attention:
            if up_d3.shape[2:] != e2.shape[2:]:
                up_d3 = F.interpolate(up_d3, size=e2.shape[2:], mode="bilinear", align_corners=True)
            e2 = self.att2(g=up_d3, x=e2)
        d2 = self.dec2(d3, e2)
        
        # Decoder 1
        up_d2 = self.dec1.up(d2)
        if self.use_attention:
            if up_d2.shape[2:] != e1.shape[2:]:
                up_d2 = F.interpolate(up_d2, size=e1.shape[2:], mode="bilinear", align_corners=True)
            e1 = self.att1(g=up_d2, x=e1)
        d1 = self.dec1(d2, e1)
        
        conc = self.out_conc(d1)
        thick = self.out_thick(d1)
        
        if self.num_classes > 1:
            # Reshape thick to (B, T, Classes, H, W)
            # thick is currently (B, T*Classes, H, W)
            B, _, H, W = thick.shape
            thick = thick.view(B, self.out_channels, self.num_classes, H, W)
            # Don't apply sigmoid/softmax here if using CrossEntropyLoss (it expects logits)
            # But we might want to apply softmax for inference.
            # Let's return logits.
        
        return {"concentration": torch.sigmoid(conc), "thickness": thick}