import torch
import torch.nn as nn
import numpy as np

# --- 1. Wavelet / FEM Tools ---
class HaarDownsample(nn.Module):
    """Discrete Wavelet Transform (Haar) for Frequency Decoupling."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 4
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        
        LL = (x00 + x01 + x10 + x11) / 4.0
        HL = (x00 - x01 + x10 - x11) / 4.0
        LH = (x00 + x01 - x10 - x11) / 4.0
        HH = (x00 - x01 - x10 + x11) / 4.0
        
        return LL, HL, LH, HH

class MultiLevelHaarDWT(nn.Module):
    """3-Level Multi-Scale Wavelet Decomposition (FEM)."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.dwt = HaarDownsample(in_channels)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Returns: LL3, [HF1, HF2, HF3] where HF = [HL, LH, HH] concatenated
        hfs = []
        ll = x
        for _ in range(3):
            if ll.shape[2] % 2 != 0 or ll.shape[3] % 2 != 0:
                # Pad to even dimensions for DWT if needed
                ll = nn.functional.pad(ll, (0, ll.shape[3] % 2, 0, ll.shape[2] % 2))
            ll, hl, lh, hh = self.dwt(ll)
            hf = torch.cat([hl, lh, hh], dim=1)
            hfs.append(hf)
        return ll, hfs

# --- 2. VSSM / Mamba Backbone Proxy ---
class MambaBlock(nn.Module):
    """Pure PyTorch implementation of Mamba Block (State-Space Model)."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=True,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        L = H * W
        u = x.view(B, C, L).transpose(1, 2) # (B, L, d_model)
        
        xz = self.in_proj(u) # (B, L, 2 * d_inner)
        x_proj, z = xz.chunk(2, dim=-1) # (B, L, d_inner) each
        
        x_proj = x_proj.transpose(1, 2)
        x_proj = self.conv1d(x_proj)[:, :, :L]
        x_proj = x_proj.transpose(1, 2)
        x_proj = self.act(x_proj)
        
        x_dbl = self.x_proj(x_proj) # (B, L, dt_rank + 2*d_state)
        delta, B_t, C_t = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        delta = torch.nn.functional.softplus(self.dt_proj(delta)) # (B, L, d_inner)
        
        # Simplified SSM step for PyTorch compatibility (no custom kernel)
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        y = torch.zeros(B, L, self.d_inner, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        # Selective scan (loop-based, proxy for parallel scan)
        # We precompute all sequence-level transformations to avoid launching 
        # thousands of tiny CUDA kernels inside the Python loop.
        # This speeds up the pure PyTorch fallback by roughly 4x-5x.
        
        delta_A = torch.exp(delta.unsqueeze(-1) * A) # (B, L, d_inner, d_state)
        delta_B_u = delta.unsqueeze(-1) * B_t.unsqueeze(2) * x_proj.unsqueeze(-1) # (B, L, d_inner, d_state)
        C_t_exp = C_t.unsqueeze(2) # (B, L, 1, d_state)
        
        for t in range(L):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            y[:, t, :] = (h * C_t_exp[:, t]).sum(dim=-1)
            
        y = y + x_proj * self.D
        y = y * self.act(z)
        out = self.out_proj(y)
        
        return out.transpose(1, 2).view(B, C, H, W) + x

class VSSMNet(nn.Module):
    """Dual-Stream, Frequency-Decoupled State-Space Backbone."""
    def __init__(self, in_channels: int = 2, scale: int = 4, dim: int = 64):
        super().__init__()
        self.scale = scale
        
        self.stem = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        
        # VSSM Blocks (using PyTorch MambaBlock)
        self.vssm_blocks = nn.Sequential(*[MambaBlock(d_model=dim) for _ in range(4)])
        
        # FEM (Frequency Decoupling) 3-Level
        self.multi_dwt = MultiLevelHaarDWT(dim)
        
        # Texture Branch (fuses HF1, HF2, HF3)
        self.tconv1 = nn.Sequential(nn.Conv2d(dim*3, dim, 3, 1, 1), nn.ReLU(True))
        self.tconv2 = nn.Sequential(nn.Conv2d(dim*3, dim, 3, 1, 1), nn.ReLU(True))
        self.tconv3 = nn.Sequential(nn.Conv2d(dim*3, dim, 3, 1, 1), nn.ReLU(True))
        self.tfuse = nn.Conv2d(dim, dim, 1)
        
        # Color Branch (processes LL3)
        self.cconv = nn.Conv2d(dim, dim, 1)
        
        self.fem_proj = nn.Conv2d(dim * 2, dim, kernel_size=1)
        
        # Upsampling via Sub-Pixel
        upsample_blocks = []
        num_upsamples = int(np.log2(scale)) if scale > 1 else 0
        for _ in range(num_upsamples):
            upsample_blocks.append(nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1))
            upsample_blocks.append(nn.PixelShuffle(upscale_factor=2))
            upsample_blocks.append(nn.SiLU())
            
        self.upsample = nn.Sequential(*upsample_blocks)
        self.conv_out = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hr, _ = self.forward_with_features(x)
        return hr
        
    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        
        vssm_feat = self.vssm_blocks(feat)
        
        # 3-Level DWT Decoupling
        ll3, hfs = self.multi_dwt(feat) # hfs: [HF1, HF2, HF3]
        
        # Texture Stream Processing
        hf1_feat = self.tconv1(hfs[0])
        hf2_feat = self.tconv2(hfs[1])
        hf3_feat = self.tconv3(hfs[2])
        
        # Upsample lower frequency HFs to match HF1 shape
        hf2_up = nn.functional.interpolate(hf2_feat, size=hf1_feat.shape[-2:], mode="bilinear", align_corners=False)
        hf3_up = nn.functional.interpolate(hf3_feat, size=hf1_feat.shape[-2:], mode="bilinear", align_corners=False)
        texture_feat = self.tfuse(hf1_feat + hf2_up + hf3_up)
        
        # Color Stream Processing
        ll3_feat = self.cconv(ll3)
        ll3_up = nn.functional.interpolate(ll3_feat, size=hf1_feat.shape[-2:], mode="bilinear", align_corners=False)
        
        freq_feat = self.fem_proj(torch.cat([texture_feat, ll3_up], dim=1))
        freq_feat = nn.functional.interpolate(freq_feat, size=vssm_feat.shape[-2:], mode="bilinear", align_corners=False)
        
        combined_feat = vssm_feat + freq_feat
        hr_feat = self.upsample(combined_feat)
        hr_ir = self.conv_out(hr_feat)
        
        return hr_ir, hr_feat

# --- 3. Cross-Attention Skip Connections ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Conv2d(q_dim, q_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(kv_dim, q_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(kv_dim, q_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(q_dim, q_dim, kernel_size=1)
        self.scale = (q_dim // heads) ** -0.5

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q.shape
        if kv.shape[-2:] != q.shape[-2:]:
            kv = nn.functional.interpolate(kv, size=(H, W), mode="bilinear", align_corners=False)
            
        # Remove transpose to perform attention across channels (D) instead of spatial tokens (N)
        Q = self.q_proj(q).view(B, self.heads, C // self.heads, -1) # B, heads, D, N
        K = self.k_proj(kv).view(B, self.heads, C // self.heads, -1) # B, heads, D, N
        V = self.v_proj(kv).view(B, self.heads, C // self.heads, -1) # B, heads, D, N
        
        # Manual channel attention math to bypass SDPA kernel limitations on large embedding dims (N=262144)
        scale = (H * W) ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale # (B, heads, D, N) @ (B, heads, N, D) -> (B, heads, D, D)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        attn_out = attn @ V # (B, heads, D, D) @ (B, heads, D, N) -> (B, heads, D, N)
        
        out = attn_out.reshape(B, C, H, W)
        return self.out_proj(out) + q

# --- 4. ControlNet-Guided Generative Colorizer ---
class ControlNetColorizer(nn.Module):
    """U-Net with Cross-Attention structural guidance."""
    def __init__(self, in_channels: int = 2, out_channels: int = 3, bridge_dim: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), nn.SiLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.SiLU())
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU())
        self.pool2 = nn.MaxPool2d(2)
        
        # Cross-Attention Bridges
        self.attn_bridge1 = CrossAttentionBlock(q_dim=32, kv_dim=bridge_dim)
        self.attn_bridge2 = CrossAttentionBlock(q_dim=64, kv_dim=bridge_dim)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.SiLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU())
        
        self.attn_bridge_bn = CrossAttentionBlock(q_dim=128, kv_dim=bridge_dim)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.SiLU())
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU())
        
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_feat = torch.zeros(x.size(0), 64, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return self.forward_conditioned(x, b_feat)
        
    def forward_conditioned(self, x: torch.Tensor, bridge_features: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x1_cond = self.attn_bridge1(x1, bridge_features)
        p1 = self.pool1(x1_cond)
        
        x2 = self.enc2(p1)
        x2_cond = self.attn_bridge2(x2, bridge_features)
        p2 = self.pool2(x2_cond)
        
        b = self.bottleneck(p2)
        b_cond = self.attn_bridge_bn(b, bridge_features)
        
        d2 = self.up2(b_cond)
        d2 = torch.cat([d2, x2_cond], dim=1)
        x_dec2 = self.dec2(d2)
        
        d1 = self.up1(x_dec2)
        d1 = torch.cat([d1, x1_cond], dim=1)
        x_dec1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(x_dec1))

# --- 5. Adversarial Discriminator ---
class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator for Adversarial Loss."""
    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
