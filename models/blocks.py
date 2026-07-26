import torch
import torch.nn as nn
import torch.nn.functional as F

from .dwt import haar_dwt2, haar_idwt2


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class VSSLiteBlock(nn.Module):
    """CPU-friendly substitute for VSS/Mamba-style mixing."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.dw = DepthwiseSeparableConv(channels)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.mix_h = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2), groups=channels)
        self.mix_v = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0), groups=channels)
        self.out = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, x):
        residual = x
        y = self.norm(x)
        y = F.gelu(self.dw(y))
        axial = self.mix_h(y) + self.mix_v(y)
        y = self.out(axial * torch.sigmoid(self.gate(y)))
        return residual + self.scale.tanh() * y


class FVSSLite(nn.Module):
    """Frequency-aware VSS-Lite with illumination conditioning."""

    def __init__(self, channels=24):
        super().__init__()
        self.in_proj = nn.Conv2d(4, channels, 3, padding=1)
        self.illum_proj = nn.Conv2d(1, channels, 3, padding=1)
        self.vss = nn.Sequential(VSSLiteBlock(channels), VSSLiteBlock(channels))
        self.freq_proj = nn.Conv2d(channels, 12, 3, padding=1)
        self.spatial_proj = nn.Conv2d(channels, 3, 3, padding=1)
        self.texture_gain = nn.Parameter(torch.tensor(0.20))

    def forward(self, reflectance, illumination):
        ll, lh, hl, hh, original_hw = haar_dwt2(reflectance)
        illum_small = F.interpolate(illumination, size=ll.shape[-2:], mode="bilinear", align_corners=False)
        base = torch.cat([ll.mean(1, keepdim=True), lh.mean(1, keepdim=True), hl.mean(1, keepdim=True), hh.mean(1, keepdim=True)], dim=1)
        feat = self.in_proj(base) + self.illum_proj(illum_small)
        feat = self.vss(feat)
        delta = self.freq_proj(feat)
        dll, dlh, dhl, dhh = torch.chunk(delta, 4, dim=1)
        gain = self.texture_gain.tanh()
        ll2 = ll + 0.05 * dll
        lh2 = lh + gain * dlh
        hl2 = hl + gain * dhl
        hh2 = hh + gain * dhh
        freq = haar_idwt2(ll2, lh2, hl2, hh2, original_hw)
        spatial = F.interpolate(self.spatial_proj(feat), size=reflectance.shape[-2:], mode="bilinear", align_corners=False)
        return (freq - reflectance) + 0.08 * spatial


class RAICLite(nn.Module):
    """Reflectance-aided illumination correction."""

    def __init__(self, channels=24):
        super().__init__()
        self.smooth = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.GELU(),
            DepthwiseSeparableConv(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.35))
        self.beta = nn.Parameter(torch.tensor(0.25))

    def forward(self, image, reflectance, illumination):
        target = image.max(dim=1, keepdim=True).values
        residual = self.smooth(torch.cat([reflectance, illumination], dim=1))
        corrected = illumination + self.alpha.sigmoid() * (target - illumination) + 0.12 * torch.tanh(residual)
        smooth = F.avg_pool2d(corrected, kernel_size=5, stride=1, padding=2)
        out = (1.0 - self.beta.sigmoid()) * corrected + self.beta.sigmoid() * smooth
        return out.clamp(0.03, 1.0)


class IGRELite(nn.Module):
    """Illumination-guided reflectance enhancement with RK2-style refinement."""

    def __init__(self, channels=24):
        super().__init__()
        self.fvss = FVSSLite(channels)
        self.gate_net = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )
        self.noise_suppress = nn.Sequential(
            DepthwiseSeparableConv(3),
            nn.Conv2d(3, 3, 3, padding=1),
        )

    def forward(self, image, reflectance, illumination):
        r_hat = (image / (illumination + 1e-3)).clamp(0.0, 1.5)
        r_hat = 0.65 * r_hat + 0.35 * reflectance
        k1 = self.fvss(r_hat, illumination)
        k2 = self.fvss((r_hat + k1).clamp(0.0, 1.5), illumination)
        gate = torch.sigmoid(self.gate_net(torch.cat([r_hat, illumination], dim=1)))
        refined = r_hat + gate * k1 + (1.0 - gate) * k2
        denoised = refined - 0.05 * torch.tanh(self.noise_suppress(refined))
        return denoised.clamp(0.0, 1.2)

