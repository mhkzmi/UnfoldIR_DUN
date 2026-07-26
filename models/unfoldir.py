import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import IGRELite, RAICLite


def _smooth_map(x, radius=5):
    return F.avg_pool2d(x, kernel_size=radius, stride=1, padding=radius // 2)


class UnfoldIR(nn.Module):

    def __init__(
        self,
        stage_num=3,
        shared_weights=True,
        feature_channels=24,
        exposure_strength=1.0,
        contrast_strength=1.0,
        saturation_strength=1.0,
    ):
        super().__init__()
        self.stage_num = int(stage_num)
        self.shared_weights = bool(shared_weights)
        self.exposure_strength = float(exposure_strength)
        self.contrast_strength = float(contrast_strength)
        self.saturation_strength = float(saturation_strength)
        if self.shared_weights:
            self.raic = RAICLite(feature_channels)
            self.igre = IGRELite(feature_channels)
        else:
            self.raic = nn.ModuleList([RAICLite(feature_channels) for _ in range(self.stage_num)])
            self.igre = nn.ModuleList([IGRELite(feature_channels) for _ in range(self.stage_num)])
        self.final_gate = nn.Parameter(torch.tensor(0.45))

    def _stage_modules(self, idx):
        if self.shared_weights:
            return self.raic, self.igre
        return self.raic[idx], self.igre[idx]

    def initialize_retinex(self, image):
        illumination = image.max(dim=1, keepdim=True).values
        illumination = _smooth_map(illumination, 7).clamp(0.03, 1.0)
        reflectance = (image / (illumination + 1e-3)).clamp(0.0, 1.2)
        return reflectance, illumination

    def _tone_map(self, reflectance, illumination):
        corrected_l = illumination.pow(1.0 / max(0.35, 1.0 + 0.35 * self.exposure_strength)).clamp(0.03, 1.0)
        fused = (reflectance * corrected_l).clamp(0.0, 1.0)
        gray = fused.mean(dim=1, keepdim=True)
        contrast = (fused - gray) * (1.0 + 0.18 * self.contrast_strength) + gray
        luminance = contrast.mean(dim=1, keepdim=True)
        saturated = luminance + (contrast - luminance) * (1.0 + 0.12 * self.saturation_strength)
        gate = self.final_gate.sigmoid()
        retinex_view = reflectance.clamp(0.0, 1.0)
        return (gate * saturated + (1.0 - gate) * retinex_view).clamp(0.0, 1.0)

    def forward(self, image):
        image = image.clamp(0.0, 1.0)
        reflectance, illumination = self.initialize_retinex(image)
        reflectances = [reflectance]
        illuminations = [illumination]
        debug = {"stage_enhanced": []}
        for idx in range(self.stage_num):
            raic, igre = self._stage_modules(idx)
            illumination = raic(image, reflectance, illumination)
            reflectance = igre(image, reflectance, illumination)
            reflectance = reflectance.clamp(0.0, 1.2)
            illumination = illumination.clamp(0.03, 1.0)
            reflectances.append(reflectance)
            illuminations.append(illumination)
            debug["stage_enhanced"].append(self._tone_map(reflectance, illumination))
        enhanced = self._tone_map(reflectance, illumination)
        return enhanced, reflectances, illuminations, debug

