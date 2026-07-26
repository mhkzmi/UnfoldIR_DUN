import torch
import torch.nn.functional as F


def retinex_initialization(image):
    illumination = image.max(dim=1, keepdim=True).values
    illumination = F.avg_pool2d(illumination, kernel_size=7, stride=1, padding=3).clamp(0.03, 1.0)
    reflectance = (image / (illumination + 1e-3)).clamp(0.0, 1.2)
    return reflectance, illumination


def simple_low_light_enhance(image, gamma=0.72):
    illum = image.max(dim=1, keepdim=True).values.clamp(0.03, 1.0)
    boosted = image / illum * illum.pow(gamma)
    return boosted.clamp(0.0, 1.0)

