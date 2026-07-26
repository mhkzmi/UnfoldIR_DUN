import math

import torch

from models.losses import ssim_index


def psnr(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim(pred, target):
    with torch.no_grad():
        return float(ssim_index(pred.clamp(0, 1), target.clamp(0, 1)).item())

