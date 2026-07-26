import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .image_io import IMAGE_EXTS, load_image, list_images


def _resize_crop(tensor, size):
    if size <= 0:
        return tensor
    tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return tensor[0]


class PairedImageDataset(Dataset):
    def __init__(self, low_dir, high_dir, image_size=256):
        low_dir = Path(low_dir)
        high_dir = Path(high_dir)
        self.pairs = []
        highs = {p.name: p for p in high_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS} if high_dir.exists() else {}
        if low_dir.exists():
            for low in sorted(low_dir.iterdir()):
                if low.is_file() and low.suffix.lower() in IMAGE_EXTS and low.name in highs:
                    self.pairs.append((low, highs[low.name]))
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low, high = self.pairs[idx]
        low_t = _resize_crop(load_image(low), self.image_size)
        high_t = _resize_crop(load_image(high), self.image_size)
        return low_t, high_t, low.name


class LowOnlyDataset(Dataset):
    def __init__(self, low_dir, image_size=256):
        self.images = list_images(low_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        return _resize_crop(load_image(path), self.image_size), path.name


def make_synthetic_low(clean):
    gamma = random.uniform(1.6, 3.0)
    low = clean.clamp(0, 1).pow(gamma)
    _, h, w = low.shape
    yy = torch.linspace(-1, 1, h).view(1, h, 1)
    xx = torch.linspace(-1, 1, w).view(1, 1, w)
    cx = random.uniform(-0.5, 0.5)
    cy = random.uniform(-0.5, 0.5)
    mask = 0.35 + 0.65 * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / random.uniform(0.35, 1.2))
    if random.random() < 0.5:
        backlit = torch.exp(-((xx) ** 2 + (yy + 0.35) ** 2) / 0.12)
        mask = (mask + 0.35 * backlit).clamp(0.2, 1.2)
    low = low * mask
    color = torch.tensor([random.uniform(0.90, 1.05), random.uniform(0.88, 1.05), random.uniform(0.92, 1.12)]).view(3, 1, 1)
    low = low * color
    noise = torch.randn_like(low) * random.uniform(0.005, 0.025)
    poisson_like = torch.randn_like(low) * torch.sqrt(low.clamp(0, 1) + 1e-4) * 0.015
    return (low + noise + poisson_like).clamp(0, 1), clean.clamp(0, 1)


class SyntheticImageDataset(Dataset):
    def __init__(self, roots, image_size=256, samples_per_image=2):
        self.images = []
        for root in roots:
            self.images.extend(list_images(root))
        self.image_size = image_size
        self.samples_per_image = max(1, int(samples_per_image))

    def __len__(self):
        return len(self.images) * self.samples_per_image

    def __getitem__(self, idx):
        path = self.images[idx % len(self.images)]
        clean = _resize_crop(load_image(path), self.image_size)
        low, high = make_synthetic_low(clean)
        return low, high, path.name

