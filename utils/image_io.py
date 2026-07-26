from pathlib import Path

import numpy as np
from PIL import Image
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def load_image(path, max_side=None):
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = tensor.detach().cpu().clamp(0, 1)
    if data.ndim == 4:
        data = data[0]
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
    arr = (data.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def tensor_to_pil(tensor, size=None):
    data = tensor.detach().cpu().clamp(0, 1)
    if data.ndim == 4:
        data = data[0]
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
    arr = (data.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(arr)
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img

