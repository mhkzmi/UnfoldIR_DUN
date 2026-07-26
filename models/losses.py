import torch
import torch.nn.functional as F


def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def _gaussian_window(channels, size=7, sigma=1.5, dtype=torch.float32):
    coords = torch.arange(size, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).view(1, 1, size, size)
    return window.repeat(channels, 1, 1, 1)


def ssim_index(x, y):
    channels = x.shape[1]
    window = _gaussian_window(channels, dtype=x.dtype).to(x.device)
    mu_x = F.conv2d(x, window, padding=3, groups=channels)
    mu_y = F.conv2d(y, window, padding=3, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(x * x, window, padding=3, groups=channels) - mu_x2
    sigma_y = F.conv2d(y * y, window, padding=3, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=3, groups=channels) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2) + 1e-8)
    return value.clamp(-1.0, 1.0).mean()


def ssim_loss(pred, target):
    return 1.0 - ssim_index(pred.clamp(0, 1), target.clamp(0, 1))


def sobel_grad(x):
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return gx, gy


def gradient_loss(pred, target):
    px, py = sobel_grad(pred)
    tx, ty = sobel_grad(target)
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def color_consistency_loss(pred, target=None):
    means = pred.mean(dim=(2, 3))
    if target is not None:
        target_means = target.mean(dim=(2, 3))
        return F.l1_loss(means, target_means)
    rg = (means[:, 0] - means[:, 1]).abs()
    rb = (means[:, 0] - means[:, 2]).abs()
    gb = (means[:, 1] - means[:, 2]).abs()
    return (rg + rb + gb).mean()


def illumination_smoothness_loss(illumination, image=None):
    dx = (illumination[:, :, :, 1:] - illumination[:, :, :, :-1]).abs()
    dy = (illumination[:, :, 1:, :] - illumination[:, :, :-1, :]).abs()
    if image is None:
        return dx.mean() + dy.mean()
    gx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(1, keepdim=True)
    gy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(1, keepdim=True)
    wx = torch.exp(-8.0 * gx)
    wy = torch.exp(-8.0 * gy)
    return (dx * wx).mean() + (dy * wy).mean()


def texture_consistency_loss(image, reflectance):
    ix, iy = sobel_grad(image)
    rx, ry = sobel_grad(reflectance)
    return F.l1_loss(rx.abs(), ix.abs()) + F.l1_loss(ry.abs(), iy.abs())


def exposure_control_loss(enhanced, target=0.58):
    pooled = F.avg_pool2d(enhanced.mean(1, keepdim=True), kernel_size=16, stride=16, ceil_mode=True)
    return ((pooled - target) ** 2).mean()


def noise_suppression_loss(enhanced):
    blur = F.avg_pool2d(enhanced, kernel_size=3, stride=1, padding=1)
    detail = enhanced - blur
    return detail.abs().mean()


def saturation_penalty(enhanced):
    high = F.relu(enhanced - 0.98).mean()
    low = F.relu(0.02 - enhanced).mean()
    return high + low


def isic_loss(reflectances, illuminations):
    if len(reflectances) < 3 or len(illuminations) < 3:
        return reflectances[-1].new_tensor(0.0)
    total = reflectances[-1].new_tensor(0.0)
    count = 0
    for idx in range(max(1, len(reflectances) - 2), len(reflectances)):
        r_k = reflectances[idx]
        r_prev = reflectances[idx - 1]
        l_k = illuminations[idx]
        l_prev = illuminations[idx - 1]
        rec_a = r_k * l_prev
        rec_b = r_k * l_k
        gx_a, gy_a = sobel_grad(r_prev * l_k)
        gx_b, gy_b = sobel_grad(r_k * l_k)
        total = total + F.mse_loss(rec_a, rec_b) + F.l1_loss(gx_a, gx_b) + F.l1_loss(gy_a, gy_b)
        count += 1
    return total / max(count, 1)


def supervised_loss(enhanced, target, image, reflectances, illuminations, weights):
    r_last = reflectances[-1]
    l_last = illuminations[-1]
    return (
        weights.get("l1", 1.0) * l1_loss(enhanced, target)
        + weights.get("ssim", 0.25) * ssim_loss(enhanced, target)
        + weights.get("gradient", 0.12) * gradient_loss(enhanced, target)
        + weights.get("color", 0.08) * color_consistency_loss(enhanced, target)
        + weights.get("illumination_smoothness", 0.12) * illumination_smoothness_loss(l_last, image)
        + weights.get("texture", 0.08) * texture_consistency_loss(image, r_last)
        + weights.get("isic", 0.10) * isic_loss(reflectances, illuminations)
    )


def self_supervised_loss(enhanced, image, reflectances, illuminations, weights, target_exposure=0.58):
    r_last = reflectances[-1]
    l_last = illuminations[-1]
    reconstruction = (r_last * l_last).clamp(0, 1)
    return (
        weights.get("reconstruction", 1.0) * F.l1_loss(reconstruction, image)
        + weights.get("exposure", 0.20) * exposure_control_loss(enhanced, target_exposure)
        + weights.get("illumination_smoothness", 0.12) * illumination_smoothness_loss(l_last, image)
        + weights.get("color", 0.08) * color_consistency_loss(enhanced)
        + weights.get("noise", 0.05) * noise_suppression_loss(enhanced)
        + weights.get("saturation", 0.05) * saturation_penalty(enhanced)
        + weights.get("texture", 0.08) * texture_consistency_loss(image, r_last)
        + weights.get("isic", 0.10) * isic_loss(reflectances, illuminations)
    )

