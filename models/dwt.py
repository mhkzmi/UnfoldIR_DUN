import torch
import torch.nn.functional as F


def _pad_even(x):
    _, _, h, w = x.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (h, w)


def haar_dwt2(x):
    """Differentiable Haar DWT for BCHW tensors."""
    x, original_hw = _pad_even(x)
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh, original_hw


def haar_idwt2(ll, lh, hl, hh, original_hw=None):
    """Inverse of haar_dwt2."""
    b, c, h, w = ll.shape
    out = ll.new_zeros((b, c, h * 2, w * 2))
    out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    out[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    out[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    if original_hw is not None:
        h0, w0 = original_hw
        out = out[:, :, :h0, :w0]
    return out


def frequency_stack(x):
    ll, lh, hl, hh, original_hw = haar_dwt2(x)
    return torch.cat([ll, lh, hl, hh], dim=1), original_hw


def frequency_unstack(coeffs, original_hw):
    c = coeffs.shape[1] // 4
    ll, lh, hl, hh = coeffs[:, :c], coeffs[:, c : 2 * c], coeffs[:, 2 * c : 3 * c], coeffs[:, 3 * c :]
    return haar_idwt2(ll, lh, hl, hh, original_hw)

