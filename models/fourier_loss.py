import torch

import torch.nn.functional as F


def fourier_edge_loss(generated, target):
    """Compute Fourier spectral loss on high-frequency bands.

    Args:
        generated: [B, 3, H, W] in [-1, 1]
        target: [B, 3, H, W] in [-1, 1]

    Returns:
        loss: scalar tensor (differentiable through FFT → generator)
    """
    # [-1, 1] → [0, 1] → grayscale
    gen_gray = ((generated + 1) / 2).clamp(0, 1).mean(dim=1, keepdim=True)
    tgt_gray = ((target + 1) / 2).clamp(0, 1).mean(dim=1, keepdim=True)

    # 2D FFT → magnitude spectrum (log-scale for numerical stability)
    gen_mag = torch.log1p(torch.fft.fft2(gen_gray).abs())
    tgt_mag = torch.log1p(torch.fft.fft2(tgt_gray).abs())

    # High-frequency mask: outer 75% of frequency space
    H, W = gen_mag.shape[-2], gen_mag.shape[-1]
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device=generated.device).float() - cy
    x = torch.arange(W, device=generated.device).float() - cx
    dist = (y[:, None] ** 2 + x[None, :] ** 2).sqrt()
    max_dist = (cy ** 2 + cx ** 2) ** 0.5
    hf_mask = (dist > 0.25 * max_dist).float()

    return F.l1_loss(gen_mag * hf_mask, tgt_mag * hf_mask)
