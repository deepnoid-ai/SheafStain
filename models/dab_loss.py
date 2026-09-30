import torch

import torch.nn.functional as F


def dab_intensity_loss(generated, target, dab_extractor):
    """Compute DAB intensity matching loss.

    Args:
        generated: [B, 3, H, W] in [-1, 1]
        target: [B, 3, H, W] in [-1, 1]
        dab_extractor: DABExtractor instance (from util.dab)

    Returns:
        loss: scalar tensor (differentiable through color deconvolution)
    """
    # DABExtractor auto-detects [-1, 1] and converts to [0, 1]
    dab_gen = dab_extractor.extract_dab_intensity(
        generated.float(), normalize="none")
    dab_tgt = dab_extractor.extract_dab_intensity(
        target.float(), normalize="none")

    gen_scores = _batched_top10_mean(dab_gen)
    tgt_scores = _batched_top10_mean(dab_tgt.detach())

    return F.l1_loss(gen_scores, tgt_scores)


def _batched_top10_mean(dab):
    """Compute mean of top-10% DAB intensity per sample.

    Gradient flows through the selected (top-10%) pixels via
    straight-through masking (analogous to ReLU).

    Args:
        dab: [B, H, W] DAB intensity map

    Returns:
        scores: [B] per-image p90 DAB scores
    """
    B = dab.shape[0]
    flat = dab.reshape(B, -1)

    # Clamp outliers at p99 (stabilizes quantile computation)
    p99 = torch.quantile(flat.detach(), 0.99, dim=1, keepdim=True)
    flat_clamped = flat.clamp(max=p99)

    # Top-10% threshold (computed on detached values for stable masking)
    p90 = torch.quantile(flat_clamped.detach(), 0.9, dim=1, keepdim=True)
    mask = (flat_clamped >= p90).float()

    # Masked mean: gradient flows to flat_clamped through mask (straight-through)
    masked_sum = (flat_clamped * mask).sum(dim=1)
    mask_count = mask.sum(dim=1).clamp(min=1)

    return masked_sum / mask_count
