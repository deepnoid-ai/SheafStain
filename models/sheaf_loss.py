import torch

import torch.nn.functional as F


def compute_overlap_crop(ref_size, offset_x, offset_y):
    """Compute crop indices for the overlap region between ref and adj patches.

    ref patch: [0, ref_size] × [0, ref_size]
    adj patch: [offset_x, offset_x + ref_size] × [offset_y, offset_y + ref_size]

    Returns:
        ref_crop: (x_start, y_start, x_end, y_end) in ref coordinates
        adj_crop: (x_start, y_start, x_end, y_end) in adj coordinates
        valid: bool
    """
    # Overlap region in absolute coordinates
    ovl_x0 = max(0, offset_x)
    ovl_y0 = max(0, offset_y)
    ovl_x1 = min(ref_size, offset_x + ref_size)
    ovl_y1 = min(ref_size, offset_y + ref_size)

    if ovl_x0 >= ovl_x1 or ovl_y0 >= ovl_y1: return None, None, False

    ref_crop = (ovl_x0, ovl_y0, ovl_x1, ovl_y1) # In ref coordinates    
    adj_crop = (ovl_x0 - offset_x, ovl_y0 - offset_y, ovl_x1 - offset_x, ovl_y1 - offset_y) # In adj coordinates (shift by -offset)

    return ref_crop, adj_crop, True


def pixel_sheaf_loss(fake_ref, fake_adj, offset_x, offset_y, mode='l1', alpha=1.0, return_components=False):
    """Compute pixel-space sheaf gluing loss between two overlapping generated patches.

    Loss = mean_diff + alpha * pixel_diff

    mean_diff:  first-moment agreement — penalizes tile-level tone/color drift.
                |mean_spatial(ref) - mean_spatial(adj)| per-channel, averaged.
    pixel_diff: pixel-wise structure agreement — existing L1/L2 on overlap pixels.

    First-moment relaxation ensures background-dominant overlaps receive
    meaningful gradient signal even when pixel-wise L1 is trivially small.

    Args:
        fake_ref: generated IHC for ref patch [B, 3, H, W]
        fake_adj: generated IHC for adjacent patch [B, 3, H, W]
        offset_x: horizontal offset of adj relative to ref (pixels)
        offset_y: vertical offset of adj relative to ref (pixels)
        mode: 'l1' or 'l2'
        alpha: weight of pixel_diff relative to mean_diff (default: 1.0)
        return_components: if True, return (total, dict) with component values

    Returns:
        loss: scalar tensor (or (loss, dict) if return_components=True)
    """
    H, W = fake_ref.shape[2], fake_ref.shape[3]
    ref_crop, adj_crop, valid = compute_overlap_crop(W, offset_x, offset_y)
    
    if not valid:
        zero = torch.tensor(0.0, device=fake_ref.device, requires_grad=True)
        if return_components: return zero, {'mean_diff': 0.0, 'pixel_diff': 0.0}
        
        return zero

    rx0, ry0, rx1, ry1 = ref_crop
    ax0, ay0, ax1, ay1 = adj_crop

    ref_overlap = fake_ref[:, :, ry0:ry1, rx0:rx1]
    adj_overlap = fake_adj[:, :, ay0:ay1, ax0:ax1]

    # First-moment agreement: per-channel spatial mean difference
    mean_diff = (ref_overlap.mean(dim=[2, 3]) - adj_overlap.mean(dim=[2, 3])).abs().mean()

    # Pixel-wise structure agreement
    if mode == 'l1': pixel_diff = F.l1_loss(ref_overlap, adj_overlap)
    elif mode == 'l2': pixel_diff = F.mse_loss(ref_overlap, adj_overlap)
    else: raise ValueError(f"Unknown mode: {mode}")

    total = mean_diff + alpha * pixel_diff

    if return_components: return total, {'mean_diff': mean_diff.item(), 'pixel_diff': pixel_diff.item()}
    
    return total


def feature_sheaf_loss(feat_ref_list, feat_adj_list, offset_x, offset_y, input_size=256, mode='l1'):
    """Multi-scale feature-space sheaf gluing loss.

    Compares encoder intermediate features of ref and adj generated patches
    in their overlap region, at each layer's spatial resolution.

    This enforces sheaf restriction compatibility across coarser open covers:
    pixel space is the finest cover; deeper layers correspond to progressively
    coarser covers where gluing must still hold.

    Args:
        feat_ref_list: list of [B, C_l, H_l, W_l] features from ref (one per layer)
        feat_adj_list: list of [B, C_l, H_l, W_l] features from adj (one per layer)
        offset_x: horizontal offset of adj relative to ref (pixels, in input space)
        offset_y: vertical offset of adj relative to ref (pixels, in input space)
        input_size: spatial size of generator input (default: 256)
        mode: 'l1' or 'l2'

    Returns:
        loss: scalar tensor (mean across layers)
    """
    total_loss = torch.tensor(0.0, device=feat_ref_list[0].device, requires_grad=True)
    count = 0

    for feat_ref, feat_adj in zip(feat_ref_list, feat_adj_list):
        H_feat = feat_ref.shape[2]
        scale = H_feat / input_size

        scaled_ox = round(offset_x * scale)
        scaled_oy = round(offset_y * scale)
        ref_crop, adj_crop, valid = compute_overlap_crop(H_feat, scaled_ox, scaled_oy)
        
        if not valid: continue

        rx0, ry0, rx1, ry1 = ref_crop
        ax0, ay0, ax1, ay1 = adj_crop

        ref_overlap = feat_ref[:, :, ry0:ry1, rx0:rx1]
        adj_overlap = feat_adj[:, :, ay0:ay1, ax0:ax1]

        # Ensure crops match (rounding may cause 1-pixel mismatch)
        min_h = min(ref_overlap.shape[2], adj_overlap.shape[2])
        min_w = min(ref_overlap.shape[3], adj_overlap.shape[3])
        
        if min_h < 1 or min_w < 1: continue
        
        ref_overlap = ref_overlap[:, :, :min_h, :min_w]
        adj_overlap = adj_overlap[:, :, :min_h, :min_w]

        if mode == 'l1': total_loss = total_loss + F.l1_loss(ref_overlap, adj_overlap)
        else: total_loss = total_loss + F.mse_loss(ref_overlap, adj_overlap)
        
        count += 1

    if count == 0: return torch.tensor(0.0, device=feat_ref_list[0].device, requires_grad=True)
    
    return total_loss / count
