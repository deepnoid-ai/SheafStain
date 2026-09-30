import os
import csv
import piq
import torch
import pyiqa
import argparse

import numpy as np
import pandas as pd
import lpips as lpips_lib
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from scipy import stats as sp_stats
from skimage import color as skcolor

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from torchmetrics.image.kid import KernelInceptionDistance


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Per-image metric keys (used across compute, stats, and CSV functions)
PERIMAGE_KEYS = ('lpips', 'dists', 'psnr', 'ssim', 'scm',
                 'brisque', 'niqe', 'piqe', 'bcs', 'ts',
                 'dab_pearson_r', 'dab_kl', 'dab_jsd', 'iod_abs_error', 'miod_abs_error', 'fod_abs_error',)

def find_matched_pairs(pred_dir, gt_dir):
    """Find matching image pairs by filename stem (ignoring extension)."""
    pred_map = {}
    for f in os.listdir(pred_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTS: pred_map[stem] = os.path.join(pred_dir, f)

    gt_map = {}
    for f in os.listdir(gt_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTS: gt_map[stem] = os.path.join(gt_dir, f)

    common = sorted(set(pred_map.keys()) & set(gt_map.keys()))

    pred_only = len(pred_map) - len(common)
    gt_only = len(gt_map) - len(common)
   
    if pred_only > 0: print(f"[Warning] {pred_only} pred images without matching GT")
    if gt_only > 0: print(f"[Warning] {gt_only} GT images without matching pred")

    pairs = []
    for stem in common:
        pred_ext = os.path.splitext(pred_map[stem])[1].lstrip('.')
        pairs.append({'stem': stem,
                      'pred_ext': pred_ext,
                      'pred_path': pred_map[stem],
                      'gt_path': gt_map[stem],})

    return pairs

def load_image(path, device='cpu', target_size=None):
    """Load image as float32 tensor [0,1], shape (1, 3, H, W).

    Args:
        target_size: Optional (H, W) tuple. If provided and the image size
                     differs, resize using bicubic interpolation.
    """
    img = Image.open(path).convert('RGB')
    if target_size is not None and (img.height, img.width) != target_size: img = img.resize((target_size[1], target_size[0]), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0

    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def detect_size_mismatch(pairs):
    """Detect size mismatch between pred and GT by sampling the first pair.

    Returns:
        gt_size: (H, W) tuple of GT image size.
        need_resize: True if pred and GT sizes differ.
    """
    gt_img = Image.open(pairs[0]['gt_path'])
    pred_img = Image.open(pairs[0]['pred_path'])
    gt_size = (gt_img.height, gt_img.width)
    pred_size = (pred_img.height, pred_img.width)
    need_resize = gt_size != pred_size

    if need_resize:
        print(f"[Info] Size mismatch detected: pred {pred_size} vs GT {gt_size}")
        print(f"[Info] Pred images will be resized to {gt_size} (bicubic)")

    return gt_size, need_resize


def crop_to_patches(img_tensor, patch_size):
    """Crop image tensor (1,3,H,W) into non-overlapping patches.

    Returns:
        list of (1, 3, patch_size, patch_size) tensors
    """
    _, _, H, W = img_tensor.shape
    patches = []
    
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size): patches.append(img_tensor[:, :, y:y+patch_size, x:x+patch_size])
    
    return patches


def get_patch_cache_dirs(pred_dir, gt_dir, patch_size):
    """Compute persistent cache directory paths for extracted patches.

    Convention: append _{patch_size} to the original directory name.
    Example: .../her2 + patch_size=512 -> .../her2_512
    """
    pred_cache = pred_dir.rstrip('/') + f"_{patch_size}"
    gt_cache = gt_dir.rstrip('/') + f"_{patch_size}"
    
    return pred_cache, gt_cache


def ensure_patch_cache(pairs, patch_size, pred_cache_dir, gt_cache_dir, pred_target_size=None):
    """Extract patches to persistent cache directories if not already cached.

    Cache hit:  reads existing patches from cache dirs (no extraction).
    Cache miss: extracts non-overlapping patches and saves to cache dirs.

    Returns:
        patch_pairs: list of pair dicts pointing to cached patch files.
    """
    # Check cache validity
    cache_valid = False

    if os.path.isdir(pred_cache_dir) and os.path.isdir(gt_cache_dir):
        gt_img = Image.open(pairs[0]['gt_path'])
        n_patches = (gt_img.height // patch_size) * (gt_img.width // patch_size)
        expected = len(pairs) * n_patches

        existing_pairs = find_matched_pairs(pred_cache_dir, gt_cache_dir)
        if len(existing_pairs) == expected: cache_valid = True

    if cache_valid:
        print(f"[Cache hit] Using cached patches ({len(existing_pairs)} files)")
        print(f"  Pred: {pred_cache_dir}")
        print(f"  GT:   {gt_cache_dir}")
        
        return existing_pairs

    # Cache miss — extract patches
    print(f"[Cache miss] Extracting patches...")
    print(f"  Pred: {pred_cache_dir}")
    print(f"  GT:   {gt_cache_dir}")
    
    os.makedirs(pred_cache_dir, exist_ok=True)
    os.makedirs(gt_cache_dir, exist_ok=True)

    for pair in tqdm(pairs, desc="Extracting patches"):
        pred = load_image(pair['pred_path'], target_size=pred_target_size)
        gt = load_image(pair['gt_path'])

        pred_patches = crop_to_patches(pred, patch_size)
        gt_patches = crop_to_patches(gt, patch_size)

        stem = pair['stem']
        for idx, (pp, gp) in enumerate(zip(pred_patches, gt_patches)):
            fname = f"{stem}_p{idx}.png"
            pred_arr = (pp.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_arr = (gp.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_arr).save(os.path.join(pred_cache_dir, fname))
            Image.fromarray(gt_arr).save(os.path.join(gt_cache_dir, fname))

        del pred, gt

    patch_pairs = find_matched_pairs(pred_cache_dir, gt_cache_dir)
    print(f"Cached {len(patch_pairs)} patches")
    
    return patch_pairs

def compute_fid(pairs, device, batch_size):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    gt_files = [p['gt_path'] for p in pairs]
    pred_files = [p['pred_path'] for p in pairs]

    print("Computing FID (GT features)...")
    m1, s1 = calculate_activation_statistics(gt_files, model, batch_size, 2048, device, num_workers=2)
    
    print("Computing FID (pred features)...")
    m2, s2 = calculate_activation_statistics(pred_files, model, batch_size, 2048, device, num_workers=2)

    fid = calculate_frechet_distance(m1, s1, m2, s2)
    return fid

def compute_kid(pairs, device, batch_size, pred_target_size=None):
    n = len(pairs)
    subset_size = min(1000, n)

    if n < 2:
        print("[Warning] Too few images for KID computation")
        return 0.0, 0.0

    kid = KernelInceptionDistance(subset_size=subset_size, normalize=True).to(device)

    print("Computing KID (GT features)...")
    for i in tqdm(range(0, n, batch_size), desc="KID-GT"):
        batch_pairs = pairs[i:i+batch_size]
        imgs = torch.cat([load_image(p['gt_path']) for p in batch_pairs], dim=0)
        kid.update(imgs.to(device), real=True)

        del imgs
        torch.cuda.empty_cache()

    print("Computing KID (pred features)...")
    for i in tqdm(range(0, n, batch_size), desc="KID-pred"):
        batch_pairs = pairs[i:i+batch_size]
        imgs = torch.cat([load_image(p['pred_path'], target_size=pred_target_size) for p in batch_pairs], dim=0)
        kid.update(imgs.to(device), real=False)

        del imgs
        torch.cuda.empty_cache()

    kid_mean, kid_std = kid.compute()

    return kid_mean.item(), kid_std.item()


# =============================================================================
# Structural Correlation Metric (SCM)
# =============================================================================
# Reference: HistDiT (Saleem et al., ICPR 2026) — isolates the *structure*
# term of SSIM, which captures normalized cross-correlation of local
# structures while being invariant to mean/variance differences (i.e.,
# stain-intensity and contrast shifts between virtual and real IHC).
#
#     s(x, y) = (sigma_xy + C) / (sigma_x * sigma_y + C)
#
# Computed per-pixel with an 11x11 Gaussian window (standard SSIM window)
# and averaged over spatial locations and channels.
#
# Range: [-1, 1]. SCM = 1 on identical inputs; SCM ~ 0 on independent
# inputs; SCM < 0 on anti-correlated inputs. Higher is better.
def _gaussian_window_1d(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))

    return g / g.sum()

def _gaussian_window_2d(window_size, sigma, n_channels, device):
    g1d = _gaussian_window_1d(window_size, sigma).to(device)
    g2d = g1d[:, None] @ g1d[None, :]  # (W, W)
    window = g2d.expand(n_channels, 1, window_size, window_size).contiguous()

    return window

def compute_scm_single(pred, gt, window_size=11, sigma=1.5, C=(0.03) ** 2):
    """Structural Correlation Metric (HistDiT structure term of SSIM).

    Args:
        pred, gt: (B, 3, H, W) float tensors in [0, 1] on the same device.
        window_size: Gaussian window side (default 11, standard SSIM).
        sigma: Gaussian sigma (default 1.5, standard SSIM).
        C: numerical stability constant; default uses K2=0.03 with L=1.

    Returns:
        scalar float: mean SCM across the batch, spatial positions, channels.
    """
    if pred.dim() != 4 or gt.dim() != 4: raise ValueError("pred/gt must be 4D (B, C, H, W) tensors")
    if pred.shape != gt.shape: raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")

    n_channels = pred.shape[1]
    window = _gaussian_window_2d(window_size, sigma, n_channels, pred.device)
    pad = window_size // 2

    mu_x = F.conv2d(pred, window, padding=pad, groups=n_channels)
    mu_y = F.conv2d(gt,   window, padding=pad, groups=n_channels)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy   = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=pad, groups=n_channels) - mu_x_sq
    sigma_y_sq = F.conv2d(gt * gt,     window, padding=pad, groups=n_channels) - mu_y_sq
    sigma_xy   = F.conv2d(pred * gt,   window, padding=pad, groups=n_channels) - mu_xy

    # Clamp variances to be non-negative (convolution can produce tiny
    # negatives due to floating-point error)
    sigma_x = sigma_x_sq.clamp(min=0).sqrt()
    sigma_y = sigma_y_sq.clamp(min=0).sqrt()

    s_map = (sigma_xy + C) / (sigma_x * sigma_y + C)

    return float(s_map.mean().item())


def compute_perimage(pairs, device, pred_target_size=None, skip_dists=False):
    """Compute per-image FR metrics (LPIPS, DISTS, PSNR, SSIM) and NR metrics (BRISQUE, NIQE, PIQE)."""
    # FR metric functions
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    dists_fn = None
    if not skip_dists: dists_fn = DeepImageStructureAndTextureSimilarity(reduction='mean').to(device)

    # NR metric functions (pyiqa)
    niqe_fn = pyiqa.create_metric('niqe', device=device)
    piqe_fn = pyiqa.create_metric('piqe', device=device)

    results = []
    for pair in tqdm(pairs, desc="Per-image metrics"):
        pred = load_image(pair['pred_path'], device, target_size=pred_target_size)
        gt = load_image(pair['gt_path'], device)

        with torch.no_grad():
            # Original FR metrics
            lpips_val = lpips_fn(pred * 2 - 1, gt * 2 - 1).item()
            dists_val = 0.0

            if dists_fn is not None:
                dists_fn.update(pred, gt)
                dists_val = dists_fn.compute().item()
                dists_fn.reset()
            
            psnr_val = psnr_fn(pred, gt).item()
            ssim_val = ssim_fn(pred, gt).item()
            scm_val = compute_scm_single(pred, gt)

            # BRISQUE, NIQE, PIQE
            try: brisque_val = piq.brisque(pred).item()
            except (AssertionError, RuntimeError): brisque_val = float('nan')
            
            try: niqe_val = niqe_fn(pred).item()
            except (RuntimeError, torch._C._LinAlgError): niqe_val = float('nan')

            try: piqe_val = piqe_fn(pred).item()
            except (RuntimeError, Exception): piqe_val = float('nan')

        psnr_fn.reset()
        ssim_fn.reset()
        results.append({'lpips': lpips_val, 'dists': dists_val,
                        'psnr': psnr_val, 'ssim': ssim_val, 'scm': scm_val,
                        'brisque': brisque_val, 'niqe': niqe_val, 'piqe': piqe_val,})
        
        del pred, gt

    return results


def compute_perimage_patches(pairs, device, n_patches, pred_cache_dir, gt_cache_dir, skip_dists=False):
    """Compute per-image metrics at patch level using cached patches.

    Loads all patches for an image into a batch tensor, then computes
    LPIPS and DISTS in a single forward pass (batched). PSNR, SSIM, and
    NR metrics are computed per-patch. Results are averaged per image.
    """
    lpips_fn = lpips_lib.LPIPS(net='alex').to(device)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    dists_fn = None
    if not skip_dists:
        from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
        dists_fn = DeepImageStructureAndTextureSimilarity(reduction='mean').to(device)

    niqe_fn = pyiqa.create_metric('niqe', device=device)
    piqe_fn = pyiqa.create_metric('piqe', device=device)

    results = []
    for pair in tqdm(pairs, desc="Per-image metrics (patch)"):
        stem = pair['stem']

        # Load all patches into batch tensors (1 disk pass)
        preds, gts = [], []
        for idx in range(n_patches):
            fname = f"{stem}_p{idx}.png"
            preds.append(load_image(os.path.join(pred_cache_dir, fname), device))
            gts.append(load_image(os.path.join(gt_cache_dir, fname), device))
       
        pred_batch = torch.cat(preds, dim=0)   # (N, 3, H, W)
        gt_batch = torch.cat(gts, dim=0)
        
        del preds, gts

        with torch.no_grad():
            # --- Batched FR metrics (single forward pass) ---
            lpips_vals = lpips_fn(pred_batch * 2 - 1, gt_batch * 2 - 1)
            lpips_mean = lpips_vals.mean().item()

            dists_mean = 0.0
            if dists_fn is not None:
                dists_fn.update(pred_batch, gt_batch)
                dists_mean = dists_fn.compute().item()
                dists_fn.reset()

            # --- Per-patch metrics (PSNR, SSIM, SCM, NR) ---
            psnr_vals, ssim_vals, scm_vals = [], [], []
            brisque_vals, niqe_vals, piqe_vals = [], [], []

            for idx in range(n_patches):
                p = pred_batch[idx:idx+1]
                g = gt_batch[idx:idx+1]

                psnr_vals.append(psnr_fn(p, g).item())
                ssim_vals.append(ssim_fn(p, g).item())
                scm_vals.append(compute_scm_single(p, g))
        
                psnr_fn.reset()
                ssim_fn.reset()

                # BRISQUE, NIQE, PIQE
                try: brisque_vals.append(piq.brisque(p).item())
                except (AssertionError, RuntimeError): brisque_vals.append(float('nan'))
                
                try: niqe_vals.append(niqe_fn(p).item())
                except (RuntimeError, Exception): niqe_vals.append(float('nan'))
                
                try: piqe_vals.append(piqe_fn(p).item())
                except (RuntimeError, Exception): piqe_vals.append(float('nan'))

        results.append({'lpips': lpips_mean, 'dists': dists_mean,
                        'psnr': float(np.mean(psnr_vals)),
                        'ssim': float(np.mean(ssim_vals)),
                        'scm': float(np.mean(scm_vals)),
                        # Skipped for speed
                        'brisque': float(np.nanmean(brisque_vals)),
                        'niqe': float(np.nanmean(niqe_vals)),
                        'piqe': float(np.nanmean(piqe_vals)),})
        
        del pred_batch, gt_batch

    return results


# legacy (dump; not used in current version)
def compute_bcs_single(img_tensor, tile_size=256, strip_width=4):
    """Compute Boundary Consistency Score for a single stitched image.

    Measures pixel-level discontinuity at tile boundaries vs interior.

    BCS ↓ (lower = fewer stitching artifacts, more consistent boundaries).

    For each boundary line (horizontal and vertical at tile_size intervals):
      - boundary_diff: mean |row[y-k] - row[y+k]| across the boundary
      - interior_diff: same measurement at tile centers (natural tissue variation)

    BCS = boundary_diff - interior_diff
    Positive values indicate boundary artifacts beyond natural variation.
    Values near zero or negative indicate seamless stitching.

    Args:
        img_tensor: [1, 3, H, W] tensor in [0, 1]
        tile_size: generation tile size (determines boundary positions)
        strip_width: number of pixel rows/cols to compare on each side

    Returns:
        dict with bcs (float), bcs_boundary (float), bcs_interior (float)
    """
    _, _, H, W = img_tensor.shape

    boundary_diffs = []
    interior_diffs = []

    # Horizontal boundaries (y = k * tile_size)
    for y in range(tile_size, H, tile_size):
        y0 = max(y - strip_width, 0)
        y1 = min(y + strip_width, H)

        if y1 - y < 1 or y - y0 < 1: continue

        actual_sw = min(y - y0, y1 - y)
        strip_above = img_tensor[:, :, y - actual_sw:y, :]
        strip_below = img_tensor[:, :, y:y + actual_sw, :]
        
        boundary_diffs.append(F.l1_loss(strip_above, strip_below).item())

        # Interior reference: same comparison at tile center
        cy = y - tile_size // 2
        cy0 = max(cy - actual_sw, 0)
        cy1 = min(cy + actual_sw, H)
        
        if cy1 - cy >= actual_sw and cy - cy0 >= actual_sw:
            ref_above = img_tensor[:, :, cy - actual_sw:cy, :]
            ref_below = img_tensor[:, :, cy:cy + actual_sw, :]
            
            interior_diffs.append(F.l1_loss(ref_above, ref_below).item())

    # Vertical boundaries (x = k * tile_size)
    for x in range(tile_size, W, tile_size):
        x0 = max(x - strip_width, 0)
        x1 = min(x + strip_width, W)

        if x1 - x < 1 or x - x0 < 1: continue

        actual_sw = min(x - x0, x1 - x)
        strip_left = img_tensor[:, :, :, x - actual_sw:x]
        strip_right = img_tensor[:, :, :, x:x + actual_sw]

        boundary_diffs.append(F.l1_loss(strip_left, strip_right).item())

        cx = x - tile_size // 2
        cx0 = max(cx - actual_sw, 0)
        cx1 = min(cx + actual_sw, W)

        if cx1 - cx >= actual_sw and cx - cx0 >= actual_sw:
            ref_left = img_tensor[:, :, :, cx - actual_sw:cx]
            ref_right = img_tensor[:, :, :, cx:cx + actual_sw]
            
            interior_diffs.append(F.l1_loss(ref_left, ref_right).item())

    if not boundary_diffs: return {'bcs': float('nan'), 'bcs_boundary': float('nan'), 'bcs_interior': float('nan')}

    mean_boundary = float(np.mean(boundary_diffs))
    mean_interior = float(np.mean(interior_diffs)) if interior_diffs else 0.0

    bcs = abs(mean_boundary - mean_interior) # BCS = |boundary - interior|  (0 = seamless, higher = artifact)

    return {'bcs': bcs, 'bcs_boundary': mean_boundary, 'bcs_interior': mean_interior}

# legacy (dump; not used in current version)
def compute_bcs_enhanced_single(img_tensor, tile_size=256, strip_width=4, tone_weight=1.0):
    """Compute Enhanced BCS: boundary consistency + tile-level tone drift.

    Extends BCS with a tile-mean tone consistency term that captures
    tile-level intensity drift (visible as background tone artifacts).

    BCS_enhanced = BCS_boundary + λ · tile_tone_drift

    where:
      BCS_boundary = |mean_boundary_L1 - mean_interior_L1|  (original BCS)
      tile_tone_drift = mean of |mean(tile_i) - mean(tile_j)| for
                        all adjacent tile pairs (horizontal + vertical)

    BCS_enhanced ↓ (lower = fewer artifacts at both boundary and tile level).

    Args:
        img_tensor: [1, 3, H, W] tensor in [0, 1]
        tile_size: generation tile size (determines boundary and tile positions)
        strip_width: number of pixel rows/cols for boundary comparison
        tone_weight: weight λ for tile-tone component (default: 1.0)

    Returns:
        dict with bcs_enhanced (float), bcs_boundary_component (float),
             bcs_tone_component (float), bcs_original (float)
    """
    _, _, H, W = img_tensor.shape

    # ── Component 1: Original BCS (boundary consistency) ──
    bcs_result = compute_bcs_single(img_tensor, tile_size, strip_width)
    bcs_original = bcs_result['bcs']

    # ── Component 2: Tile-mean tone drift ──
    # Extract mean RGB per tile
    rows = H // tile_size
    cols = W // tile_size
    tile_means = {}  # (row, col) → [3] mean RGB

    for r in range(rows):
        for c in range(cols):
            tile = img_tensor[:, :, r * tile_size:(r + 1) * tile_size, c * tile_size:(c + 1) * tile_size]
            tile_means[(r, c)] = tile.mean(dim=(0, 2, 3))  # [3] per-channel mean

    # Compute mean tone difference for all adjacent pairs
    tone_diffs = []

    for r in range(rows):
        for c in range(cols):
            # Right neighbor (horizontal adjacency)
            if c + 1 < cols:
                diff = (tile_means[(r, c)] - tile_means[(r, c + 1)]).abs().mean()
                tone_diffs.append(diff.item())

            # Bottom neighbor (vertical adjacency)
            if r + 1 < rows:
                diff = (tile_means[(r, c)] - tile_means[(r + 1, c)]).abs().mean()
                tone_diffs.append(diff.item())

    if not tone_diffs: tone_component = 0.0
    else: tone_component = float(np.mean(tone_diffs))

    # Combined metric
    if np.isnan(bcs_original): bcs_enhanced = tone_component * tone_weight
    else: bcs_enhanced = bcs_original + tone_weight * tone_component

    return {'bcs_enhanced': bcs_enhanced,
            'bcs_boundary_component': bcs_original if not np.isnan(bcs_original) else 0.0,
            'bcs_tone_component': tone_component,
            'bcs_original': bcs_original,}

# legacy (dump; not used in current version)
def _collect_boundary_diffs(img_arr, tile_size, strip_width, reduce_fn, h_list, v_list, interior_h, interior_v):
    """Shared iteration over tile boundaries for BCS variants.

    Appends (boundary_stat, interior_stat) for each horizontal/vertical
    boundary to the caller-provided lists. `reduce_fn` maps a per-pixel
    diff array to a scalar (e.g., `np.mean`, functools.partial(np.percentile, q=95)).

    Args:
        img_arr: (H, W, C) float array, C >= 1.
        h_list, v_list: boundary stats are appended here (horizontal/vertical).
        interior_h, interior_v: corresponding interior-reference stats.
    """
    H, W = img_arr.shape[:2]

    for y in range(tile_size, H, tile_size):
        y0 = max(y - strip_width, 0)
        y1 = min(y + strip_width, H)
        
        if y1 - y < 1 or y - y0 < 1: continue

        sw = min(y - y0, y1 - y)
        above = img_arr[y - sw:y, :]
        below = img_arr[y:y + sw, :]

        h_list.append(reduce_fn(np.abs(above - below)))

        cy = y - tile_size // 2
        if cy - sw >= 0 and cy + sw <= H:
            ref_above = img_arr[cy - sw:cy, :]
            ref_below = img_arr[cy:cy + sw, :]

            interior_h.append(reduce_fn(np.abs(ref_above - ref_below)))

    for x in range(tile_size, W, tile_size):
        x0 = max(x - strip_width, 0)
        x1 = min(x + strip_width, W)

        if x1 - x < 1 or x - x0 < 1: continue

        sw = min(x - x0, x1 - x)
        left = img_arr[:, x - sw:x]
        right = img_arr[:, x:x + sw]

        v_list.append(reduce_fn(np.abs(left - right)))

        cx = x - tile_size // 2
        if cx - sw >= 0 and cx + sw <= W:
            ref_left = img_arr[:, cx - sw:cx]
            ref_right = img_arr[:, cx:cx + sw]

            interior_v.append(reduce_fn(np.abs(ref_left - ref_right)))

# legacy (dump; not used in current version)
def compute_bcs_lab(img_tensor, tile_size=256, strip_width=4):
    """BCS computed in LAB colorspace, with L and AB reported separately.

    Rationale: tile-level tone drift is primarily a chroma/lightness issue.
    Splitting into L (perceptual lightness) and AB (chroma) disentangles
    brightness-seam artifacts from color-seam artifacts.

    Args:
        img_tensor: [1, 3, H, W] tensor in [0, 1], RGB order.
        tile_size, strip_width: same semantics as compute_bcs_single.

    Returns:
        dict with bcs_L, bcs_L_boundary, bcs_L_interior,
                  bcs_AB, bcs_AB_boundary, bcs_AB_interior.
    """
    rgb = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    lab = skcolor.rgb2lab(rgb)  # L in [0, 100], AB in [-128, 127]

    L = lab[..., 0:1]
    AB = lab[..., 1:3]

    out = {}
    for name, arr in (('L', L), ('AB', AB)):
        h_b, v_b, h_i, v_i = [], [], [], []
        _collect_boundary_diffs(arr, tile_size, strip_width, np.mean, h_b, v_b, h_i, v_i)
        boundary = h_b + v_b
        interior = h_i + v_i

        if not boundary:
            out[f'bcs_{name}'] = float('nan')
            out[f'bcs_{name}_boundary'] = float('nan')
            out[f'bcs_{name}_interior'] = float('nan')
            continue

        mb = float(np.mean(boundary))
        mi = float(np.mean(interior)) if interior else 0.0

        out[f'bcs_{name}'] = abs(mb - mi)
        out[f'bcs_{name}_boundary'] = mb
        out[f'bcs_{name}_interior'] = mi

    return out

# legacy (dump; not used in current version)
def compute_bcs_p95(img_tensor, tile_size=256, strip_width=4):
    """BCS variant that uses 95th percentile of per-pixel seam diffs.

    Rationale: the mean can dilute localized artifacts over a long seam.
    p95 emphasizes the "worst part" of each seam — closer to what a
    human sees when scanning for tile boundaries. Computed in RGB [0, 1].

    Args:
        img_tensor: [1, 3, H, W] tensor in [0, 1].
        tile_size, strip_width: same semantics as compute_bcs_single.

    Returns:
        dict with bcs_p95, bcs_p95_boundary, bcs_p95_interior.
    """
    def p95(a):
        return float(np.percentile(a, 95))

    rgb = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    h_b, v_b, h_i, v_i = [], [], [], []
    _collect_boundary_diffs(rgb, tile_size, strip_width, p95, h_b, v_b, h_i, v_i)
    
    boundary = h_b + v_b
    interior = h_i + v_i
    
    if not boundary:
        return {'bcs_p95': float('nan'),
                'bcs_p95_boundary': float('nan'),
                'bcs_p95_interior': float('nan')}
    
    mb = float(np.mean(boundary))  # average of per-seam p95
    mi = float(np.mean(interior)) if interior else 0.0
    
    return {'bcs_p95': abs(mb - mi),
            'bcs_p95_boundary': mb,
            'bcs_p95_interior': mi,}

# legacy (dump; not used in current version)
def compute_bcs_hv(img_tensor, tile_size=256, strip_width=4):
    """BCS split by boundary orientation (horizontal vs vertical).

    Rationale: some architectures or overlap patterns produce asymmetric
    artifacts — e.g., stronger horizontal vs vertical seams depending on
    memory layout or attention window alignment. Separating H/V makes
    that asymmetry visible.

    Args:
        img_tensor: [1, 3, H, W] tensor in [0, 1].
        tile_size, strip_width: same semantics as compute_bcs_single.

    Returns:
        dict with bcs_h, bcs_h_boundary, bcs_h_interior,
                  bcs_v, bcs_v_boundary, bcs_v_interior.
    """
    rgb = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    h_b, v_b, h_i, v_i = [], [], [], []
    _collect_boundary_diffs(rgb, tile_size, strip_width, np.mean, h_b, v_b, h_i, v_i)

    def _agg(boundary, interior):
        if not boundary: return float('nan'), float('nan'), float('nan')
        
        mb = float(np.mean(boundary))
        mi = float(np.mean(interior)) if interior else 0.0
        
        return abs(mb - mi), mb, mi

    bcs_h, mb_h, mi_h = _agg(h_b, h_i)
    bcs_v, mb_v, mi_v = _agg(v_b, v_i)

    return {'bcs_h': bcs_h, 'bcs_h_boundary': mb_h, 'bcs_h_interior': mi_h,
            'bcs_v': bcs_v, 'bcs_v_boundary': mb_v, 'bcs_v_interior': mi_v,}

# legacy (dump; not used in current version)
def compute_bcs_all(pairs, device, tile_size=256, strip_width=4):
    """Compute BCS for all prediction images.

    BCS is a no-reference metric — only pred images are needed.

    Args:
        pairs: list of dicts with 'pred_path'
        device: torch device
        tile_size: generation tile size
        strip_width: strip width for comparison

    Returns:
        list of float (one BCS value per image)
    """
    bcs_values = []

    for pair in tqdm(pairs, desc="BCS", leave=False):
        img = load_image(pair['pred_path'], device)
        result = compute_bcs_single(img, tile_size, strip_width)
        bcs_values.append(result['bcs'])

        del img

    return bcs_values

# Tiling Score (TS)
# Reference: Madar & Fried, "Tiled Diffusion," CVPR 2025 (arXiv:2412.15185).
# Original repo: github.com/madaror/tiled-diffusion (no explicit license).
#
# Independent re-implementation from the paper's algorithm description.
# The authors' original `mean_absolute_gradient(img_1, img_2, direction)`
# was designed for two separate tile images to be stitched at a seam.
# SheafStain's test-time output is a single pre-stitched image, so we adapt:
# at each tile boundary in the stitched output we extract the seam column
# (or row) and compute the same max-of-three-means aggregation as the
# original paper.
#
# TS = max( g_seam, g_internal_left, g_internal_right )
#   g_seam           = mean |pixel[b - 1] - pixel[b]|  across the seam line
#   g_internal_left  = mean |pixel[b - w - 1] - pixel[b - w]|   (w pixels left)
#   g_internal_right = mean |pixel[b + w] - pixel[b + w + 1]|   (w pixels right)
#
# Per-image TS is averaged over all tile boundaries (vertical + horizontal).
# Lower TS = less prominent gradient anywhere in the seam region relative to
# natural internal texture. Values are on the [0, 1] pixel scale to match
# the rest of this file (paper is on [0, 255] scale, factor 255x).
def _mean_abs_diff_vec(a, b):
    """Mean absolute difference between two equal-shape arrays/tensors."""
    if isinstance(a, torch.Tensor): return float((a - b).abs().mean().item())
    
    return float(np.abs(a - b).mean())

def compute_ts_single(img_tensor, tile_size=256, width_size=15):
    """Tiling Score for a single stitched image (SheafStain adaptation).

    Args:
        img_tensor: [1, 3, H, W] tensor in [0, 1].
        tile_size:  generation tile size (boundary positions = k * tile_size).
        width_size: distance (pixels) from the seam used for the internal
                    reference gradients (default 15, matches Madar & Fried).

    Returns:
        dict with ts (float, per-image mean across boundaries),
             ts_seam (float, mean of seam gradients only — useful for
                      comparison against BCS).
    """
    _, _, H, W = img_tensor.shape
    ts_per_boundary = []
    seam_per_boundary = []

    w = int(width_size)

    # Vertical seams (x = k * tile_size): img_1 | img_2 meeting on X
    for b in range(tile_size, W, tile_size):
        # Need internal references on both sides of the seam
        if b - w - 1 < 0 or b + w + 1 >= W: continue

        col_seam_L = img_tensor[:, :, :, b - 1]
        col_seam_R = img_tensor[:, :, :, b]

        col_int_L1 = img_tensor[:, :, :, b - w - 1]
        col_int_L2 = img_tensor[:, :, :, b - w]

        col_int_R1 = img_tensor[:, :, :, b + w]
        col_int_R2 = img_tensor[:, :, :, b + w + 1]

        g_seam = _mean_abs_diff_vec(col_seam_L, col_seam_R)
        g_intL = _mean_abs_diff_vec(col_int_L1, col_int_L2)
        g_intR = _mean_abs_diff_vec(col_int_R1, col_int_R2)

        ts_per_boundary.append(max(g_seam, g_intL, g_intR))
        seam_per_boundary.append(g_seam)

    # Horizontal seams (y = k * tile_size): img_1 / img_2 meeting on Y
    for b in range(tile_size, H, tile_size):
        if b - w - 1 < 0 or b + w + 1 >= H: continue
        row_seam_T = img_tensor[:, :, b - 1, :]
        row_seam_B = img_tensor[:, :, b, :]

        row_int_T1 = img_tensor[:, :, b - w - 1, :]
        row_int_T2 = img_tensor[:, :, b - w, :]

        row_int_B1 = img_tensor[:, :, b + w, :]
        row_int_B2 = img_tensor[:, :, b + w + 1, :]

        g_seam = _mean_abs_diff_vec(row_seam_T, row_seam_B)
        g_intT = _mean_abs_diff_vec(row_int_T1, row_int_T2)
        g_intB = _mean_abs_diff_vec(row_int_B1, row_int_B2)

        ts_per_boundary.append(max(g_seam, g_intT, g_intB))
        seam_per_boundary.append(g_seam)

    if not ts_per_boundary: return {'ts': float('nan'), 'ts_seam': float('nan')}

    return {'ts': float(np.mean(ts_per_boundary)),
            'ts_seam': float(np.mean(seam_per_boundary)),}

def compute_ts_all(pairs, device, tile_size=256, width_size=15):
    """Compute TS for all prediction images (no-reference metric)."""
    ts_values = []
    for pair in tqdm(pairs, desc="TS", leave=False):
        img = load_image(pair['pred_path'], device)
        result = compute_ts_single(img, tile_size, width_size)
        ts_values.append(result['ts'])

        del img

    return ts_values


# DAB / IOD Metrics (Beer-Lambert color deconvolution)
# Stain vectors for Hematoxylin-DAB (Ruifrok & Johnston, 2001)
STAIN_MATRIX_HDAB = np.array([[0.650, 0.704, 0.286],   # Hematoxylin
                              [0.268, 0.570, 0.776],   # DAB
                              [0.7110, 0.4245, 0.5609]]) # Residual

def _normalize_stain_matrix(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return mat / norms

def color_deconvolution(rgb_image, stain_matrix=None):
    """Perform color deconvolution to separate stain channels.

    Args:
        rgb_image: (H, W, 3) uint8 [0, 255]
        stain_matrix: (3, 3) stain vectors. Default: H-DAB.

    Returns:
        (H, W, 3) optical density per stain channel.
        Channel 0 = Hematoxylin, 1 = DAB, 2 = Residual
    """
    if stain_matrix is None: stain_matrix = STAIN_MATRIX_HDAB.copy()
    stain_matrix = _normalize_stain_matrix(stain_matrix)
    
    img = rgb_image.astype(np.float64) + 1
    od = -np.log(img / 256.0)
    deconv_matrix = np.linalg.inv(stain_matrix)
    
    # Beer-Lambert deconvolution: concentrations = OD @ inv(S). Each row of the
    # stain matrix S is one stain vector, so od (N, 3 RGB) @ inv(S) gives the
    # per-stain concentrations; channel 1 is DAB. No transpose (od @ INV).
    channels = od.reshape(-1, 3) @ deconv_matrix
    
    return np.clip(channels, 0, None).reshape(rgb_image.shape)


def extract_dab_channel(rgb_image):
    """Extract DAB optical density from RGB IHC image."""
    return color_deconvolution(rgb_image)[:, :, 1]


def compute_iod(dab, threshold=0.1):
    """Integrated Optical Density: sum of DAB in positive regions."""
    mask = dab > threshold
    
    return float(np.sum(dab[mask]))


def compute_miod(dab, threshold=0.1):
    """Mean Integrated Optical Density in positive regions."""
    mask = dab > threshold
    n = np.sum(mask)
    
    return float(np.sum(dab[mask]) / n) if n > 0 else 0.0


def compute_fod(dab, threshold=0.1):
    """Fraction of DAB-positive area."""
    return float(np.sum(dab > threshold) / dab.size)


def compute_dab_pearson_r(dab_real, dab_gen):
    """Pearson correlation of DAB intensities (PSPStain metric)."""
    flat_r = dab_real.flatten()
    flat_g = dab_gen.flatten()
    
    if np.std(flat_r) < 1e-8 or np.std(flat_g) < 1e-8: return 0.0
    r, _ = sp_stats.pearsonr(flat_r, flat_g)
    
    return float(r)


def compute_dab_kl(dab_real, dab_gen, n_bins=256, epsilon=1e-10):
    """KL divergence between DAB histograms (UNIStainNet metric)."""
    max_val = max(dab_real.max(), dab_gen.max(), 1e-8)
    bins = np.linspace(0, max_val, n_bins + 1)
    
    hist_r, _ = np.histogram(dab_real.flatten(), bins=bins, density=True)
    hist_g, _ = np.histogram(dab_gen.flatten(), bins=bins, density=True)
    
    hist_r = (hist_r + epsilon) / (hist_r + epsilon).sum()
    hist_g = (hist_g + epsilon) / (hist_g + epsilon).sum()
    
    return float(np.sum(hist_r * np.log(hist_r / hist_g)))


def compute_dab_jsd(dab_real, dab_gen, n_bins=256, epsilon=1e-10):
    """Jensen-Shannon divergence between DAB histograms (symmetric)."""
    max_val = max(dab_real.max(), dab_gen.max(), 1e-8)
    bins = np.linspace(0, max_val, n_bins + 1)
    
    hist_r, _ = np.histogram(dab_real.flatten(), bins=bins, density=True)
    hist_g, _ = np.histogram(dab_gen.flatten(), bins=bins, density=True)
    
    hist_r = (hist_r + epsilon) / (hist_r + epsilon).sum()
    hist_g = (hist_g + epsilon) / (hist_g + epsilon).sum()
    
    m = (hist_r + hist_g) / 2
    kl_rm = float(np.sum(hist_r * np.log(hist_r / m)))
    kl_gm = float(np.sum(hist_g * np.log(hist_g / m)))
    
    return (kl_rm + kl_gm) / 2


def compute_dab_pair(gt_path, pred_path, pred_target_size=None, threshold=0.1, n_bins=256):
    """Compute DAB/IOD metrics for a single image pair.

    Returns dict with: dab_pearson_r, dab_kl, dab_jsd,
                       iod_abs_error, miod_abs_error, fod_abs_error
    """
    gt_img = np.array(Image.open(gt_path).convert('RGB'))
    pred_img = np.array(Image.open(pred_path).convert('RGB'))

    if pred_target_size is not None:
        pred_img = np.array(Image.fromarray(pred_img).resize((pred_target_size[1], pred_target_size[0]), Image.BICUBIC))

    dab_gt = extract_dab_channel(gt_img)
    dab_pred = extract_dab_channel(pred_img)

    return {'dab_pearson_r': compute_dab_pearson_r(dab_gt, dab_pred),
            'dab_kl': compute_dab_kl(dab_gt, dab_pred, n_bins),
            'dab_jsd': compute_dab_jsd(dab_gt, dab_pred, n_bins),
            'iod_abs_error': abs(compute_iod(dab_gt, threshold) - compute_iod(dab_pred, threshold)),
            'miod_abs_error': abs(compute_miod(dab_gt, threshold) - compute_miod(dab_pred, threshold)),
            'fod_abs_error': abs(compute_fod(dab_gt, threshold) - compute_fod(dab_pred, threshold)),}

def compute_dab_all(pairs, pred_target_size=None, threshold=0.1, n_bins=256):
    """Compute DAB/IOD metrics for all pairs.

    Returns list of dicts (one per image), same keys as compute_dab_pair.
    """
    results = []
    for pair in tqdm(pairs, desc="DAB/IOD metrics"):
        r = compute_dab_pair(pair['gt_path'], pair['pred_path'], pred_target_size, threshold, n_bins)
        results.append(r)

    return results


# Sample Grid Visualization
def save_sample_grid(pairs, he_dir=None, output_path='sample_grid.png', n=8):
    """Save H&E | Generated | Real comparison grid.

    Args:
        pairs: list of pair dicts with gt_path, pred_path, stem
        he_dir: directory containing H&E images (optional)
        output_path: output PNG path
        n: number of sample rows
    """
    n = min(n, len(pairs))
    sample_pairs = pairs[:n]

    # Determine image size from first GT
    first_gt = Image.open(sample_pairs[0]['gt_path']).convert('RGB')
    img_w, img_h = first_gt.size

    has_he = he_dir is not None and os.path.isdir(he_dir)
    n_cols = 3 if has_he else 2
    gap = 4  # pixel gap between images

    grid_w = n_cols * img_w + (n_cols - 1) * gap
    grid_h = n * img_h + (n - 1) * gap

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    for row, pair in enumerate(sample_pairs):
        y = row * (img_h + gap)
        col = 0

        if has_he:
            # Find H&E image by stem
            he_path = None

            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                candidate = os.path.join(he_dir, pair['stem'] + ext)
                if os.path.exists(candidate):
                    he_path = candidate; break

            if he_path:
                he_img = Image.open(he_path).convert('RGB').resize((img_w, img_h), Image.BICUBIC)
                grid.paste(he_img, (col * (img_w + gap), y))

            col += 1

        # Generated
        pred_img = Image.open(pair['pred_path']).convert('RGB').resize((img_w, img_h), Image.BICUBIC)
        grid.paste(pred_img, (col * (img_w + gap), y))

        col += 1

        # Real (GT)
        gt_img = Image.open(pair['gt_path']).convert('RGB').resize((img_w, img_h), Image.BICUBIC)
        grid.paste(gt_img, (col * (img_w + gap), y))

    grid.save(output_path)

    print(f"Sample grid saved: {output_path} ({n} rows, {n_cols} cols)")


# Per-class Statistics
def load_labels(labels_csv, split=None):
    """Load image labels from CSV. Returns {stem: label} dict."""
    df = pd.read_csv(labels_csv)
    if split: df = df[df['split'] == split]

    label_col = None
    for col in ['label', 'class', 'grade', 'her2_grade']:
        if col in df.columns:
            label_col = col; break
    
    if label_col is None: return None

    return {str(row['image_id']): int(row[label_col]) for _, row in df.iterrows()}


def compute_perclass_stats(perimage, pairs, label_map):
    """Compute per-class mean/std for all metrics.

    Returns dict: {label: {metric: {'mean': x, 'std': x, 'n': x}}}
    """
    class_data = defaultdict(list)

    for pair, metrics in zip(pairs, perimage):
        label = label_map.get(pair['stem'], -1)
        if label >= 0: class_data[label].append(metrics)

    result = {}
    for label in sorted(class_data.keys()):
        items = class_data[label]
        n = len(items)
        stats = {'n': n}

        for key in PERIMAGE_KEYS:
            vals = np.array([m[key] for m in items])
            stats[key] = {'mean': float(np.nanmean(vals)),
                          'std': float(np.nanstd(vals)) if n > 1 else 0.0,}

        result[label] = stats

    return result


# CSV output
def compute_stats(perimage):
    """Compute mean and std for per-image metrics.

    Uses nanmean/nanstd to gracefully handle NaN values (e.g. BRISQUE
    failures on uniform patches).

    Returns:
        (means, stds, nan_counts): dicts keyed by PERIMAGE_KEYS.
        nan_counts[key] = number of NaN values for that metric.
    """
    n = len(perimage)
    means, stds, nan_counts = {}, {}, {}

    for key in PERIMAGE_KEYS:
        vals = np.array([r[key] for r in perimage])
        nan_cnt = int(np.isnan(vals).sum())
        nan_counts[key] = nan_cnt
        means[key] = float(np.nanmean(vals))
        valid_n = n - nan_cnt
        stds[key] = float(np.nanstd(vals, ddof=1)) if valid_n > 1 else 0.0

    return means, stds, nan_counts


def write_csv(pairs, perimage, fid, kid_mean, kid_std, output_csv):
    """Write evaluation results to CSV.

    Per-image rows: per-image metrics filled; FID, KID left blank.
    Average row:    all metrics filled.
    Std row:        per-image metric std (FID empty, KID std from torchmetrics).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    means, stds, nan_counts = compute_stats(perimage)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['img_name', 'img_ext', 'gt_path', 'pred_path',
                         'FID', 'KID',
                         'LPIPS', 'DISTS', 'PSNR', 'SSIM', 'SCM',
                         'BRISQUE', 'NIQE', 'PIQE', 'BCS', 'TS',
                         'DAB_Pearson_R', 'DAB_KL', 'DAB_JSD',
                         'IOD_AbsErr', 'mIOD_AbsErr', 'FOD_AbsErr',])

        def _fmt(val, fmt):
            return 'NaN' if np.isnan(val) else f"{val:{fmt}}"

        for pair, m in zip(pairs, perimage):
            writer.writerow([pair['stem'],
                             pair['pred_ext'],
                             pair['gt_path'],
                             pair['pred_path'],
                             '', '',
                             _fmt(m['lpips'], '.6f'),
                             _fmt(m['dists'], '.6f'),
                             _fmt(m['psnr'], '.4f'),
                             _fmt(m['ssim'], '.6f'),
                             _fmt(m['scm'], '.6f'),
                             _fmt(m['brisque'], '.4f'),
                             _fmt(m['niqe'], '.4f'),
                             _fmt(m['piqe'], '.4f'),
                             _fmt(m['bcs'], '.6f'),
                             _fmt(m['ts'], '.6f'),
                             _fmt(m['dab_pearson_r'], '.4f'),
                             _fmt(m['dab_kl'], '.6f'),
                             _fmt(m['dab_jsd'], '.6f'),
                             _fmt(m['iod_abs_error'], '.4f'),
                             _fmt(m['miod_abs_error'], '.6f'),
                             _fmt(m['fod_abs_error'], '.6f'),])

        writer.writerow(['average', '', '', '',
                         f"{fid:.4f}",
                         f"{kid_mean:.6f}",
                         f"{means['lpips']:.6f}",
                         f"{means['dists']:.6f}",
                         f"{means['psnr']:.4f}",
                         f"{means['ssim']:.6f}",
                         f"{means['scm']:.6f}",
                         f"{means['brisque']:.4f}",
                         f"{means['niqe']:.4f}",
                         f"{means['piqe']:.4f}",
                         f"{means['bcs']:.6f}",
                         f"{means['ts']:.6f}",
                         f"{means['dab_pearson_r']:.4f}",
                         f"{means['dab_kl']:.6f}",
                         f"{means['dab_jsd']:.6f}",
                         f"{means['iod_abs_error']:.4f}",
                         f"{means['miod_abs_error']:.6f}",
                         f"{means['fod_abs_error']:.6f}",])

        writer.writerow(['std', '', '', '',
                         '',
                         f"{kid_std:.6f}",
                         f"{stds['lpips']:.6f}",
                         f"{stds['dists']:.6f}",
                         f"{stds['psnr']:.4f}",
                         f"{stds['ssim']:.6f}",
                         f"{stds['scm']:.6f}",
                         f"{stds['brisque']:.4f}",
                         f"{stds['niqe']:.4f}",
                         f"{stds['piqe']:.4f}",
                         f"{stds['bcs']:.6f}",
                         f"{stds['ts']:.6f}",
                         f"{stds['dab_pearson_r']:.4f}",
                         f"{stds['dab_kl']:.6f}",
                         f"{stds['dab_jsd']:.6f}",
                         f"{stds['iod_abs_error']:.4f}",
                         f"{stds['miod_abs_error']:.6f}",
                         f"{stds['fod_abs_error']:.6f}",])


# Main
def main():
    parser = argparse.ArgumentParser(description="Quantitative evaluation for virtual staining")

    parser.add_argument('--pred_dir', required=True, help='Directory containing generated images')
    parser.add_argument('--gt_dir', required=True, help='Directory containing ground truth images')
    parser.add_argument('--output_csv', required=True, help='Output CSV file path')
    parser.add_argument('--device', default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for FID/KID computation (default: 50)')
    parser.add_argument('--no_dists', action='store_true', help='Skip DISTS metric computation')
    parser.add_argument('--patch_size', type=int, default=0, help='If >0, evaluate at patch level with non-overlapping '
                                                                  'crops of this size (e.g., 512 for 1024x1024 images). '
                                                                  'Removes seam artifact effects from metrics.')
    parser.add_argument('--bcs_tile_size', type=int, default=256, help='Tile size for BCS computation (boundary positions). '
                                                                       'Should match the generation patch size (default: 256)')
    parser.add_argument('--bcs_strip_width', type=int, default=4, help='Strip width for BCS boundary comparison (default: 4)')
    parser.add_argument('--no_bcs', action='store_true', help='Skip BCS metric computation')
    parser.add_argument('--ts_tile_size', type=int, default=256, help='Tile size for TS computation (boundary positions). '
                                                                      'Should match the generation patch size (default: 256)')
    parser.add_argument('--ts_width_size', type=int, default=15, help='Internal reference distance (pixels) for TS (default: 15, per Madar & Fried CVPR 2025)')
    parser.add_argument('--no_ts', action='store_true', help='Skip TS (Tiling Score) metric computation')
    parser.add_argument('--no_dab', action='store_true', help='Skip DAB/IOD metric computation')
    parser.add_argument('--dab_threshold', type=float, default=0.1, help='DAB positive threshold in OD space (default: 0.1)')
    parser.add_argument('--labels_csv', default=None, help='Labels CSV for per-class breakdown (image_id, split, label)')
    parser.add_argument('--split', default=None, help='Filter to this split (requires --labels_csv)')
    parser.add_argument('--he_dir', default=None, help='H&E image directory for sample grid visualization')

    args = parser.parse_args()

    assert os.path.isdir(args.pred_dir), f"Not found: {args.pred_dir}"
    assert os.path.isdir(args.gt_dir), f"Not found: {args.gt_dir}"

    # 1. Match image pairs
    pairs = find_matched_pairs(args.pred_dir, args.gt_dir)
    assert len(pairs) > 0, "No matched image pairs found"

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Matched pairs: {len(pairs)}")

    # 2. Detect size mismatch
    gt_size, need_resize = detect_size_mismatch(pairs)
    pred_target_size = gt_size if need_resize else None

    # 2b. Patch-level setup
    patch_size = args.patch_size
    n_patches_per_img = 0

    if patch_size > 0:
        eff_h, eff_w = gt_size
        n_y, n_x = eff_h // patch_size, eff_w // patch_size
        n_patches_per_img = n_y * n_x
        rem_y, rem_x = eff_h % patch_size, eff_w % patch_size

        print(f"\n[Patch mode] patch_size={patch_size}, grid={n_y}x{n_x} ({n_patches_per_img} patches/image)")

        if rem_y > 0 or rem_x > 0: print(f"[Warning] Image {eff_h}x{eff_w} not perfectly divisible by patch_size={patch_size} (remainder {rem_y}x{rem_x} discarded)")

        pred_cache_dir, gt_cache_dir = get_patch_cache_dirs(args.pred_dir, args.gt_dir, patch_size)
        patch_pairs = ensure_patch_cache(pairs, patch_size, pred_cache_dir, gt_cache_dir, pred_target_size)
        
        print(f"Total patches: {len(patch_pairs)} ({len(pairs)} images x {n_patches_per_img} patches)")

    # 3. Distribution-level metrics
    print("\n--- FID ---")
    fid_pairs = patch_pairs if patch_size > 0 else pairs
    fid = compute_fid(fid_pairs, device, args.batch_size)
    print(f"FID: {fid:.4f}")

    print("\n--- KID ---")
    kid_pairs = patch_pairs if patch_size > 0 else pairs
    kid_target = None if patch_size > 0 else pred_target_size
    kid_mean, kid_std = compute_kid(kid_pairs, device, args.batch_size, pred_target_size=kid_target)
    print(f"KID: {kid_mean:.6f} +/- {kid_std:.6f}")

    # 4. Per-image metrics
    fr_label = "LPIPS, PSNR, SSIM, SCM" if args.no_dists else "LPIPS, DISTS, PSNR, SSIM, SCM"
    metrics_label = f"{fr_label}, BRISQUE, NIQE, PIQE"
    if patch_size > 0:
        print(f"\n--- Per-image patch-level ({metrics_label}) ---")
        perimage = compute_perimage_patches(pairs, device, n_patches_per_img, pred_cache_dir, gt_cache_dir, skip_dists=args.no_dists)
    else:
        print(f"\n--- Per-image ({metrics_label}) ---")
        perimage = compute_perimage(pairs, device, pred_target_size=pred_target_size, skip_dists=args.no_dists)

    # 5. Boundary Consistency Score (computed on full-size pred images)
    if not args.no_bcs:
        print(f"\n--- BCS (tile={args.bcs_tile_size}, strip={args.bcs_strip_width}) ---")
        bcs_values = compute_bcs_all(pairs, device, args.bcs_tile_size, args.bcs_strip_width)
        
        for r, bcs_val in zip(perimage, bcs_values): r['bcs'] = bcs_val
        
        print(f"BCS: {np.nanmean(bcs_values):.6f} +/- {np.nanstd(bcs_values):.6f}")
    else:
        for r in perimage: r['bcs'] = float('nan')

    # 5a. Tiling Score (Madar & Fried, CVPR 2025)
    if not args.no_ts:
        print(f"\n--- TS (tile={args.ts_tile_size}, width={args.ts_width_size}) ---")
        ts_values = compute_ts_all(pairs, device, args.ts_tile_size, args.ts_width_size)
        for r, ts_val in zip(perimage, ts_values): r['ts'] = ts_val
        
        print(f"TS: {np.nanmean(ts_values):.6f} +/- {np.nanstd(ts_values):.6f}")
    else:
        for r in perimage: r['ts'] = float('nan')

    # 5b. DAB / IOD metrics (Beer-Lambert color deconvolution)
    if not args.no_dab:
        print(f"\n--- DAB/IOD metrics (threshold={args.dab_threshold}) ---")
        dab_results = compute_dab_all(pairs, pred_target_size, args.dab_threshold)
        for r, dab_r in zip(perimage, dab_results): r.update(dab_r)
        dab_means = {k: np.nanmean([d[k] for d in dab_results]) for k in dab_results[0]}

        print(f"  Pearson-R : {dab_means['dab_pearson_r']:.4f}")
        print(f"  DAB KL    : {dab_means['dab_kl']:.4f}")
        print(f"  DAB JSD   : {dab_means['dab_jsd']:.4f}")
        print(f"  IOD err   : {dab_means['iod_abs_error']:.4f}")
        print(f"  mIOD err  : {dab_means['miod_abs_error']:.6f}")
        print(f"  FOD err   : {dab_means['fod_abs_error']:.6f}")

    else:
        for r in perimage:
            for k in ('dab_pearson_r', 'dab_kl', 'dab_jsd', 'iod_abs_error', 'miod_abs_error', 'fod_abs_error'): r[k] = float('nan')

    # 6. Write CSV
    write_csv(pairs, perimage, fid, kid_mean, kid_std, args.output_csv)
    print(f"\nResults saved to {args.output_csv}")

    # 6b. Sample grid
    if args.he_dir:
        grid_path = os.path.splitext(args.output_csv)[0] + '_grid.png'
        save_sample_grid(pairs, he_dir=args.he_dir, output_path=grid_path)

    # 7. Summary
    means, stds, nan_counts = compute_stats(perimage)
    print(f"\n{'='*60}")
    print(f"  Matched : {len(perimage)} images")
    
    if patch_size > 0: print(f"  Patch   : {patch_size}x{patch_size} ({n_patches_per_img} patches/image)")
    
    print(f"  FID     : {fid:.4f}")
    print(f"  KID     : {kid_mean:.6f} +/- {kid_std:.6f}")
    print(f"  LPIPS   : {means['lpips']:.6f} +/- {stds['lpips']:.6f}")
    print(f"  DISTS   : {means['dists']:.6f} +/- {stds['dists']:.6f}")
    print(f"  PSNR    : {means['psnr']:.4f} +/- {stds['psnr']:.4f}")
    print(f"  SSIM    : {means['ssim']:.6f} +/- {stds['ssim']:.6f}")
    print(f"  SCM     : {means['scm']:.6f} +/- {stds['scm']:.6f}")
    print(f"  BRISQUE : {means['brisque']:.4f} +/- {stds['brisque']:.4f}")
    print(f"  NIQE    : {means['niqe']:.4f} +/- {stds['niqe']:.4f}")
    print(f"  PIQE    : {means['piqe']:.4f} +/- {stds['piqe']:.4f}")
    
    if not args.no_bcs: print(f"  BCS     : {means['bcs']:.6f} +/- {stds['bcs']:.6f}")
    if not args.no_ts: print(f"  TS      : {means['ts']:.6f} +/- {stds['ts']:.6f}")
    if not args.no_dab:
                        print(f"  DAB-R   : {means['dab_pearson_r']:.4f} +/- {stds['dab_pearson_r']:.4f}")
                        print(f"  DAB-KL  : {means['dab_kl']:.6f} +/- {stds['dab_kl']:.6f}")
                        print(f"  DAB-JSD : {means['dab_jsd']:.6f} +/- {stds['dab_jsd']:.6f}")
                        print(f"  IOD err : {means['iod_abs_error']:.4f} +/- {stds['iod_abs_error']:.4f}")
                        print(f"  mIOD err: {means['miod_abs_error']:.6f} +/- {stds['miod_abs_error']:.6f}")
                        print(f"  FOD err : {means['fod_abs_error']:.6f} +/- {stds['fod_abs_error']:.6f}")

    # 7b. Per-class breakdown (if labels available)
    if args.labels_csv:
        label_map = load_labels(args.labels_csv, args.split)
        if label_map:
            class_stats = compute_perclass_stats(perimage, pairs, label_map)
            print(f"\n  --- Per-class breakdown ---")

            for label, stats in class_stats.items():
                print(f"  Class {label} (n={stats['n']}):")

                for key in ('lpips', 'ssim', 'psnr', 'dab_pearson_r', 'dab_kl', 'iod_abs_error'):
                    if key in stats:
                        s = stats[key]
                        print(f"    {key:<16s}: {s['mean']:.4f} +/- {s['std']:.4f}")

    # NaN warnings
    total_nan = sum(nan_counts.values())
    if total_nan > 0:
        print(f"\n  [WARNING] NaN values detected (excluded from stats):")

        for key in PERIMAGE_KEYS:
            if nan_counts[key] > 0: print(f"    {key:>16s}: {nan_counts[key]} / {len(perimage)} images")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
