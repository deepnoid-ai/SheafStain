"""
SheafStain inference - per-tile sheaf-conditioned H&E -> IHC virtual staining.

For each 256x256 reference tile of the input H&E image, a frozen pathology
Vision Foundation Model (VFM; Prov-GigaPath / UNI2-h / Virchow2) encodes a
randomized open cover of overlapping 224x224 neighbourhood patches. The patch
tokens and CLS tokens form the per-tile sheaf conditioning (spatial_map +
neighborhood_cls) on which the Schrodinger-bridge generator is conditioned;
tile outputs are assembled with a smooth partition of unity.

Pipeline:
  1. stride-192 backbone open cover (raised-cosine overlap blending)
  2. FARD cover refinement (config: use_fard, default on): 9 extra tiles placed
     by DOCR-bimodal FFT-energy priority over spectrally complex regions
  3. FFT-adaptive per-tile overlap-patch count K (config: overlap_mode)
  4. cosine-taper partition of unity for the refinement tiles

Per-tile protocol (matches training):
  1. Extract overlap 224x224 patches (8 directions, stride 80, +/-32 jitter),
     bounded only by the image edge.
  2. VFM batched forward -> patch tokens + CLS tokens.
  3. Build spatial_map (build_spatial_conditioning_map) and neighborhood_cls
     (spectral-weighted or uniform mean of the cover's CLS tokens).
  4. Generator forward conditioned on (spatial_map, neighborhood_cls).
  5. Blend tile outputs with the partition of unity.

The tiling stride, overlap strategy, and FARD refinement read from
config.yaml (see its inference section). FARD defaults to the paper setting (on,
9 extra tiles, bimodal placement); use_fard: false selects the backbone-only
ablation.

Usage:
    bash script/run_inference.sh              # reads config.yaml
    python inference.py --config config.yaml --name NAME --epoch latest --gpu_ids 0
"""

import os
import cv2
import sys
import time
import json
import torch
import random

import numpy as np
import util.util as util

from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from models.sheaf_modules import load_gigapath, build_spatial_conditioning_map
from util.vfm_dispatch import load_vfm
from data.sheaf_dataset import compute_overlap_region, _direction_displacement, DIR_N, DIR_S, DIR_E, DIR_W, DIR_NE, DIR_NW, DIR_SE, DIR_SW


# Constants
VFM_DIM = 1536
VFM_TOKENS_PER_AXIS = 14
TOKEN_PX = 16
TILE_SIZE = 256                         # REF patch / generator crop
TILE_TOKENS = TILE_SIZE // TOKEN_PX     # 16
OVL_PATCH_SIZE = 224                    # VFM input
OVL_STRIDE = 80                         # overlap-patch major-axis displacement
OVL_JITTER_RANGE = 32                   # minor-axis jitter ±
OVL_NUM_JITTER_VH = 2                   # cardinal jitter count per direction
VFM_BATCH_SIZE = 32

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# Padding / cropping
def pad_to_aligned(image_np, align=TOKEN_PX):
    h, w = image_np.shape[:2]
    target_h = ((h + align - 1) // align) * align
    target_w = ((w + align - 1) // align) * align

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h == 0 and pad_w == 0: return image_np, (h, w, 0, 0)
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    padded = np.pad(image_np, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')

    return padded, (h, w, pad_top, pad_left)


def crop_to_original(result_padded, info):
    h, w, pad_top, pad_left = info

    return result_padded[pad_top:pad_top + h, pad_left:pad_left + w]


# Tissue / background helpers
def compute_tissue_ratio(patch_np):
    gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return np.sum(binary < 128) / binary.size


def compute_background_mask(patch_np):
    gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary >= 128


def background_tone_match(he_patch_np, target_mean, target_std):
    patch = he_patch_np.astype(np.float64)
    src_mean = patch.mean(axis=(0, 1))
    src_std = patch.std(axis=(0, 1)) + 1e-6
    matched = (patch - src_mean) / src_std * target_std + target_mean

    return matched.clip(0, 255).astype(np.uint8)


def extract_overlap_patches_for_ref(image_np, ref_px, ref_py, ref_size=TILE_SIZE, ovl_size=OVL_PATCH_SIZE, stride=OVL_STRIDE, num_jitter_vh=OVL_NUM_JITTER_VH, jitter_range=OVL_JITTER_RANGE, rng=None, target_count=None):
    if rng is None: rng = random.Random()

    H, W = image_np.shape[:2]

    ref_cx = ref_px + ref_size // 2
    ref_cy = ref_py + ref_size // 2

    patches = []
    metadata = []

    def _try(direction_id, dx, dy, jitter_offset=0):
        ocx = ref_cx + dx
        ocy = ref_cy + dy

        ox = ocx - ovl_size // 2
        oy = ocy - ovl_size // 2

        if not (0 <= ox and ox + ovl_size <= W and 0 <= oy and oy + ovl_size <= H): return False
        region = compute_overlap_region(ref_px, ref_py, ox, oy, ref_size=ref_size, ovl_size=ovl_size)
        
        if region is None: return False
        patch = image_np[oy:oy + ovl_size, ox:ox + ovl_size]
        
        patches.append(patch)
        metadata.append({'direction_id': direction_id,
                         'offset': stride,
                         'jitter': jitter_offset,
                         'position': (ox, oy),
                         'center': (ocx, ocy),
                         'region_in_ref': region['ref_crop'],
                         'region_in_ovl': region['ovl_crop'],})

        return True

    for d in (DIR_N, DIR_S, DIR_E, DIR_W):
        mx, my = _direction_displacement(d, stride)
        _try(d, mx, my)
        for _ in range(num_jitter_vh):
            j = rng.randint(-jitter_range, jitter_range)
            
            if d in (DIR_N, DIR_S): _try(d, mx + j, my, jitter_offset=j)
            else: _try(d, mx, my + j, jitter_offset=j)
    
    for d in (DIR_NE, DIR_NW, DIR_SE, DIR_SW):
        dx, dy = _direction_displacement(d, stride)
        _try(d, dx, dy)

    # subsample to target_count if requested and smaller
    if target_count is not None and 0 < target_count < len(patches):
        idx = sorted(rng.sample(range(len(patches)), target_count))
        patches = [patches[i] for i in idx]
        metadata = [metadata[i] for i in idx]

    return patches, metadata


def build_spatial_map_vectorized(patch_tokens, positions, ref_x, ref_y, ref_size=TILE_SIZE, ovl_size=OVL_PATCH_SIZE, token_px=TOKEN_PX):
    N, N_tok, D = patch_tokens.shape
    ref_grid = ref_size // token_px
    ovl_grid = ovl_size // token_px

    assert N_tok == ovl_grid * ovl_grid, (N_tok, ovl_grid)

    dev = patch_tokens.device
    if positions.device != dev: positions = positions.to(dev)

    # Token centers within each patch's 14x14 grid (in patch-local coords).
    rows = torch.arange(ovl_grid, device=dev)
    cols = torch.arange(ovl_grid, device=dev)

    rr, cc = torch.meshgrid(rows, cols, indexing='ij')
    dy = (rr * token_px + token_px // 2)
    dx = (cc * token_px + token_px // 2)

    # Absolute centers per patch [N, 14, 14]
    tx = positions[:, 0, None, None] + dx[None]
    ty = positions[:, 1, None, None] + dy[None]

    # Valid (inside ref patch) mask [N, 14, 14]
    valid = ((tx >= ref_x) & (tx < ref_x + ref_size) & (ty >= ref_y) & (ty < ref_y + ref_size))

    ref_col = ((tx - ref_x) // token_px).clamp(0, ref_grid - 1)
    ref_row = ((ty - ref_y) // token_px).clamp(0, ref_grid - 1)

    flat_valid = valid.flatten()
    if not flat_valid.any():
        return (torch.zeros(D, ref_grid, ref_grid, device=dev, dtype=patch_tokens.dtype), torch.zeros(ref_grid, ref_grid, dtype=torch.bool, device=dev))

    flat_tok = patch_tokens.reshape(-1, D)[flat_valid]
    flat_idx = (ref_row.flatten() * ref_grid + ref_col.flatten())[flat_valid]

    spatial_sum = torch.zeros(ref_grid * ref_grid, D, device=dev, dtype=patch_tokens.dtype)
    spatial_sum.index_add_(0, flat_idx, flat_tok)
    count = torch.zeros(ref_grid * ref_grid, device=dev, dtype=torch.float32)
    count.index_add_(0, flat_idx, torch.ones(flat_idx.numel(), device=dev, dtype=torch.float32))

    count_safe = count.clamp(min=1.0).unsqueeze(-1)
    spatial_avg = (spatial_sum.float() / count_safe).to(patch_tokens.dtype)
    spatial_map = spatial_avg.reshape(ref_grid, ref_grid, D).permute(2, 0, 1).contiguous()
    coverage_mask = (count > 0).reshape(ref_grid, ref_grid)

    return spatial_map, coverage_mask


# Spectral-similarity subordinate partition of unity
def _spectral_signature(patch_np, lo_freq=1.0 / 16.0, hi_freq=1.0 / 4.0, resize_to=OVL_PATCH_SIZE):
    if patch_np.ndim == 3: gray = (0.299 * patch_np[..., 0] + 0.587 * patch_np[..., 1] + 0.114 * patch_np[..., 2]).astype(np.float32)
    else: gray = patch_np.astype(np.float32)

    H, W = gray.shape
    if H != resize_to or W != resize_to:
        pil = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
        gray = np.asarray(pil.resize((resize_to, resize_to), Image.BICUBIC)).astype(np.float32)
        H = W = resize_to

    # Centered FFT magnitude (log for numerical stability across orders of magnitude in natural images)
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(F))

    # Annular mid-band mask
    cy, cx = H // 2, W // 2
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / float(min(H, W))
    mask = (r >= lo_freq) & (r < hi_freq)

    # Zero-mean + unit-norm → cosine similarity reduces to inner product
    sig = mag[mask].astype(np.float32)
    sig = sig - sig.mean()
    n = np.linalg.norm(sig)
    
    if n > 1e-8: sig = sig / n
    
    return sig


def compute_spectral_weights(ref_patch_np, overlap_patches_np, temperature,lo_freq=1.0 / 16.0, hi_freq=1.0 / 4.0):
    N = len(overlap_patches_np)

    if N == 0: return np.zeros(0, dtype=np.float32)
    if not np.isfinite(temperature): return np.full(N, 1.0 / N, dtype=np.float32)

    sig_ref = _spectral_signature(ref_patch_np)
    # Cosine sim ≡ inner product since both are unit-norm
    sims = np.empty(N, dtype=np.float32)
    for i in range(N):
        sig_i = _spectral_signature(overlap_patches_np[i])
        if sig_ref.shape != sig_i.shape: sims[i] = 0.0
        else: sims[i] = float(np.dot(sig_ref, sig_i))

    logits = sims / float(temperature)
    logits -= logits.max()  # numerical stability
    w = np.exp(logits)
    w /= w.sum()

    return w.astype(np.float32)


def compute_tile_conditioning_batched(image_np, tile_positions, vfm, device, vfm_batch_size=VFM_BATCH_SIZE, rng=None, spectral_T=None, per_tile_K=None):
    mean_dev = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    std_dev = IMAGENET_STD.to(device).view(1, 3, 1, 1)

    # Phase A: extract patches as uint8 numpy (CPU only, no H2D yet)
    per_tile_np = []
    per_tile_meta = []
    per_tile_range = []
    total = 0

    for tile_idx, (ref_px, ref_py) in enumerate(tile_positions):
        target_K = (per_tile_K[tile_idx] if per_tile_K is not None else None)
        patches, metadata = extract_overlap_patches_for_ref(image_np, ref_px, ref_py, rng=rng, target_count=target_K)
        
        if patches: arr = np.stack(patches, axis=0)  # [n, 224, 224, 3] uint8
        else: arr = np.zeros((0, OVL_PATCH_SIZE, OVL_PATCH_SIZE, 3), dtype=np.uint8)
        
        per_tile_np.append(arr)
        per_tile_meta.append(metadata)
        per_tile_range.append((total, total + arr.shape[0]))
        total += arr.shape[0]

    _vfm_D = getattr(getattr(vfm, 'spec', None), 'embed_dim', VFM_DIM)

    if total == 0:
        return [{'spatial_map': torch.zeros(_vfm_D, TILE_TOKENS, TILE_TOKENS), 'neighborhood_cls': torch.zeros(_vfm_D), 'n_patches': 0} for _ in tile_positions]

    # Phase B: single CPU stack + VFM in vfm_batch_size slices with
    # non-blocking H2D per batch (pipelined with GPU compute).
    all_np = np.concatenate(per_tile_np, axis=0) if per_tile_np else np.zeros((0, OVL_PATCH_SIZE, OVL_PATCH_SIZE, 3), dtype=np.uint8)

    all_t_cpu = torch.from_numpy(all_np).permute(0, 3, 1, 2).contiguous()
    
    try: all_t_cpu = all_t_cpu.pin_memory()
    except Exception: pass

    patch_tokens_all = []
    cls_tokens_all = []
    
    for i in range(0, total, vfm_batch_size):
        batch = all_t_cpu[i:i + vfm_batch_size].to(device, non_blocking=True).float().div_(255.0)
        batch = (batch - mean_dev) / std_dev

        with torch.no_grad():
            cls_b_dev, pt_b_dev = vfm.extract(batch)
            patch_tokens_all.append(pt_b_dev.cpu())
            cls_tokens_all.append(cls_b_dev.cpu())

    patch_tokens_all = torch.cat(patch_tokens_all, dim=0)  # [total, 196, D]
    cls_tokens_all = torch.cat(cls_tokens_all, dim=0)       # [total, D]

    results = []
    for tile_idx, (ref_px, ref_py) in enumerate(tile_positions):
        start, end = per_tile_range[tile_idx]
        n = end - start
        if n == 0:
            results.append({'spatial_map': torch.zeros(_vfm_D, TILE_TOKENS, TILE_TOKENS), 'neighborhood_cls': torch.zeros(_vfm_D), 'n_patches': 0,})
            continue

        tokens = patch_tokens_all[start:end]       # [n, 196, D]

        token_list = [tokens[i] for i in range(n)]
        meta_list = per_tile_meta[tile_idx]
        spatial_map, _ = build_spatial_conditioning_map(token_list, meta_list, ref_px, ref_py, ref_size=TILE_SIZE, ovl_size=OVL_PATCH_SIZE, token_px=TOKEN_PX)

        cls_stack = cls_tokens_all[start:end].float()   # [n, D]
        if spectral_T is None or not np.isfinite(spectral_T): neighborhood_cls = cls_stack.mean(dim=0)
        else:
            ref_patch_np = image_np[ref_py:ref_py + TILE_SIZE, ref_px:ref_px + TILE_SIZE]
            overlap_patches_np = [per_tile_np[tile_idx][i] for i in range(n)]

            w_np = compute_spectral_weights(ref_patch_np, overlap_patches_np, temperature=float(spectral_T))
            w = torch.from_numpy(w_np).to(cls_stack.dtype)  # [n]
            neighborhood_cls = (w.unsqueeze(-1) * cls_stack).sum(dim=0)

        results.append({'spatial_map': spatial_map.float(), 'neighborhood_cls': neighborhood_cls, 'n_patches': n,})

    return results


# Tile positions, weight maps, stitching
def generate_tile_positions(image_h, image_w, tile_size=256, stride=128):
    assert stride % TOKEN_PX == 0
    assert stride <= tile_size

    ys = list(range(0, image_h - tile_size + 1, stride))
    xs = list(range(0, image_w - tile_size + 1, stride))

    if not ys or ys[-1] + tile_size < image_h:
        last_y = max(0, image_h - tile_size)
        if last_y not in ys: ys.append(last_y)

    if not xs or xs[-1] + tile_size < image_w:
        last_x = max(0, image_w - tile_size)
        if last_x not in xs: xs.append(last_x)

    return [(px, py) for py in ys for px in xs]


def _spectral_complexity_map(image_np, lo_freq=1.0 / 16.0, hi_freq=1.0 / 4.0):
    if image_np.ndim == 3:
        gray = (0.299 * image_np[..., 0] + 0.587 * image_np[..., 1] + 0.114 * image_np[..., 2]).astype(np.float32)
    else: gray = image_np.astype(np.float32)

    H, W = gray.shape
    F = np.fft.fftshift(np.fft.fft2(gray))

    cy, cx = H // 2, W // 2
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / float(min(H, W))
    mask = (r >= lo_freq) & (r < hi_freq)

    F_mid = F * mask
    spatial = np.fft.ifft2(np.fft.ifftshift(F_mid))
    energy = (spatial.real ** 2 + spatial.imag ** 2).astype(np.float32)

    return energy


def _tile_energy_from_sat(sat, px, py, tile_size=TILE_SIZE):
    x0, y0 = px, py
    x1 = px + tile_size - 1
    y1 = py + tile_size - 1
    H, W = sat.shape

    x1 = min(x1, W - 1); y1 = min(y1, H - 1)
    total = sat[y1, x1]

    if y0 > 0: total -= sat[y0 - 1, x1]
    if x0 > 0: total -= sat[y1, x0 - 1]
    if x0 > 0 and y0 > 0: total += sat[y0 - 1, x0 - 1]

    return float(total)


def _determine_overlap_count(energy_value, q_low, q_high, K_min=5, K_max=16):
    if q_high <= q_low + 1e-12: return K_max  # degenerate spread → be conservative, use max
    if energy_value <= q_low: return int(K_min)
    if energy_value >= q_high: return int(K_max)
    
    t = (energy_value - q_low) / (q_high - q_low)
    
    return int(round(K_min + t * (K_max - K_min)))


def generate_spectrally_adaptive_positions(image_np, tile_size=TILE_SIZE, base_stride=192, extra_tiles=9, min_dist_extra=None, candidate_step=None, lo_freq=1.0 / 16.0, hi_freq=1.0 / 4.0, return_meta=False, metric_mode='structured'):
    H, W = image_np.shape[:2]

    if min_dist_extra is None: min_dist_extra = max(TOKEN_PX, base_stride // 2)
    if candidate_step is None: candidate_step = TOKEN_PX

    # Backbone (guaranteed coverage)
    base_positions = generate_tile_positions(H, W, tile_size, base_stride)
    if extra_tiles <= 0:
        if return_meta: return {'positions': base_positions, 'backbone_count': len(base_positions), 'energy_map': None, 'sat': None,}

        return base_positions

    # Complexity map + per-tile energy integral
    energy = _spectral_complexity_map(image_np, lo_freq, hi_freq)

    # Summed-area table for O(1) per-tile energy queries
    sat = energy.cumsum(axis=0).cumsum(axis=1)

    def tile_energy(px, py):
        x0, y0 = px, py
        x1, y1 = px + tile_size - 1, py + tile_size - 1
        total = sat[y1, x1]

        if y0 > 0: total -= sat[y0 - 1, x1]
        if x0 > 0: total -= sat[y1, x0 - 1]
        if x0 > 0 and y0 > 0: total += sat[y0 - 1, x0 - 1]
        
        return float(total)

    max_px = W - tile_size
    max_py = H - tile_size
    candidates = []
    
    for py in range(0, max_py + 1, candidate_step):
        for px in range(0, max_px + 1, candidate_step): candidates.append((px, py, tile_energy(px, py)))

    if metric_mode == 'structured': pass  # energies already the priority
    
    elif metric_mode == 'bimodal':
        if candidates:
            energies = np.array([c[2] for c in candidates], dtype=np.float64)
            N = len(energies)
            
            if N <= 1: candidates = [(c[0], c[1], 1.0) for c in candidates]
            else:
                ranks = energies.argsort().argsort().astype(np.float64)
                percentile = ranks / float(N - 1)                 # [0, 1]
                priority = 2.0 * np.abs(percentile - 0.5)          # [0, 1]
                candidates = [(c[0], c[1], float(priority[i])) for i, c in enumerate(candidates)]
    
    elif metric_mode == 'uniform':
        if candidates:
            energies = np.array([c[2] for c in candidates], dtype=np.float64)
            e_max = float(energies.max())
            priority = e_max - energies
            candidates = [(c[0], c[1], float(priority[i])) for i, c in enumerate(candidates)]
    
    else: raise ValueError(f"metric_mode must be structured|bimodal|uniform (got {metric_mode!r})")

    # Highest priority first
    candidates.sort(key=lambda c: -c[2])

    selected = list(base_positions)
    extra_placed = []
    for px, py, _e in candidates:
        if all(max(abs(px - sx), abs(py - sy)) >= min_dist_extra for sx, sy in selected):
            selected.append((px, py))
            extra_placed.append((px, py))
            if len(extra_placed) >= extra_tiles: break

    if return_meta: return {'positions': selected, 'backbone_count': len(base_positions), 'energy_map': energy, 'sat': sat,}
    
    return selected


def create_weight_map(patch_size, overlap):
    w = np.ones(patch_size, dtype=np.float32)
    
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    
    return w[:, None] * w[None, :]


def create_cosine_taper_weight(patch_size, taper):
    if taper <= 0: return np.ones((patch_size, patch_size), dtype=np.float32)
    
    taper = min(int(taper), patch_size // 2)
    w = np.ones(patch_size, dtype=np.float32)
    t = np.arange(taper, dtype=np.float32)
    
    ramp = 0.5 * (1.0 - np.cos(np.pi * (t + 0.5) / float(taper)))
    w[:taper] = ramp
    w[-taper:] = ramp[::-1]
    
    return (w[:, None] * w[None, :]).astype(np.float32)


def stitch_with_blending(patches_dict, positions, stride, patch_size=TILE_SIZE, image_h=None, image_w=None, backbone_count=None, fard_taper=None):
    overlap = patch_size - stride
    wm_backbone = create_weight_map(patch_size, overlap)

    if fard_taper is None: fard_taper = patch_size // 2  # full cosine by default
    wm_fard = (create_cosine_taper_weight(patch_size, fard_taper) if backbone_count is not None else None)

    canvas = np.zeros((image_h, image_w, 3), dtype=np.float32)
    weight = np.zeros((image_h, image_w), dtype=np.float32)
    
    for idx, (px, py) in enumerate(positions):
        if idx not in patches_dict: continue
        
        if backbone_count is not None and idx >= backbone_count: wm = wm_fard
        else: wm = wm_backbone
        
        patch = patches_dict[idx].astype(np.float32)
        canvas[py:py + patch_size, px:px + patch_size] += patch * wm[:, :, None]
        weight[py:py + patch_size, px:px + patch_size] += wm
    
    mask = weight > 0
    out = np.zeros_like(canvas, dtype=np.uint8)
    out[mask] = (canvas[mask] / weight[mask, None]).clip(0, 255).astype(np.uint8)
    out[~mask] = 255
    
    return out


def process_image_per_tile(image_np, vfm, model, device, opt, tile_stride, tile_batch_size, he_transform, skip_background=False, tissue_threshold=0.1, unify_background=False, rng=None,
                           spectral_T=None, fard_extra=9, overlap_mode='max', overlap_K_min=5, overlap_q_low=25.0, overlap_q_high=75.0, overlap_K_max=16, overlap_random_K=8,
                           fard_taper=None, fard_metric_mode='structured'):

    H, W = image_np.shape[:2]
    energy_map = None
    sat = None
    backbone_count = None

    meta = generate_spectrally_adaptive_positions(image_np, tile_size=TILE_SIZE, base_stride=tile_stride, extra_tiles=fard_extra, return_meta=True, metric_mode=fard_metric_mode,)
    positions = meta['positions']
    backbone_count = meta['backbone_count']

    energy_map = meta['energy_map']
    sat = meta['sat']

    tissue_idx = []
    bg_idx = []

    for tile_idx, (px, py) in enumerate(positions):
        tile_np = image_np[py:py + TILE_SIZE, px:px + TILE_SIZE]
        
        if skip_background and compute_tissue_ratio(tile_np) < tissue_threshold: bg_idx.append(tile_idx)
        else: tissue_idx.append(tile_idx)

    tissue_positions = [positions[i] for i in tissue_idx]

    per_tile_K = None  # default = 'max'
    if overlap_mode == 'random':
        per_tile_K = []
        for i in tissue_idx:
            if i >= backbone_count: per_tile_K.append(None)
            else: per_tile_K.append(int(overlap_random_K))
    elif overlap_mode == 'fft':
        if sat is None:
            if energy_map is None: energy_map = _spectral_complexity_map(image_np)
            sat = energy_map.cumsum(axis=0).cumsum(axis=1)
 
        all_energies = [_tile_energy_from_sat(sat, px, py, TILE_SIZE) for (px, py) in positions]
        q_low = float(np.percentile(all_energies, overlap_q_low))
        q_high = float(np.percentile(all_energies, overlap_q_high))
        per_tile_K = []

        for i in tissue_idx:
            if i >= backbone_count: per_tile_K.append(None)
            else:
                K = _determine_overlap_count(all_energies[i], q_low, q_high, K_min=overlap_K_min, K_max=overlap_K_max)
                per_tile_K.append(K)

    tile_conds = compute_tile_conditioning_batched(image_np, tissue_positions, vfm, device,
                                                   vfm_batch_size=VFM_BATCH_SIZE, rng=rng,
                                                   spectral_T=spectral_T, per_tile_K=per_tile_K)

    current_patches = {}

    for batch_start in range(0, len(tissue_idx), tile_batch_size):
        batch_end = min(batch_start + tile_batch_size, len(tissue_idx))
        batch_tile_indices = tissue_idx[batch_start:batch_end]
        B = len(batch_tile_indices)

        batch_he = []
        batch_cond = []
        batch_cls = []

        for i, tile_idx in enumerate(batch_tile_indices):
            px, py = positions[tile_idx]
            tile_np = image_np[py:py + TILE_SIZE, px:px + TILE_SIZE]
            batch_he.append(he_transform(Image.fromarray(tile_np)))
            c = tile_conds[batch_start + i]
            batch_cond.append(c['spatial_map'])
            batch_cls.append(c['neighborhood_cls'])

        data = {'A': torch.stack(batch_he),
                'B': torch.stack(batch_he),
                'sheaf_cond': torch.stack(batch_cond),
                'global_cls': torch.stack(batch_cls),
                'has_adj': torch.zeros(B, dtype=torch.long),
                'has_adj2': torch.zeros(B, dtype=torch.long),
                'A_paths': 'per_tile',}

        model.set_input(data, data)
        model.test()
        
        fake = getattr(model, f'fake_{opt.num_timesteps}', getattr(model, 'fake_B', None))
        for i in range(B): current_patches[batch_tile_indices[i]] = util.tensor2im(fake[i:i + 1])

    if bg_idx and current_patches:
        bg_pixels = []
        
        for fake_np in current_patches.values():
            bm = compute_background_mask(fake_np)
            if bm.any(): bg_pixels.append(fake_np[bm].astype(np.float64))
        
        if bg_pixels:
            all_bg = np.concatenate(bg_pixels, axis=0)
            target_mean = all_bg.mean(axis=0)
            target_std = all_bg.std(axis=0) + 1e-6
        
        else:
            all_gen = np.stack(list(current_patches.values()))
            target_mean = all_gen.reshape(-1, 3).astype(np.float64).mean(axis=0)
            target_std = all_gen.reshape(-1, 3).astype(np.float64).std(axis=0) + 1e-6
        
        for tile_idx in bg_idx:
            px, py = positions[tile_idx]
            he_patch = image_np[py:py + TILE_SIZE, px:px + TILE_SIZE]
            current_patches[tile_idx] = background_tone_match(he_patch, target_mean, target_std)
    
    elif bg_idx and not current_patches:
        for tile_idx in bg_idx:
            px, py = positions[tile_idx]
            current_patches[tile_idx] = image_np[py:py + TILE_SIZE, px:px + TILE_SIZE]

    stitched = stitch_with_blending(current_patches, positions, tile_stride, patch_size=TILE_SIZE, image_h=H, image_w=W, backbone_count=backbone_count, fard_taper=fard_taper)

    if unify_background:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, tissue_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_mask_full = tissue_binary >= 128

        if bg_mask_full.sum() / bg_mask_full.size > 0.01:
            tissue_mask = (tissue_binary < 128).astype(np.float32)
            sigma = TILE_SIZE // 4
            ksize = int(sigma * 6) | 1
            alpha = cv2.GaussianBlur(tissue_mask, (ksize, ksize), sigma)
            bg_mask_stitched = compute_background_mask(stitched)

            if bg_mask_stitched.any():
                ihc_bg = stitched[bg_mask_stitched].astype(np.float64)
                tmu = ihc_bg.mean(axis=0)
                tsu = ihc_bg.std(axis=0) + 1e-6

            else:
                tmu = stitched.reshape(-1, 3).astype(np.float64).mean(axis=0)
                tsu = stitched.reshape(-1, 3).astype(np.float64).std(axis=0) + 1e-6

            he_bg = image_np[bg_mask_full].astype(np.float64)
            sm = he_bg.mean(axis=0)
            ss = he_bg.std(axis=0) + 1e-6
            bg_canvas = ((image_np.astype(np.float64) - sm) / ss * tsu + tmu).clip(0, 255).astype(np.uint8)

            a3 = alpha[..., None]
            blended = (a3 * stitched.astype(np.float64) + (1 - a3) * bg_canvas.astype(np.float64))

            return blended.clip(0, 255).astype(np.uint8)

    return stitched


def initialize_model_from_first_tile(model, image_np, vfm, device, he_transform, opt, spectral_T=None):
    first_tile = image_np[:TILE_SIZE, :TILE_SIZE]
    conds = compute_tile_conditioning_batched(image_np, [(0, 0)], vfm, device, spectral_T=spectral_T)

    init_cond = conds[0]['spatial_map'].unsqueeze(0)
    init_cls = conds[0]['neighborhood_cls'].unsqueeze(0)
    tile_he = he_transform(Image.fromarray(first_tile)).unsqueeze(0)
    init_data = {'A': tile_he, 'B': tile_he,
                 'sheaf_cond': init_cond,
                 'global_cls': init_cls,
                 'has_adj': torch.zeros(1, dtype=torch.long),
                 'has_adj2': torch.zeros(1, dtype=torch.long),
                 'A_paths': 'init',}

    model.data_dependent_initialize(init_data, init_data)
    model.setup(opt)

    model.parallelize()
    model.eval()


# Main
if __name__ == '__main__':
    opt = TestOptions().parse()

    num_workers = getattr(opt, 'num_workers', 1)
    worker_id = getattr(opt, 'worker_id', 0)
    worker_tag = f"[W{worker_id}]" if num_workers > 1 else "[NEW]"

    tile_stride = getattr(opt, 'test_stride', 192)
    tile_batch_size = int(getattr(opt, 'tile_batch_size', 16))
    tissue_threshold = getattr(opt, 'tissue_threshold', 0.1)
    skip_background = getattr(opt, 'skip_background', False)
    unify_background = bool(getattr(opt, 'unify_background', False))

    _sT = getattr(opt, 'sheaf_spectral_t', None)
    spectral_T = None if _sT in (None, '') else float(_sT)

    use_fard = bool(getattr(opt, 'use_fard', True))
    fard_extra = int(getattr(opt, 'fard_extra', 9)) if use_fard else 0
    fard_metric_mode = str(getattr(opt, 'fard_metric_mode', 'bimodal')).strip().lower()
    
    if fard_metric_mode not in ('structured', 'bimodal', 'uniform'): raise ValueError(f"fard_metric_mode must be structured|bimodal|uniform, got {fard_metric_mode!r}")
    
    _ft = getattr(opt, 'fard_taper', None)
    fard_taper = None if _ft in (None, '') else int(_ft)

    overlap_mode = str(getattr(opt, 'overlap_mode', 'max')).strip().lower()
    
    if overlap_mode not in ('max', 'fft', 'random'): raise ValueError(f"OVERLAP_MODE must be max|fft|random, got {overlap_mode!r}")
    
    overlap_K_min = int(getattr(opt, 'overlap_k_min', 5))
    overlap_K_max = int(getattr(opt, 'overlap_k_max', 16))
    overlap_random_K = int(getattr(opt, 'overlap_random_k', 8))
    overlap_q_low = float(getattr(opt, 'overlap_q_low', 25))
    overlap_q_high = float(getattr(opt, 'overlap_q_high', 75))

    print(f"  spectral_T: {spectral_T if spectral_T is not None else 'uniform (off)'}")
    print(f"  FARD: {'on' if fard_extra > 0 else 'off (backbone-only)'}" + (f" (extra={fard_extra}, metric={fard_metric_mode})" if fard_extra > 0 else ""))
    print(f"  FARD stitch taper: " + ("default (TILE_SIZE//2)" if fard_taper is None else f"{fard_taper} px"))
    print(f"  overlap mode={overlap_mode}" + (f" K_min={overlap_K_min} K_max={overlap_K_max} q=[{overlap_q_low},{overlap_q_high}]" if overlap_mode == 'fft' else f" K_random={overlap_random_K}" if overlap_mode == 'random' else ""))

    dataset_mode = True

    opt.dataset_mode = 'sheaf_test'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.phase = 'test'

    device = f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu'

    he_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    if dataset_mode:
        dataset = create_dataset(opt)
        test_dataset = dataset.dataset
        all_indices = list(range(test_dataset.num_images))
        my_indices = all_indices[worker_id::num_workers]

        results_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}_new')
        stitched_dir = os.path.join(results_dir, 'stitched')
        _override_stitched = str(getattr(opt, 'override_stitched_dir', None) or '').strip() or None

        if _override_stitched:
            stitched_dir = _override_stitched
            print(f"  (override) 1024 stitched → {stitched_dir}")

        os.makedirs(stitched_dir, exist_ok=True)

        print(f"{worker_tag} Dataset mode: {len(my_indices)}/{len(all_indices)} images")
        print(f"  Stride: {tile_stride}, TileBatch: {tile_batch_size}")

        vfm = load_vfm(getattr(opt, 'vfm_name', 'gigapath'), opt.vfm_model_path, device)

        model = create_model(opt)
        initialized = False

        for img_idx in tqdm(my_indices, desc=f"{worker_tag} images"):
            he_img = Image.open(test_dataset.A_paths[img_idx]).convert('RGB')
            image_np = np.array(he_img)
            img_name = os.path.splitext(os.path.basename(test_dataset.A_paths[img_idx]))[0]

            stitched_path = os.path.join(stitched_dir, f'{img_name}.png')
            stitched_done = os.path.exists(stitched_path)

            if stitched_done: continue
            image_padded, pad_info = pad_to_aligned(image_np)

            if not initialized:
                initialize_model_from_first_tile(model, image_padded, vfm, device, he_transform, opt, spectral_T=spectral_T)
                initialized = True

            if not stitched_done:
                result_padded = process_image_per_tile(image_padded, vfm, model, device, opt, tile_stride, tile_batch_size, he_transform,
                                                       skip_background=skip_background, tissue_threshold=tissue_threshold, unify_background=unify_background,
                                                       spectral_T=spectral_T, fard_extra=fard_extra, overlap_mode=overlap_mode, 
                                                       overlap_K_min=overlap_K_min, overlap_K_max=overlap_K_max, overlap_random_K=overlap_random_K,
                                                       overlap_q_low=overlap_q_low, overlap_q_high=overlap_q_high,
                                                       fard_taper=fard_taper, fard_metric_mode=fard_metric_mode)

                result = crop_to_original(result_padded, pad_info)
                Image.fromarray(result).save(stitched_path)

        del vfm, model
        torch.cuda.empty_cache()

        print(f"{worker_tag} Complete. {len(my_indices)} images → {stitched_dir}")
