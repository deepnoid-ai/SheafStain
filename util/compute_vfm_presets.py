"""
Precompute Spatial Sheaf Conditioning Presets — multi-VFM dispatcher.

Supports three pathology Vision Foundation Models (VFMs) under a single
unified pipeline:

  - Prov-GigaPath  (ViT-G/14, embed_dim=1536, native 14x14 patch grid)
  - UNI2-h         (ViT-H/14, embed_dim=1536, native 16x16 patch grid)
  - Virchow2       (ViT-H/14, embed_dim=1280, native 16x16 patch grid,
                    with 4 register tokens in addition to the CLS token)

Output format:
    {output_dir}/spatial_preset_{id:03d}.pt
    {
        'preset_id': int,
        'version': 'spatial_v1',
        'vfm': 'gigapath' | 'uni2h' | 'virchow2',
        'vfm_embed_dim': int,                     # 1536 or 1280
        'num_images': int,
        'num_entries': int,
        'cache': {
            (img_idx, grid_idx): {
                'spatial_map':       tensor [D, 16, 16] fp16,
                'coverage_mask':     tensor [16, 16]    bool,
                'ref_position':      (ref_x, ref_y),
                'neighborhood_cls':  tensor [D]         fp32,
            }
        },
        'global_cls': {},   # Empty attributes; dump 
    }

Usage:
    # GigaPath
    python util/compute_vfm_presets.py \\
        --vfm_name gigapath \\
        --vfm_model_path /path/to/prov-gigapath \\
        --preset_start 0 --preset_end 3 --gpu 0 \\
        --dataroot ... --output_dir .../sheaf_spatial_presets_v1

    # UNI2-h (drop-in: same embed_dim 1536, separate output dir)
    python util/compute_vfm_presets.py \\
        --vfm_name uni2h \\
        --vfm_model_path /path/to/UNI2-h \\
        --preset_start 0 --preset_end 3 --gpu 0 \\
        --dataroot ... --output_dir .../sheaf_spatial_presets_v1_uni2h

    # Virchow2 (embed_dim 1280 -- requires generator with matching cond dim)
    python util/compute_vfm_presets.py \\
        --vfm_name virchow2 \\
        --vfm_model_path /path/to/Virchow2 \\
        --preset_start 0 --preset_end 3 --gpu 0 \\
        --dataroot ... --output_dir .../sheaf_spatial_presets_v1_virchow2
"""

import os
import sys
import json
import timm
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass
from torchvision import transforms
from types import SimpleNamespace


# Add project root to path. Must be at position 0 (before the script's own
# directory `util/`), otherwise Python picks up `util/util.py` first and the
# `from util.vfm_dispatch import ...` below fails with "util is not a package".
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root in sys.path: sys.path.remove(project_root)
sys.path.insert(0, project_root)

from data.sheaf_dataset import (SheafDataset, GRID_SIZE, REF_SIZE, OVL_SIZE, NUM_GRIDS, GRID_ORIGINS,)
from models.sheaf_modules import (load_gigapath, build_spatial_conditioning_map,)
from util.vfm_dispatch import (TARGET_TOKEN_GRID, TARGET_NUM_TOKENS,
                               VFMSpec, VFMWrapper, load_vfm, _VFM_BUILDERS,
                               _try_load_local_config, _load_state_dict_strict_or_warn,
                               _build_gigapath, _build_uni, _build_uni2h, _build_virchow2,)


# ============================================================================
# VFM dispatcher
# ============================================================================
# A single VFM forward pass must produce BOTH a CLS token and a 14x14 patch
# token grid (matching the original GigaPath geometry that downstream code
# is built around). The wrapper standardizes this regardless of the
# underlying VFM's native grid resolution or prefix-token layout.
#
# Notes on the three supported VFMs (best-effort architecture defaults; we
# also try to load `config.json` first if the user has one alongside
# `pytorch_model.bin`):
#
#   - Prov-GigaPath: timm `vit_giant_patch14_dinov2`. The official config.json
#     overrides `patch_size=16`, giving a native 14x14 token grid at 224
#     input. embed_dim=1536, depth=40, num_heads=24. Single CLS token.
#   - UNI2-h: ViT-H/14 (DINOv2). embed_dim=1536, depth=24, num_heads=24.
#     Single CLS token. Native 16x16 token grid at 224 input -> resampled
#     to 14x14 in the wrapper for downstream compatibility.
#   - Virchow2: ViT-H/14 with SwiGLU FFN and 4 register tokens.
#     embed_dim=1280, depth=32, num_heads=16. Native 16x16 token grid.
#     Output layout: [B, 1+4+256, 1280] = [CLS; register x 4; patches].
# ============================================================================

# VFM dispatcher (VFMSpec / VFMWrapper / load_vfm / _build_*) lives in
# `util/vfm_dispatch.py` and is imported above. Keep this script as the
# preset-computation entry point only.


# ============================================================================
# Preset computation (multi-VFM compatible)
# ============================================================================
def extract_patches_all_grids(dataset, preset_id, img_idx):
    cache_refresh_freq = dataset.cache_refresh_freq
    eff_epoch = (preset_id // cache_refresh_freq) * cache_refresh_freq

    he_img = Image.open(dataset.A_paths[img_idx]).convert('RGB')

    for grid_idx in range(NUM_GRIDS):
        rng = random.Random(img_idx * 1000 + grid_idx * 100 + eff_epoch * 7)

        gx, gy = GRID_ORIGINS[grid_idx]
        grid_he = he_img.crop((gx, gy, gx + GRID_SIZE, gy + GRID_SIZE))

        ref_x, ref_y = dataset._select_reference_from_grid(grid_he, rng)
        ovl_result = dataset._extract_overlapped_from_grid(grid_he, ref_x, ref_y, rng)

        yield (grid_idx, ovl_result['patches'], ovl_result['metadata'], (ref_x, ref_y))


def compute_global_cls(vfm: VFMWrapper, image_path, device, transform_fn, tile_size=256, num_tiles=4):
    """Compute global CLS token from `num_tiles**2` uniform tiles.

    not used in current setup (whole-image global CLS is replaced by per-grid
    neighborhood_cls). Kept here for backward compatibility; updated to
    use the unified VFM wrapper so it continues to work if re-enabled.
    """
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    D = vfm.spec.embed_dim
    in_size = vfm.spec.img_size
    tiles = []

    for row in range(num_tiles):
        for col in range(num_tiles):
            x = col * tile_size
            y = row * tile_size

            if x + tile_size <= W and y + tile_size <= H:
                tile = img.crop((x, y, x + tile_size, y + tile_size))
                tile_in = tile.resize((in_size, in_size), Image.BICUBIC)
                tiles.append(transform_fn(tile_in))

    if not tiles: return torch.zeros(D)

    batch = torch.stack(tiles).to(device)
    cls_tokens, _ = vfm.extract(batch)               # [n_tiles, D]
    global_cls = cls_tokens.mean(dim=0)              # [D]
    
    return global_cls.cpu()


def compute_spatial_preset(dataset, preset_id, vfm: VFMWrapper, device, output_path, batch_size=64):
    """Compute preset: spatial_map + neighborhood_cls per (img_idx, grid_idx).

    Per grid, overlap patches around Ref are encoded by the VFM in a single
    forward pass. Patch tokens (resampled to 14x14 in the wrapper) feed the
    spatial map; CLS tokens are mean-pooled into the neighborhood CLS.
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    to_tensor = transforms.ToTensor()

    # Transform for global CLS tiles (dead code; kept in case re-enabled)
    cls_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

    D = vfm.spec.embed_dim                                     # was hardcoded 1536
    REF_TOKEN_GRID = REF_SIZE // 16                            # 256/16 = 16

    cache = {}
    global_cls_dict = {}
    n_images = dataset.num_images

    for img_idx in tqdm(range(n_images), desc=f"Preset {preset_id}"):

        for grid_idx, patches, metadata, ref_position in extract_patches_all_grids(dataset, preset_id, img_idx):
            ref_x, ref_y = ref_position

            if len(patches) == 0:
                # Empty open cover -> trivial zero summary section.
                cache[(img_idx, grid_idx)] = {'spatial_map':   torch.zeros(D, REF_TOKEN_GRID, REF_TOKEN_GRID, dtype=torch.float16),
                                              'coverage_mask': torch.zeros(REF_TOKEN_GRID, REF_TOKEN_GRID, dtype=torch.bool),
                                              'ref_position':  ref_position,
                                              'neighborhood_cls': torch.zeros(D, dtype=torch.float32),}
                continue

            # PIL -> tensor -> ImageNet normalize. Resize to VFM input size if
            # the patch resolution differs from spec.img_size. (Default
            # OVL_SIZE=224 already matches.)
            patch_tensors = torch.stack([to_tensor(p) for p in patches]).to(device)
            
            if patch_tensors.shape[-1] != vfm.spec.img_size:
                patch_tensors = F.interpolate(patch_tensors, size=(vfm.spec.img_size, vfm.spec.img_size), mode='bicubic', align_corners=False)

            patch_tensors = (patch_tensors - IMAGENET_MEAN[None]) / IMAGENET_STD[None]

            # Single VFM pass produces both the patch-token grid (for the
            # spatial map) and the CLS token (for the neighborhood CLS).
            all_patch_tokens = []
            all_cls_tokens = []

            with torch.no_grad():
                for start in range(0, len(patches), batch_size):
                    batch = patch_tensors[start:start + batch_size]
                    cls_b, patch_b = vfm.extract(batch)

                    # patch_b: [B, 196, D] (already resampled to 14x14)
                    all_patch_tokens.append(patch_b)
                    all_cls_tokens.append(cls_b)

            all_patch_tokens = torch.cat(all_patch_tokens, dim=0)  # [N_patches, 196, D]
            all_cls_tokens   = torch.cat(all_cls_tokens, dim=0)    # [N_patches, D]

            # Neighborhood CLS = mean over open cover {U_i} of overlap patches.
            neighborhood_cls = all_cls_tokens.mean(dim=0).cpu().float()

            # Build spatial conditioning map (D-agnostic).
            token_list = [all_patch_tokens[i].cpu() for i in range(len(patches))]
            spatial_map, coverage_mask = build_spatial_conditioning_map(token_list, metadata, ref_x, ref_y)

            cache[(img_idx, grid_idx)] = {'spatial_map':      spatial_map.half(),
                                          'coverage_mask':    coverage_mask,
                                          'ref_position':     ref_position,
                                          'neighborhood_cls': neighborhood_cls,} # [D] fp32

        # Global CLS (whole-image 16 uniform tiles)
        # Superseded by per-grid neighborhood_cls. Kept as commented-out
        # code so it can be re-enabled without structural changes;
        # `global_cls_dict` is saved as an empty dict to preserve the
        # output schema.
        # global_cls = compute_global_cls(
        #     vfm, dataset.A_paths[img_idx], device, cls_transform)
        # global_cls_dict[img_idx] = global_cls.half()

    # Save
    torch.save({'preset_id':      preset_id,
                'version':        'spatial_v1',
                'vfm':            vfm.spec.name,
                'vfm_embed_dim':  D,
                'num_images':     n_images,
                'num_entries':    len(cache),
                'cache':          cache,
                'global_cls':     global_cls_dict,}, output_path)

    n_neighborhood = sum(1 for v in cache.values() if 'neighborhood_cls' in v)
    size_mb = os.path.getsize(output_path) / 1e6

    print(f"[Preset {preset_id}] Saved to {output_path} "
          f"({size_mb:.1f} MB, {len(cache)} entries, "
          f"{n_neighborhood} neighborhood_cls, "
          f"vfm={vfm.spec.name}, D={D})")


def main():
    parser = argparse.ArgumentParser(
        description='Precompute Spatial Sheaf Presets Full Grid + per-grid Neighborhood CLS; multi-VFM dispatcher)')

    parser.add_argument('--preset_start', type=int, required=True)
    parser.add_argument('--preset_end',   type=int, required=True)

    parser.add_argument('--dataroot',       type=str, required=True)
    parser.add_argument('--output_dir',     type=str, required=True)
    parser.add_argument('--vfm_model_path', type=str, required=True, help='Local directory containing pytorch_model.bin (and optionally config.json) for the VFM.')
    parser.add_argument('--vfm_name',       type=str, default='gigapath', choices=list(_VFM_BUILDERS.keys()), help='Which VFM to use (default: gigapath, matching the legacy script).')

    parser.add_argument('--stain', type=str, default='her2', choices=['her2', 'ki67', 'er', 'pr'])
    parser.add_argument('--img_ext', type=str, default='.png')
    parser.add_argument('--train_split_mode', type=str, default='bci', choices=['bci', 'mist'])
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--sheaf_stride', type=int, default=80) # fixed value (from the previous experiment)
    parser.add_argument('--sheaf_num_jitter_vh', type=int, default=2)
    parser.add_argument('--sheaf_jitter_range', type=int, default=32)
    parser.add_argument('--min_tissue_ratio', type=float, default=0.3)
    parser.add_argument('--sheaf_cache_refresh_freq', type=int, default=5)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    EXPECTED_VERSION = 'spatial_v1'
    EXPECTED_VFM     = args.vfm_name

    for fname in sorted(os.listdir(args.output_dir)):
        if not (fname.startswith('spatial_preset_') and fname.endswith('.pt')): continue
        fpath = os.path.join(args.output_dir, fname)
        
        try: meta = torch.load(fpath, map_location='cpu', weights_only=False)
        except Exception: continue
        
        existing_version = meta.get('version', 'unknown')
        existing_vfm     = meta.get('vfm', 'gigapath') # legacy (dump)
        
        if existing_version != EXPECTED_VERSION:
            raise RuntimeError(f"Refusing to write into '{args.output_dir}': found existing "
                               f"preset '{fname}' with version='{existing_version}', but this "
                               f"script produces version='{EXPECTED_VERSION}'. Use a separate "
                               f"output_dir to avoid mixing schemas.")
        if existing_vfm != EXPECTED_VFM:
            raise RuntimeError(f"Refusing to write into '{args.output_dir}': found existing "
                               f"preset '{fname}' built with VFM='{existing_vfm}', but you "
                               f"requested VFM='{EXPECTED_VFM}'. Use a separate output_dir "
                               f"per VFM (their embed_dim and feature semantics differ).")
        
        break  # one match is enough to confirm consistency

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)

    # Create dataset (same as training: labels.csv, phase='train')
    opt = SimpleNamespace(dataroot=args.dataroot,
                          phase=args.phase, stain=args.stain, img_ext=args.img_ext,
                          train_split_mode=args.train_split_mode,
                          sheaf_stride=args.sheaf_stride, sheaf_num_jitter_vh=args.sheaf_num_jitter_vh, sheaf_jitter_range=args.sheaf_jitter_range,
                          min_tissue_ratio=args.min_tissue_ratio, max_ref_attempts=50,
                          sheaf_cond_dim=256, sheaf_cache_refresh_freq=args.sheaf_cache_refresh_freq, sheaf_spatial=True,
                          return_overlap_patches=False, grid_augment=False, grid_color_jitter=0,
                          use_ddp=False, rank=0, isTrain=True, max_dataset_size=float('inf'),
                          lambda_sheaf=0, lambda_sheaf_feat=0, lambda_sheaf_cocycle=0,)
    
    dataset = SheafDataset(opt)

    # Load VFM (once)
    print(f"Loading VFM '{args.vfm_name}' from {args.vfm_model_path} on GPU {args.gpu}...")
    vfm = load_vfm(args.vfm_name, args.vfm_model_path, device)
    print(f"  spec: {vfm.spec}")

    # Compute presets
    for preset_id in range(args.preset_start, args.preset_end + 1):
        output_path = os.path.join(args.output_dir, f'spatial_preset_{preset_id:03d}.pt')

        if os.path.exists(output_path):
            print(f"[Preset {preset_id}] Already exists, skipping.")
            continue

        compute_spatial_preset(dataset, preset_id, vfm, device, output_path, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
