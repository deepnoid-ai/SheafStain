import os
import glob
import json
import math
import timm
import torch
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms
from data.sheaf_dataset import (NUM_GRIDS, DIR_N, DIR_S, DIR_E, DIR_W, DIR_NE, DIR_NW, DIR_SE, DIR_SW,)


# Conditioning Layer
class SheafConditioningLayer(nn.Module):
    """Sheaf conditioning injection for ResnetBlock_cond.

    Three modes:
        additive: out += proj(cond)
        adaln:    out = γ * out + β  (γ, β from cond)
        gating:   out = out * σ(proj(cond))

    Args:
        channels: feature map channels (e.g., 256 for UNSB bottleneck)
        cond_dim: sheaf conditioning vector dimension
        mode: 'additive' | 'adaln' | 'gating'
    """

    def __init__(self, channels, cond_dim, mode='additive', zero_init=True):
        super().__init__()
        self.mode = mode

        if mode == 'additive':
            self.proj = nn.Linear(cond_dim, channels)
            if zero_init:
                nn.init.zeros_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
            else:
                nn.init.xavier_normal_(self.proj.weight, gain=1.0)
                nn.init.zeros_(self.proj.bias)

        elif mode == 'adaln':
            self.proj = nn.Linear(cond_dim, channels * 2)
            if zero_init:
                nn.init.zeros_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
                with torch.no_grad(): self.proj.bias[:channels] = 1.0  # γ init = 1
            else:
                nn.init.xavier_normal_(self.proj.weight, gain=1.0)
                nn.init.zeros_(self.proj.bias)
                with torch.no_grad(): self.proj.bias[:channels] = 1.0  # γ init = 1
        
        elif mode == 'gating':
            self.proj = nn.Linear(cond_dim, channels)
            if zero_init:
                nn.init.zeros_(self.proj.weight)
                nn.init.constant_(self.proj.bias, 2.0)
            else:
                nn.init.xavier_normal_(self.proj.weight, gain=1.0)
                nn.init.constant_(self.proj.bias, 2.0)
        
        else: raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x, cond):
        """Apply sheaf conditioning to feature map.

        Args:
            x: feature map [B, C, H, W]
            cond: conditioning vector [B, cond_dim]

        Returns:
            conditioned feature map [B, C, H, W]
        """
        if self.mode == 'additive': return x + self.proj(cond)[:, :, None, None]

        elif self.mode == 'adaln':
            params = self.proj(cond)
            gamma, beta = params.chunk(2, dim=1)

            return gamma[:, :, None, None] * x + beta[:, :, None, None]

        elif self.mode == 'gating':
            gate = torch.sigmoid(self.proj(cond))

            return x * gate[:, :, None, None]


# Spatial Conditioning Layer (Fiber Bundle)
VFM_DIM = 1536       # VFM feature dimension (e.g., Prov-GigaPath is 1536) 
TOKEN_PX = 16        # VFM token pixel size
REF_TOKEN_GRID = 16  # 256 / 16 = 16 (ref patch in token space)
OVL_TOKEN_GRID = 14  # 224 / 16 = 14 (overlapped patch in token space)

def build_spatial_conditioning_map(patch_tokens_list, metadata_list, ref_x, ref_y, ref_size=256, ovl_size=224, token_px=TOKEN_PX):
    """Map overlapped patches' VFM tokens to ref patch spatial grid.

    Implements the fiber bundle structure: each position in the ref patch
    receives a conditioning vector derived from overlapping VFM observations.
    Multiple directions' tokens at the same ref position are averaged
    (implicit denoising of context contamination).

    Args:
        patch_tokens_list: list of [196, 1536] tensors (one per overlapped patch)
        metadata_list: list of dicts with 'position': (ovl_x, ovl_y) in grid coords
        ref_x, ref_y: ref patch top-left in grid coordinates

    Returns:
        spatial_map: [VFM_DIM, ref_grid, ref_grid] tensor (CHW format)
        coverage_mask: [ref_grid, ref_grid] bool tensor
    """
    ref_grid = ref_size // token_px  # 16
    ovl_grid = ovl_size // token_px  # 14
    dim = patch_tokens_list[0].shape[-1]  # 1536

    spatial_sum = torch.zeros(ref_grid, ref_grid, dim)
    spatial_count = torch.zeros(ref_grid, ref_grid)

    for tokens, meta in zip(patch_tokens_list, metadata_list):
        ovl_x, ovl_y = meta['position']

        # Compute token centers for this overlapped patch
        for row in range(ovl_grid):
            for col in range(ovl_grid):
                # Token center in grid pixel coordinates
                tx = ovl_x + col * token_px + token_px // 2
                ty = ovl_y + row * token_px + token_px // 2

                # Check if within ref patch
                if (ref_x <= tx < ref_x + ref_size and ref_y <= ty < ref_y + ref_size):
                    ref_col = (tx - ref_x) // token_px
                    ref_row = (ty - ref_y) // token_px

                    token_idx = row * ovl_grid + col
                    token_val = tokens[token_idx]

                    if token_val.is_cuda: token_val = token_val.detach().cpu()

                    spatial_sum[ref_row, ref_col] += token_val
                    spatial_count[ref_row, ref_col] += 1

    # Average tokens at each position
    coverage_mask = spatial_count > 0
    spatial_map = torch.zeros_like(spatial_sum)
    spatial_map[coverage_mask] = (spatial_sum[coverage_mask] / spatial_count[coverage_mask].unsqueeze(-1))

    # Convert to CHW format: [ref_grid, ref_grid, dim] → [dim, ref_grid, ref_grid]
    spatial_map = spatial_map.permute(2, 0, 1)

    return spatial_map, coverage_mask


class SpatialSheafConditioningLayer(nn.Module):
    """Spatially-varying sheaf conditioning (fiber bundle implementation).

    Unlike SheafConditioningLayer which broadcasts a single vector to all positions,
    this layer applies position-dependent conditioning from a spatial map.

    Each spatial position receives its own conditioning vector (stalk), implementing
    a non-trivial fiber bundle over the image manifold.

    Args:
        channels: feature map channels (e.g., 256 for bottleneck)
        cond_channels: conditioning map channels (1536 for VFM)
        mode: 'additive' | 'adaln' | 'gating'
        zero_init: whether to zero-initialize projection weights
    """

    def __init__(self, channels, cond_channels=VFM_DIM, mode='additive', zero_init=True):
        super().__init__()
        self.mode = mode

        if mode == 'additive':
            self.proj = nn.Conv2d(cond_channels, channels, kernel_size=1)
            if zero_init:
                nn.init.zeros_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
            else:
                nn.init.xavier_normal_(self.proj.weight, gain=1.0)
                nn.init.zeros_(self.proj.bias)
        
        elif mode == 'adaln':
            self.proj = nn.Conv2d(cond_channels, channels * 2, kernel_size=1)
            if zero_init:
                nn.init.zeros_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
                with torch.no_grad(): self.proj.bias[:channels] = 1.0
            else:
                nn.init.xavier_normal_(self.proj.weight, gain=1.0)
                nn.init.zeros_(self.proj.bias)
                with torch.no_grad(): self.proj.bias[:channels] = 1.0
        
        elif mode == 'gating':
            self.proj = nn.Conv2d(cond_channels, channels, kernel_size=1)
            if zero_init:
                nn.init.zeros_(self.proj.weight)
                nn.init.constant_(self.proj.bias, 2.0)
            else:
                nn.init.xavier_normal_(self.proj.weight, gain=1.0)
                nn.init.constant_(self.proj.bias, 2.0)
        
        else: raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x, spatial_cond):
        """Apply spatially-varying sheaf conditioning.

        Args:
            x: feature map [B, C, H, W] (e.g., [B, 256, 64, 64])
            spatial_cond: conditioning map [B, C_cond, H_map, W_map]
                          (e.g., [B, 1536, 16, 16])

        Returns:
            conditioned feature map [B, C, H, W]
        """
        # Project channels: [B, C_cond, H_map, W_map] → [B, C, H_map, W_map]
        cond = self.proj(spatial_cond)

        # Upsample to match feature map resolution
        if cond.shape[2:] != x.shape[2:]: cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)

        if self.mode == 'additive': return x + cond

        elif self.mode == 'adaln':
            channels = x.shape[1]
            gamma = cond[:, :channels]
            beta = cond[:, channels:]

            return gamma * x + beta

        elif self.mode == 'gating': return x * torch.sigmoid(cond)


# Aggregator

class SheafAggregator(nn.Module):
    """Aggregate VFM patch tokens into a conditioning vector.

    Two modes:
        agnostic:  mean pool all tokens → project
        direction: per-direction pool → concat → project
                   Groups: cardinal (N,S,E,W) and diagonal (NE,NW,SE,SW)
                   → 2 groups × vfm_dim → project

    Args:
        vfm_dim: VFM feature dimension (1536 for GigaPath)
        cond_dim: output conditioning dimension
        mode: 'agnostic' | 'direction'
    """

    # Group directions: cardinal vs diagonal (for direction-aware pooling)
    CARDINAL_DIRS = {DIR_N, DIR_S, DIR_E, DIR_W}
    DIAGONAL_DIRS = {DIR_NE, DIR_NW, DIR_SE, DIR_SW}
    NUM_GROUPS = 2  # cardinal, diagonal

    def __init__(self, vfm_dim=1536, cond_dim=256, mode='agnostic'):
        super().__init__()
        self.mode = mode
        self.vfm_dim = vfm_dim

        if mode == 'agnostic': self.proj = nn.Linear(vfm_dim, cond_dim)
        elif mode == 'direction': self.proj = nn.Linear(vfm_dim * self.NUM_GROUPS, cond_dim)
        else: raise ValueError(f"Unknown aggregation mode: {mode}")

    def forward(self, patch_tokens_list, direction_ids_list):
        """Aggregate patch tokens into conditioning vector.

        Args:
            patch_tokens_list: list of [N_tokens, vfm_dim] tensors
                               (one per valid overlapped patch)
            direction_ids_list: list of direction IDs (int) for each patch

        Returns:
            conditioning vector [cond_dim]
        """
        if len(patch_tokens_list) == 0:
            device = self.proj.weight.device

            return torch.zeros(self.proj.out_features, device=device)

        if self.mode == 'agnostic':
            all_tokens = torch.cat(patch_tokens_list, dim=0)
            pooled = all_tokens.mean(dim=0)

            return self.proj(pooled)

        elif self.mode == 'direction':
            # Pool by group: cardinal vs diagonal
            groups = [self.CARDINAL_DIRS, self.DIAGONAL_DIRS]
            group_pools = []

            for group_set in groups:
                group_tokens = [t for t, d in zip(patch_tokens_list, direction_ids_list) if d in group_set]
                if group_tokens: pooled = torch.cat(group_tokens, dim=0).mean(dim=0)
                else: pooled = torch.zeros(self.vfm_dim, device=self.proj.weight.device)
                
                group_pools.append(pooled)

            concat = torch.cat(group_pools, dim=0)  # [2 * vfm_dim]
            
            return self.proj(concat)


# VFM Loader
def load_gigapath(model_path=None, device='cuda'):
    assert model_path is not None, 'load_gigapath: model_path is required (set vfm_model_path in config.yaml).'
    """Load GigaPath VFM model.

    Args:
        model_path: path to prov-gigapath directory
        device: target device

    Returns:
        model: GigaPath model in eval mode
    """
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f: config = json.load(f)

    model_args = config['model_args']
    model = timm.create_model("vit_giant_patch14_dinov2",
                              pretrained=False,
                              num_classes=0,
                              img_size=model_args['img_size'],
                              patch_size=model_args['patch_size'],
                              embed_dim=model_args['embed_dim'],
                              depth=model_args['depth'],
                              num_heads=model_args['num_heads'],
                              mlp_ratio=model_args['mlp_ratio'],
                              init_values=model_args['init_values'],)

    state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu', weights_only=True,)
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    
    return model


# LRM (ViT d256 Restriction Map)
class ViTRestrictionMap(nn.Module):
    """Lightweight ViT restriction map (frozen, pre-trained).

    Architecture: ρ(x) = x + up_proj(ViT(down_proj(x) + pos_embed))
    From exp3_2 ablation: ViT d256 (cos=0.908, cocycle=0.088, stability=1.001)
    """

    def __init__(self, dim=1536, n_tokens=126, d_inner=256, n_layers=2, n_heads=4):
        super().__init__()

        self.dim = dim
        self.n_tokens = n_tokens
        self.d_inner = d_inner

        self.down_proj = nn.Linear(dim, d_inner)
        self.up_proj = nn.Linear(d_inner, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_inner) * 0.02)

        layer = nn.TransformerEncoderLayer(d_model=d_inner, nhead=n_heads, dim_feedforward=d_inner * 4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True,)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def _residual(self, x):
        h = self.down_proj(x) + self.pos_embed
        h = self.transformer(h)
        
        return self.up_proj(h)

    def forward(self, x):
        return x + self._residual(x)

    def interpolate_pos_embed(self, target_n_tokens):
        """Interpolate positional embeddings to handle different token counts.

        Standard ViT technique (Dosovitskiy et al., 2021).

        Args:
            target_n_tokens: desired number of tokens
        """
        if target_n_tokens == self.n_tokens: return

        old_pos = self.pos_embed.data  # [1, n_tokens, d_inner]
        new_pos = F.interpolate(old_pos.permute(0, 2, 1), size=target_n_tokens, mode='linear', align_corners=False,).permute(0, 2, 1) # [1, target_n_tokens, d_inner]

        self.pos_embed = nn.Parameter(new_pos, requires_grad=False)
        self.n_tokens = target_n_tokens


def load_lrm(checkpoint_path, device='cuda', target_n_tokens=None):
    """Load pre-trained LRM (ViT d256) from checkpoint.

    Args:
        checkpoint_path: path to .pt file (e.g., vit_d256_he_seed42.pt)
        device: target device
        target_n_tokens: if set, interpolate pos_embed to this token count.
                         None = keep original (126).

    Returns:
        lrm: ViTRestrictionMap in eval mode, frozen
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    variant = ckpt['variant']
    params = variant['params']

    lrm = ViTRestrictionMap(dim=1536,
                            n_tokens=params.get('n_tokens', 126),
                            d_inner=params['d_inner'],
                            n_layers=params['n_layers'],
                            n_heads=params['n_heads'],)
    lrm.load_state_dict(ckpt['rho_a'])

    if target_n_tokens is not None: lrm.interpolate_pos_embed(target_n_tokens)

    lrm = lrm.to(device).eval()
    for p in lrm.parameters(): p.requires_grad = False

    return lrm


def _extract_overlap_token_indices(ref_x, ref_y, ovl_x, ovl_y, ref_size=256, ovl_size=224, token_px=16):
    """Compute which VFM tokens of the overlapped patch fall within the ref region.

    Args:
        ref_x, ref_y: ref patch top-left in grid coords
        ovl_x, ovl_y: overlapped patch top-left in grid coords

    Returns:
        list of token indices (into the flattened 196-token sequence)
        that overlap with the reference patch.
    """
    token_grid = ovl_size // token_px  # 14

    indices = []
    for row in range(token_grid):
        for col in range(token_grid):
            # Token center in grid coordinates
            tx = ovl_x + col * token_px + token_px // 2
            ty = ovl_y + row * token_px + token_px // 2

            # Check if within ref patch
            if (ref_x <= tx < ref_x + ref_size and ref_y <= ty < ref_y + ref_size): indices.append(row * token_grid + col)

    return indices


# Embedding Cache

# LRM token strategy constants
TOKEN_STRATEGY_OVERLAP_INTERP = 'overlap_interp'     # B: overlap tokens + small interpolation
TOKEN_STRATEGY_FULL_INTERP = 'full_interp'           # A: all 196 tokens + large interpolation
TOKEN_STRATEGY_OVERLAP_SUBSAMPLE = 'overlap_subsample'  # C: overlap tokens + subsampling to 126


class SheafEmbeddingCache:
    """Compute and cache VFM + LRM embeddings for sheaf conditioning.

    Pipeline: overlapped patches → VFM → (LRM correction) → aggregate → cache

    LRM Token Strategies:
        overlap_interp:    Extract overlap tokens (~140), interpolate LRM pos_embed (126→140)
        full_interp:       Use all 196 tokens, interpolate LRM pos_embed (126→196)
        overlap_subsample: Extract overlap tokens, subsample to 126 (no interpolation)

    Usage:
        cache = SheafEmbeddingCache(opt)
        cache.compute_epoch(dataset, epoch, device)
        dataset.set_sheaf_cache(cache.embeddings)
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, opt):
        self.opt = opt
        self.vfm_dim = 1536
        self.cond_dim = getattr(opt, 'sheaf_cond_dim', 256)
        self.agg_mode = getattr(opt, 'sheaf_agg_mode', 'agnostic')
        self.vfm_path = getattr(opt, 'vfm_model_path', None)

        assert self.vfm_path, 'vfm_model_path is not set (configure it in config.yaml or pass --vfm_model_path).'
        
        self.batch_size = getattr(opt, 'sheaf_cache_batch_size', 64)
        self.lrm_checkpoint = getattr(opt, 'lrm_checkpoint', None)
        self.token_strategy = getattr(opt, 'sheaf_token_strategy', TOKEN_STRATEGY_OVERLAP_INTERP)

        # Aggregator
        self.aggregator = SheafAggregator(vfm_dim=self.vfm_dim, cond_dim=self.cond_dim, mode=self.agg_mode,)

        # Cache storage
        self.embeddings = {}           # aggregated: {(img, grid): [cond_dim]}
        self.patch_embeddings = {}     # per-patch: {(img, grid): {embeddings, direction_ids}}
        self.cls_tokens = {}

        # Models (loaded on demand)
        self._vfm = None
        self._lrm = None
        self._lrm_original_n_tokens = 126

    def _ensure_vfm(self, device):
        if self._vfm is None:
            print("[SheafEmbeddingCache] Loading GigaPath VFM...")
            self._vfm = load_gigapath(self.vfm_path, device)
            print("[SheafEmbeddingCache] GigaPath loaded.")

        return self._vfm

    def _ensure_lrm(self, device, n_tokens_override=None):
        """Load LRM if checkpoint is provided."""
        if self.lrm_checkpoint is None: return None
        if self._lrm is None:
            print(f"[SheafEmbeddingCache] Loading LRM from {self.lrm_checkpoint}...")
            self._lrm = load_lrm(self.lrm_checkpoint, device, target_n_tokens=n_tokens_override,)

            print(f"[SheafEmbeddingCache] LRM loaded "
                  f"(strategy={self.token_strategy}, "
                  f"n_tokens={self._lrm.n_tokens}).")
        
        return self._lrm

    def _release_models(self):
        
        if self._vfm is not None:
            del self._vfm
            self._vfm = None
        
        if self._lrm is not None:
            del self._lrm
            self._lrm = None
        
        torch.cuda.empty_cache()

    def _get_lrm_with_pos_embed(self, lrm, n_tokens):
        """Get LRM pos_embed matching the required token count.

        Dynamically interpolates pos_embed if needed. Caches results.
        """
        if n_tokens == lrm.n_tokens: return lrm.pos_embed
 
        if not hasattr(self, '_pos_embed_cache'): self._pos_embed_cache = {}

        if n_tokens not in self._pos_embed_cache:
            orig_pos = lrm.pos_embed.data  # [1, current_n, d_inner]
            new_pos = F.interpolate(orig_pos.permute(0, 2, 1), size=n_tokens, mode='linear', align_corners=False,).permute(0, 2, 1)
            self._pos_embed_cache[n_tokens] = new_pos

        return self._pos_embed_cache[n_tokens]

    def _lrm_forward_dynamic(self, lrm, x):
        """Run LRM forward with dynamic pos_embed interpolation.

        Args:
            lrm: ViTRestrictionMap
            x: [1, K, 1536] where K may differ from lrm.n_tokens
        """
        n_tokens = x.shape[1]
        pos_embed = self._get_lrm_with_pos_embed(lrm, n_tokens)
        h = lrm.down_proj(x) + pos_embed
        h = lrm.transformer(h)

        return x + lrm.up_proj(h)

    def _apply_lrm(self, patch_tokens, lrm, metadata, ref_position):
        """Apply LRM to patch tokens according to the selected strategy.

        Args:
            patch_tokens: [N_ovl, 196, 1536]
            lrm: ViTRestrictionMap (frozen)
            metadata: list of dicts with 'position' keys
            ref_position: (ref_x, ref_y) in grid coords

        Returns:
            list of [n_tokens, 1536] tensors (one per patch, LRM-corrected)
        """
        ref_x, ref_y = ref_position
        corrected = []

        for i in range(patch_tokens.shape[0]):
            tokens = patch_tokens[i]  # [196, 1536]
            ovl_x, ovl_y = metadata[i]['position']

            if self.token_strategy == TOKEN_STRATEGY_FULL_INTERP:
                # Strategy A: use all 196 tokens, dynamic pos_embed
                out = self._lrm_forward_dynamic(lrm, tokens.unsqueeze(0)).squeeze(0)
                corrected.append(out)

            elif self.token_strategy == TOKEN_STRATEGY_OVERLAP_INTERP:
                # Strategy B: extract overlap tokens, dynamic pos_embed
                indices = _extract_overlap_token_indices(ref_x, ref_y, ovl_x, ovl_y)

                if len(indices) == 0:
                    corrected.append(tokens)
                    continue

                overlap_tokens = tokens[indices]  # [K, 1536]
                out = self._lrm_forward_dynamic(lrm, overlap_tokens.unsqueeze(0)).squeeze(0)
                corrected.append(out)

            elif self.token_strategy == TOKEN_STRATEGY_OVERLAP_SUBSAMPLE:
                # Strategy C: extract overlap, subsample to original 126
                indices = _extract_overlap_token_indices(ref_x, ref_y, ovl_x, ovl_y)
                
                if len(indices) == 0:
                    corrected.append(tokens)
                    continue
                
                overlap_tokens = tokens[indices]  # [K, 1536]
                target = self._lrm_original_n_tokens
                
                if len(indices) > target:
                    step = len(indices) / target
                    sub_idx = [int(j * step) for j in range(target)]
                    overlap_tokens = overlap_tokens[sub_idx]
                
                elif len(indices) < target:
                    pad = torch.zeros(target - len(indices), self.vfm_dim, device=overlap_tokens.device)
                    overlap_tokens = torch.cat([overlap_tokens, pad], dim=0)

                out = lrm(overlap_tokens.unsqueeze(0)).squeeze(0)
                corrected.append(out)

        return corrected

    @torch.no_grad()
    def compute_epoch(self, dataset, epoch, device='cuda', keep_models=False, show_progress=True):
        """Compute sheaf embeddings for all images/grids.

        Pipeline: patches → VFM → LRM(if available) → aggregate → cache
        """
        if getattr(self, '_is_spatial', False):
            raise RuntimeError("compute_epoch() is for on-the-fly vector mode only. "
                               "Cannot mix with spatial preset mode. "
                               "Create a new SheafEmbeddingCache instance if mode switch is needed.")

        dataset.current_epoch = epoch
        vfm = self._ensure_vfm(device)

        # Load LRM (pos_embed interpolation handled dynamically per patch)
        lrm = None
        if self.lrm_checkpoint is not None: lrm = self._ensure_lrm(device, n_tokens_override=None) # Load with original n_tokens; interpolation done per-patch in _apply_lrm

        self.aggregator = self.aggregator.to(device)
        self.aggregator.eval()

        self.embeddings = {}
        self.cls_tokens = {}

        mean = self.IMAGENET_MEAN.to(device)
        std = self.IMAGENET_STD.to(device)

        # Respect max_dataset_size (e.g., debug_mode limits to 10)
        max_images = min(dataset.num_images, getattr(dataset.opt, 'max_dataset_size', float('inf')))
        max_images = int(max_images)

        iterator = range(max_images)
        if show_progress: iterator = tqdm(iterator, desc=f"[Epoch {epoch}] Sheaf cache ({max_images} images)")

        for img_idx in iterator:
            for grid_idx in range(NUM_GRIDS):
                result = dataset.get_all_overlap_patches(img_idx, grid_idx)
                patches = result['patches']
                metadata = result['metadata']
                ref_position = result['ref_position']

                if len(patches) == 0:
                    self.embeddings[(img_idx, grid_idx)] = torch.zeros(self.cond_dim)
                    self.cls_tokens[(img_idx, grid_idx)] = torch.zeros(self.vfm_dim)
                    continue

                # PIL → tensor → normalize
                to_tensor = transforms.ToTensor()
                patch_tensors = torch.stack([to_tensor(p) for p in patches])
                patch_tensors = patch_tensors.to(device)
                patch_tensors = (patch_tensors - mean[None]) / std[None]

                # VFM forward
                all_patch_tokens = []
                all_cls = []

                for start in range(0, len(patches), self.batch_size):
                    batch = patch_tensors[start:start + self.batch_size]
                    features = vfm.forward_features(batch)
                    all_cls.append(features[:, 0])
                    all_patch_tokens.append(features[:, 1:])

                all_cls = torch.cat(all_cls, dim=0)
                all_patch_tokens = torch.cat(all_patch_tokens, dim=0)

                # Apply LRM correction (if available)
                if lrm is not None: token_list = self._apply_lrm(all_patch_tokens, lrm, metadata, ref_position)
                else: token_list = [all_patch_tokens[i] for i in range(len(patches))]

                # Store individual patch embeddings (for per-epoch random subset)
                direction_ids = [m['direction_id'] for m in metadata]
                patch_embs = [t.cpu() for t in token_list]  # list of [K, 1536]

                self.patch_embeddings[(img_idx, grid_idx)] = {'embeddings': patch_embs, 'direction_ids': direction_ids,}
                self.cls_tokens[(img_idx, grid_idx)] = all_cls.mean(0).cpu()

        # Build aggregated conditioning from full set (default)
        self._aggregate_all()

        if not keep_models: self._release_models()

        strategy_str = self.token_strategy if lrm is not None else 'no_lrm'
        print(f"[SheafEmbeddingCache] Cached {len(self.patch_embeddings)} entries "
              f"(agg={self.agg_mode}, lrm={strategy_str})")

    def _aggregate_all(self):
        """Aggregate all cached patch embeddings into conditioning vectors."""
        if getattr(self, '_is_spatial', False):
            raise RuntimeError("Cannot call _aggregate_all() in spatial preset mode. "
                               "Create a new SheafEmbeddingCache instance if mode switch is needed.")
        device = self.aggregator.proj.weight.device
        self.embeddings = {}
        
        for key, data in self.patch_embeddings.items():
            # Ensure each embedding is [1, vfm_dim] for aggregator compatibility
            embs = [e.to(device).unsqueeze(0) if e.dim() == 1 else e.to(device) for e in data['embeddings']]

            if len(embs) == 0:
                self.embeddings[key] = torch.zeros(self.cond_dim)
                continue

            cond = self.aggregator(embs, data['direction_ids'])
            self.embeddings[key] = cond.detach().cpu()

    def resample_subset(self, min_patches=5, max_patches=16, seed=None):
        """Resample a random subset of cached patch embeddings and re-aggregate.

        Called at each epoch (between cache refreshes) to add diversity.
        VFM/LRM not needed — only re-aggregates from cached embeddings.

        Not available in spatial preset mode (raises RuntimeError).

        Args:
            min_patches: minimum patches to select per entry
            max_patches: maximum patches to select (16 = all)
            seed: random seed for reproducibility
        """
        if getattr(self, '_is_spatial', False):
            raise RuntimeError("Cannot call resample_subset() in spatial preset mode. "
                               "Create a new SheafEmbeddingCache instance if mode switch is needed.")
        rng = random.Random(seed)
        device = self.aggregator.proj.weight.device
        self.embeddings = {}

        for key, data in self.patch_embeddings.items():
            n_total = len(data['embeddings'])
            if n_total == 0:
                self.embeddings[key] = torch.zeros(self.cond_dim)
                continue

            # Random subset size
            n_select = rng.randint(min(min_patches, n_total), min(max_patches, n_total))
            indices = rng.sample(range(n_total), n_select)

            subset_embs = [data['embeddings'][i].to(device).unsqueeze(0)
                          if data['embeddings'][i].dim() == 1
                          else data['embeddings'][i].to(device)
                          for i in indices]
            
            subset_dirs = [data['direction_ids'][i] for i in indices]
            cond = self.aggregator(subset_embs, subset_dirs)
            
            self.embeddings[key] = cond.detach().cpu()

    def _check_preset_vfm(self, preset_path, data):
        """Stop early when a preset was built with a different VFM.

        A preset stores the backbone and embedding width it was produced with.
        Nothing else compares them against the config, so a mismatch surfaces
        much later as a shape error in the conditioning path, or not at all:
        two backbones can share an embedding width and the run then trains on
        embeddings from the wrong model. Presets written before these fields
        existed carry neither, and are left alone.
        """
        want_dim = getattr(self.opt, 'vfm_embed_dim', None)
        got_dim = data.get('vfm_embed_dim')

        if want_dim is not None and got_dim is not None and int(got_dim) != int(want_dim):
            raise ValueError(f"{preset_path} was built with vfm_embed_dim={got_dim}, but the "
                             f"config sets vfm_embed_dim={want_dim}. Point sheaf_preset_dir "
                             f"at presets built with this VFM, or rebuild them with "
                             f"`bash script/run_presets.sh`.")

        want_vfm = getattr(self.opt, 'vfm_name', None)
        got_vfm = data.get('vfm')

        if want_vfm and got_vfm and str(got_vfm) != str(want_vfm):
            raise ValueError(f"{preset_path} was built with vfm={got_vfm!r}, but the config "
                             f"sets vfm_name={want_vfm!r}. Point sheaf_preset_dir at presets "
                             f"built with this VFM, or rebuild them with "
                             f"`bash script/run_presets.sh`.")

    def load_spatial_preset_from_dir(self, preset_dir, epoch=0, selection='roundrobin'):
        """Load a spatial preset containing [16, 16, 1536] maps.

        Args:
            preset_dir: directory containing spatial_preset_*.pt files
            epoch: current epoch (for selection)
            selection: 'roundrobin', 'random', or 'stratified'

        Returns:
            preset_lookup: {image_idx: (grid_idx, (ref_x, ref_y))}
        """
        preset_files = sorted(glob.glob(os.path.join(preset_dir, 'spatial_preset_*.pt')))

        if len(preset_files) == 0:
            raise FileNotFoundError(f"No spatial preset files found in {preset_dir}. "
                                    f"Generate them first with script/run_presets.sh (util/compute_vfm_presets.py).")

        n = len(preset_files)

        if selection == 'stratified':
            cycle = epoch // n
            pos = epoch % n
            rng = random.Random(cycle)
            order = list(range(n))
            rng.shuffle(order)
            idx = order[pos]

        elif selection == 'random':
            rng = random.Random(epoch)
            idx = rng.randint(0, n - 1)
        
        else: idx = epoch % n

        preset_path = preset_files[idx]
        
        print(f"[SheafEmbeddingCache] Loading spatial preset {preset_path} "
              f"(epoch={epoch}, idx={idx}/{n}, {selection})")

        data = torch.load(preset_path, map_location='cpu', weights_only=False)
        self._check_preset_vfm(preset_path, data)
        cache = data['cache']

        self.embeddings = {}
        self.preset_lookup = {}

        for key, entry in cache.items():
            if isinstance(key, str): key = eval(key)
            img_idx, grid_idx = key

            # spatial_map: [1536, 16, 16] fp16 → float32
            spatial_map = entry['spatial_map'].float()
            self.embeddings[key] = spatial_map

            ref_pos = entry.get('ref_position', (128, 128))
            if img_idx not in self.preset_lookup: self.preset_lookup[img_idx] = {}
            self.preset_lookup[img_idx][grid_idx] = ref_pos

        # ── CLS token loading: branch on --cls_token_type ─────────────
        # 'global'       → whole-image CLS from preset['global_cls'],
        #                  keyed by img_idx.
        # 'neighborhood' → per-grid CLS from cache entries'
        #                  'neighborhood_cls' field, keyed by
        #                  (img_idx, grid_idx).
        #
        # When --use_global_cls=False, CLS injection is fully disabled at
        # the model level (no global_proj is instantiated, forward guard
        # never fires), so we skip touching self.global_cls entirely.
        cls_token_type = getattr(self.opt, 'cls_token_type', 'global')
        use_cls = getattr(self.opt, 'use_global_cls', False)
        preset_version = data.get('version', 'unknown')

        if use_cls:
            if cls_token_type == 'neighborhood':
                neighborhood = {}
                for key, entry in cache.items():
                    if isinstance(key, str): key = eval(key)
                    if 'neighborhood_cls' not in entry:
                        raise RuntimeError(f"--cls_token_type=neighborhood but preset "
                                           f"{preset_path} entry {key} missing "
                                           f"'neighborhood_cls' field "
                                           f"(preset version={preset_version}). "
                                           f"Regenerate the presets with script/run_presets.sh, "
                                           f"or switch to --cls_token_type=global.")
                    neighborhood[tuple(key)] = entry['neighborhood_cls'].float()
                self.global_cls = neighborhood

            elif cls_token_type == 'global':
                # Only overwrite if preset has non-empty global_cls;
                # otherwise preserve previously loaded data (e.g., from
                # load_global_cls() file).
                preset_global_cls = data.get('global_cls', {})

                if preset_global_cls: self.global_cls = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in preset_global_cls.items()}
                elif getattr(self, 'global_cls', None): pass  # preserve file-loaded data
                else: raise RuntimeError(f"--cls_token_type=global but preset {preset_path} "
                                         f"has empty 'global_cls' dict (preset "
                                         f"version={preset_version}). Options: "
                                         f"(i) --cls_token_type=neighborhood, "
                                         f"(ii) load a preset that includes global_cls, "
                                         f"(iii) provide a separate file via "
                                         f"--global_cls_path.")

            else:
                raise ValueError(f"Unknown --cls_token_type: {cls_token_type!r} "
                                 f"(expected 'global' or 'neighborhood')")

        # Mark as spatial mode — prevents accidental vector mode method calls
        self._is_spatial = True

        cls_info = ''
        if use_cls: cls_info = (f', {len(getattr(self, "global_cls", {}))} '
                                f'{cls_token_type}_cls')
        
        print(f"[SheafEmbeddingCache] Loaded {len(self.embeddings)} spatial maps "
              f"from preset {data.get('preset_id', '?')} "
              f"(version={preset_version}){cls_info}")

        return self.preset_lookup

    def get_global_cls_dict(self):
        """Return CLS cache.

        Keyed by image_idx (int) when --cls_token_type=global, or by
        (image_idx, grid_idx) (tuple) when --cls_token_type=neighborhood.
        Values are tensor[1536] fp32. Empty dict if CLS disabled.
        """
        return getattr(self, 'global_cls', {})

    def load_global_cls(self, path):
        """Load global CLS from a separate precomputed file.

        Args:
            path: path to global_cls_*.pt file
                  Format: {image_idx: tensor[1536]}

        Raises:
            FileNotFoundError: if path does not exist (--use_global_cls
                               requires the file to be precomputed)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Global CLS file not found: {path}. "
                                    f"Point --global_cls_path at an existing file, "
                                    f"or disable --use_global_cls.")
        data = torch.load(path, map_location='cpu', weights_only=False)
        self.global_cls = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        print(f"[SheafEmbeddingCache] Loaded {len(self.global_cls)} global CLS "
              f"vectors from {path}")

    def save_aggregator(self, save_dir, suffix='latest'):
        """Save aggregator state_dict alongside model checkpoints.

        Args:
            save_dir: checkpoint directory (opt.checkpoints_dir / opt.name)
            suffix: 'latest', epoch number, or 'iter_N'
        """
        save_path = os.path.join(save_dir, f'sheaf_aggregator_{suffix}.pt')
        torch.save(self.aggregator.state_dict(), save_path)

        return save_path

    def load_aggregator(self, save_dir, suffix='latest', device='cpu'):
        """Load aggregator state_dict from checkpoint.

        Args:
            save_dir: checkpoint directory
            suffix: 'latest', epoch number, or 'iter_N'
            device: target device
        """
        load_path = os.path.join(save_dir, f'sheaf_aggregator_{suffix}.pt')
        if not os.path.exists(load_path):
            print(f"[SheafEmbeddingCache] WARNING: aggregator checkpoint not found: {load_path}")

            return False

        state_dict = torch.load(load_path, map_location=device, weights_only=True)
        self.aggregator.load_state_dict(state_dict)
        print(f"[SheafEmbeddingCache] Loaded aggregator from {load_path}")

        return True

    def get_cache_dict(self):
        return self.embeddings

    def get_cls_cache_dict(self):
        return self.cls_tokens
