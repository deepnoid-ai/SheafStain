import os
import math
import torch
import random

import numpy as np
import pandas as pd

from PIL import Image
from data.base_dataset import BaseDataset
from data.tissue_filter import is_valid_tissue_patch

# Constants
IMAGE_SIZE = 1024
GRID_SIZE = 512
REF_SIZE = 256        # Generator crop size
OVL_SIZE = 224        # VFM input size
NUM_GRIDS = 4         # 2×2

# Grid top-left corners in image coordinates
GRID_ORIGINS = [(0, 0), (512, 0), (0, 512), (512, 512)]

# Direction ID mapping (for tensor-based metadata)
# 8 cardinal + intercardinal directions
DIR_N = 0       # North (up)
DIR_S = 1       # South (down)
DIR_E = 2       # East (right)
DIR_W = 3       # West (left)
DIR_NE = 4      # Northeast (right-upper)
DIR_NW = 5      # Northwest (left-upper)
DIR_SE = 6      # Southeast (right-lower)
DIR_SW = 7      # Southwest (left-lower)

# Legacy aliases for compatibility
DIR_VERTICAL = DIR_N
DIR_HORIZONTAL = DIR_E
DIR_DIAG_RU = DIR_NE
DIR_DIAG_LU = DIR_NW

# Direction → (dx, dy) unit vector (stride=1)
_DIR_VECTORS = {
    DIR_N:  (0, -1),
    DIR_S:  (0, +1),
    DIR_E:  (+1, 0),
    DIR_W:  (-1, 0),
    DIR_NE: (+1, -1),
    DIR_NW: (-1, -1),
    DIR_SE: (+1, +1),
    DIR_SW: (-1, +1),
}


def _direction_displacement(direction_id, distance):
    """Convert direction + distance to (dx, dy) pixel displacement.

    Args:
        direction_id: DIR_N through DIR_SW
        distance: stride in pixels

    Returns:
        (dx, dy) pixel offset from reference center
    """
    ux, uy = _DIR_VECTORS[direction_id]
    if direction_id in (DIR_NE, DIR_NW, DIR_SE, DIR_SW):
        # Diagonal: scale by 1/√2 to maintain same stride distance
        d = int(distance / math.sqrt(2))
        return (ux * d, uy * d)
    else:
        return (ux * distance, uy * distance)


def compute_overlap_region(ref_x, ref_y, ovl_x, ovl_y, ref_size=REF_SIZE, ovl_size=OVL_SIZE):
    """Compute overlap region between reference and overlapped patch.

    Returns crop boxes in both patches' local coordinate frames.
    Used for the sheaf loss on the generated output.

    Args:
        ref_x, ref_y: reference patch top-left in grid coordinates
        ovl_x, ovl_y: overlapped patch top-left in grid coordinates

    Returns:
        dict with 'ref_crop' and 'ovl_crop' as (left, top, right, bottom),
        or None if no overlap.
    """
    # Overlap in grid coordinates
    o_left = max(ref_x, ovl_x)
    o_top = max(ref_y, ovl_y)
    o_right = min(ref_x + ref_size, ovl_x + ovl_size)
    o_bottom = min(ref_y + ref_size, ovl_y + ovl_size)

    if o_left >= o_right or o_top >= o_bottom:
        return None

    return {
        'ref_crop': (o_left - ref_x, o_top - ref_y,
                     o_right - ref_x, o_bottom - ref_y),
        'ovl_crop': (o_left - ovl_x, o_top - ovl_y,
                     o_right - ovl_x, o_bottom - ovl_y),
    }


class SheafDataset(BaseDataset):
    """BCI Dataset with 2×2 grid structure and sheaf patch extraction.

    Each __getitem__ returns one grid's reference patch + overlapped patches.

    Compatible with SheafStain training loop:
        dataset = create_dataset(opt)   # --dataset_mode sheaf
        dataset2 = create_dataset(opt)
        for data, data2 in zip(dataset, dataset2):
            model.set_input(data, data2)
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add sheaf-specific dataset options.

        Note: sheaf_cond_dim, sheaf_injection_mode, sheaf_agg_mode are defined
        in base_options.py (shared across dataset and model).
        """
        # sheaf_stride, sheaf_num_jitter_vh, sheaf_jitter_range defined in base_options.py
        parser.add_argument('--min_tissue_ratio', type=float, default=0.3, help='Minimum tissue ratio for reference patch (Otsu)')
        parser.add_argument('--max_ref_attempts', type=int, default=50, help='Max attempts to find tissue-rich reference position')
        parser.add_argument('--return_overlap_patches', action='store_true', help='Return raw overlap patches (for cache computation)')
        parser.add_argument('--grid_augment', action='store_true', help='Apply augmentation at grid level (flip, rotation, color jitter) before patch extraction. Preserves spatial correspondence.')
        parser.add_argument('--grid_color_jitter', type=float, default=0.1, help='Color jitter strength for grid augmentation (brightness, contrast, saturation, hue)')

        # Dataset structure selectors (consumed by _load_paths)
        parser.add_argument('--stain', type=str, default='her2', choices=['her2', 'ki67', 'er', 'pr'], help='IHC stain subdirectory under image/psi/{he,ihc}/{stain}')
        parser.add_argument('--train_split_mode', type=str, default='bci', choices=['bci', 'mist'], help='Train-time split policy. bci: use split in {train,val}; mist: use split == train only')
        parser.add_argument('--img_ext', type=str, default='.png', help='Image file extension (e.g., .png, .jpg, .tif)')
        
        return parser

    def __init__(self, opt):
        """Initialize sheaf-aware BCI dataset.

        Args:
            opt: Options with dataroot, phase, sheaf_stride, etc.
        """
        BaseDataset.__init__(self, opt)

        # Patch extraction parameters
        self.stride = opt.sheaf_stride
        self.min_tissue_ratio = opt.min_tissue_ratio
        self.max_ref_attempts = opt.max_ref_attempts
        self.cache_refresh_freq = getattr(opt, 'sheaf_cache_refresh_freq', 5)
        self.return_overlap_patches = getattr(opt, 'return_overlap_patches', False)
        self.sheaf_cond_dim = getattr(opt, 'sheaf_cond_dim', 256)

        # Grid-level augmentation
        self.grid_augment = getattr(opt, 'grid_augment', False)
        self.grid_color_jitter = getattr(opt, 'grid_color_jitter', 0.1)

        # Surrounding patch extraction (8 directions)
        # V/H: 1 center + num_jitter lateral patches per direction (×4 dirs = N,S,E,W)
        # Diag: 1 patch per direction (×4 dirs = NE,NW,SE,SW)
        self.num_jitter_vh = getattr(opt, 'sheaf_num_jitter_vh', 2)
        self.jitter_range = getattr(opt, 'sheaf_jitter_range', 32)
        self.patches_per_cardinal = 1 + self.num_jitter_vh  # center + jitter
        # max_overlaps: 4 cardinal × (1+jitter) + 4 diagonal × 1
        self.max_overlaps = 4 * self.patches_per_cardinal + 4

        # Load image paths
        self._load_paths(opt)

        # Reference patch transform: [-1, 1] for generator
        self.ref_transform = self._build_ref_transform()

        # Sheaf embedding cache: {(image_idx, grid_idx): tensor}
        # Vector mode: tensor is [cond_dim], Spatial mode: tensor is [VFM_DIM, 16, 16]
        self._sheaf_cache = {}
        self._sheaf_spatial = getattr(opt, 'sheaf_spatial', False)

        # Global CLS cache: {image_idx: tensor[1536]}
        # Image-level, grid-independent. Applied to 100% samples.
        self._global_cls_cache = {}

        # Preset lookup: {image_idx: (grid_idx, (ref_x, ref_y))}
        # When set, __getitem__ uses these instead of RNG-based selection
        self._preset_lookup = None

        is_rank0 = not getattr(opt, 'use_ddp', False) or opt.rank == 0
        if is_rank0:
            print(f"[SheafDataset] {self.num_images} images, "
                  f"stride={self.stride}, "
                  f"V/H: {self.patches_per_cardinal}/dir (1 center + {self.num_jitter_vh} jitter), "
                  f"Diag: 1/dir, "
                  f"max_overlaps={self.max_overlaps}")

    def _load_paths(self, opt):
        """Load HE/IHC image paths from dataset structure (BCI/MIST)."""
        stain = getattr(opt, 'stain', 'her2')
        img_ext = getattr(opt, 'img_ext', '.png')
        train_split_mode = getattr(opt, 'train_split_mode', 'bci')

        dir_A = os.path.join(opt.dataroot, 'image', 'psi', 'he', stain)
        dir_B = os.path.join(opt.dataroot, 'image', 'psi', 'ihc', stain)
        labels_path = os.path.join(opt.dataroot, 'label', 'psi', stain, 'labels.csv')

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.csv not found at {labels_path}")

        df = pd.read_csv(labels_path)

        if opt.phase == 'train':
            if train_split_mode == 'mist':
                df = df[df['split'] == 'train']
            else:
                df = df[df['split'].isin(['train', 'val'])]
        elif opt.phase == 'val':
            df = df[df['split'] == 'val']
        elif opt.phase == 'test':
            df = df[df['split'] == 'test']
        else:
            df = df[df['split'] == opt.phase]

        self.A_paths = []
        self.B_paths = []
        for _, row in df.iterrows():
            image_id = row['image_id']
            a_path = os.path.join(dir_A, f"{image_id}{img_ext}")
            b_path = os.path.join(dir_B, f"{image_id}{img_ext}")
            if os.path.exists(a_path) and os.path.exists(b_path):
                self.A_paths.append(a_path)
                self.B_paths.append(b_path)

        self.num_images = len(self.A_paths)
        if self.num_images == 0:
            raise ValueError(f"No images found for phase '{opt.phase}'")

    def _augment_grid(self, he_grid, ihc_grid, rng):
        """Apply augmentation to grid-level images (512×512).

        Applied BEFORE patch extraction so all patches share the same
        augmentation state, preserving spatial correspondence.

        Augmentations (all applied consistently to HE and IHC):
          - Random horizontal flip (50%)
          - Random vertical flip (50%)
          - Random 90° rotation (0/90/180/270)

        Args:
            he_grid, ihc_grid: PIL Images (512×512)
            rng: seeded Random instance

        Returns:
            augmented (he_grid, ihc_grid) as PIL Images,
            aug_params: dict with 'hflip', 'vflip', 'rot_k' for replay
        """
        hflip = rng.random() > 0.5
        vflip = rng.random() > 0.5
        rot_k = rng.randint(0, 3)  # 0, 90, 180, 270

        if hflip:
            he_grid = he_grid.transpose(Image.FLIP_LEFT_RIGHT)
            ihc_grid = ihc_grid.transpose(Image.FLIP_LEFT_RIGHT)

        if vflip:
            he_grid = he_grid.transpose(Image.FLIP_TOP_BOTTOM)
            ihc_grid = ihc_grid.transpose(Image.FLIP_TOP_BOTTOM)

        if rot_k > 0:
            rot_map = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}
            he_grid = he_grid.transpose(rot_map[rot_k])
            ihc_grid = ihc_grid.transpose(rot_map[rot_k])

        aug_params = {'hflip': hflip, 'vflip': vflip, 'rot_k': rot_k}
        return he_grid, ihc_grid, aug_params

    @staticmethod
    def _transform_ref_coords(ref_x, ref_y, aug_params,
                              grid_size=GRID_SIZE, ref_size=REF_SIZE):
        """Transform reference coordinates to match grid augmentation.

        Preset ref_position is in the original (pre-augmentation) grid.
        This function computes where that tissue ended up after augmentation,
        so cropping at the transformed coordinates yields the same tissue.

        Transforms are applied in the same order as _augment_grid:
        hflip → vflip → rotation.

        Args:
            ref_x, ref_y: original ref position (top-left of 256×256 patch)
            aug_params: dict with 'hflip', 'vflip', 'rot_k'
            grid_size: 512
            ref_size: 256

        Returns:
            (new_ref_x, new_ref_y): transformed coordinates
        """
        x, y = ref_x, ref_y
        max_coord = grid_size - ref_size  # 256

        if aug_params['hflip']:
            x = max_coord - x

        if aug_params['vflip']:
            y = max_coord - y

        rot_k = aug_params['rot_k']
        if rot_k == 1:      # 90° CCW
            x, y = y, max_coord - x
        elif rot_k == 2:    # 180°
            x, y = max_coord - x, max_coord - y
        elif rot_k == 3:    # 270° CCW (= 90° CW)
            x, y = max_coord - y, x

        return x, y

    @staticmethod
    def _apply_aug_to_spatial_map(spatial_map, aug_params):
        """Apply the same augmentation to a spatial conditioning map.

        Matches the augmentation applied to the grid by _augment_grid().
        spatial_map is [C, H, W] tensor; flip/rotation on spatial dims (1, 2).

        Args:
            spatial_map: [C, H, W] tensor (e.g., [1536, 16, 16])
            aug_params: dict from _augment_grid with 'hflip', 'vflip', 'rot_k'

        Returns:
            augmented spatial_map [C, H, W]
        """
        if aug_params['hflip']:
            spatial_map = torch.flip(spatial_map, dims=[2])  # flip W axis

        if aug_params['vflip']:
            spatial_map = torch.flip(spatial_map, dims=[1])  # flip H axis

        if aug_params['rot_k'] > 0:
            # PIL ROTATE_90 = 90° counter-clockwise = torch.rot90 k=1
            spatial_map = torch.rot90(spatial_map, k=aug_params['rot_k'], dims=[1, 2])

        return spatial_map

    def _build_ref_transform(self):
        """Transform for reference patches: [0,255] PIL → [-1,1] tensor."""
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToTensor(),                                    # [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [-1, 1]
        ])

    def __len__(self):
        """All 4 grids per image. index → (image_idx, grid_idx)."""
        return self.num_images * NUM_GRIDS

    def __getitem__(self, index):
        """Return reference patch + overlap metadata for one grid.

        Returns dict with keys:
            A:                   [3, 256, 256]  H&E reference (generator input)
            B:                   [3, 256, 256]  IHC reference (target)
            A_paths, B_paths:    str            image file paths
            sheaf_cond:          [sheaf_cond_dim]  cached conditioning (zeros if not cached)
            overlap_patches:     [max_ovl, 3, 224, 224]  raw patches (if return_overlap_patches)
            overlap_valid:       [max_ovl]      1.0 for valid, 0.0 for padded
            overlap_positions:   [max_ovl, 2]   (x, y) top-left in grid coords
            overlap_directions:  [max_ovl]      direction IDs (0=V, 1=H, 2=DRU, 3=DLU)
            overlap_offsets:     [max_ovl]      stride offsets in pixels
            overlap_regions_ref: [max_ovl, 4]   crop box in reference frame (l, t, r, b)
            overlap_regions_ovl: [max_ovl, 4]   crop box in overlap frame (l, t, r, b)
            ref_position:        [2]            (x, y) reference top-left in grid coords
            grid_idx:            int
            image_idx:           int
            num_overlaps:        int            number of valid overlaps
        """
        image_idx = index // NUM_GRIDS
        grid_idx = index % NUM_GRIDS

        # Determine ref_position
        eff_epoch = (self.current_epoch // self.cache_refresh_freq) * self.cache_refresh_freq

        if self._preset_lookup is not None and image_idx in self._preset_lookup:
            grid_lookup = self._preset_lookup[image_idx]  # {grid_idx: ref_pos}
            if grid_idx in grid_lookup:
                ref_x, ref_y = grid_lookup[grid_idx]
                rng = random.Random(
                    image_idx * 1000 + grid_idx * 100 + eff_epoch * 7)
            else:
                # Grid not in preset → RNG-based ref position
                rng = random.Random(
                    image_idx * 1000 + grid_idx * 100 + eff_epoch * 7)
                ref_x, ref_y = None, None
        else:
            # Fallback: RNG-based selection (no preset)
            rng = random.Random(
                image_idx * 1000 + grid_idx * 100 + eff_epoch * 7)
            ref_x, ref_y = None, None

        # Load full images
        he_img = Image.open(self.A_paths[image_idx]).convert('RGB')
        ihc_img = Image.open(self.B_paths[image_idx]).convert('RGB')

        gx, gy = GRID_ORIGINS[grid_idx]

        # Crop grid (512×512)
        grid_he = he_img.crop((gx, gy, gx + GRID_SIZE, gy + GRID_SIZE))
        grid_ihc = ihc_img.crop((gx, gy, gx + GRID_SIZE, gy + GRID_SIZE))

        # Grid-level augmentation (before any patch extraction)
        aug_params = None
        if self.grid_augment and self.opt.isTrain:
            aug_rng = random.Random()
            grid_he, grid_ihc, aug_params = self._augment_grid(
                grid_he, grid_ihc, aug_rng)

        # Transform preset coordinates to match augmented grid
        # Preset ref_position is in the original (pre-augmentation) grid.
        # After augmentation, the tissue at that position has moved.
        # Transform the coordinates so they point to the same tissue.
        if ref_x is not None and aug_params is not None:
            ref_x, ref_y = self._transform_ref_coords(
                ref_x, ref_y, aug_params)

        # Select reference patch (256×256)
        if ref_x is None or ref_y is None:
            # RNG-based: Otsu-filtered random selection
            ref_x, ref_y = self._select_reference_from_grid(grid_he, rng)

        ref_he = grid_he.crop((ref_x, ref_y, ref_x + REF_SIZE, ref_y + REF_SIZE))
        ref_ihc = grid_ihc.crop((ref_x, ref_y, ref_x + REF_SIZE, ref_y + REF_SIZE))

        # Extract overlapped patches (224×224) from grid
        ovl_result = self._extract_overlapped_from_grid(
            grid_he, ref_x, ref_y, rng)

        # Grid tensors for logging
        from torchvision import transforms
        grid_to_tensor = transforms.ToTensor()
        grid_he_t = grid_to_tensor(grid_he)     # [3, 512, 512] in [0, 1]
        grid_ihc_t = grid_to_tensor(grid_ihc)

        # Apply transforms
        ref_he_t = self.ref_transform(ref_he)
        ref_ihc_t = self.ref_transform(ref_ihc)

        # Pack into fixed-size tensors (pad to max_overlaps)
        num_ovl = len(ovl_result['patches'])

        # Overlap validity mask
        overlap_valid = torch.zeros(self.max_overlaps)
        overlap_valid[:num_ovl] = 1.0

        # Positions, directions, offsets
        overlap_positions = torch.zeros(self.max_overlaps, 2)
        overlap_directions = torch.zeros(self.max_overlaps, dtype=torch.long)
        overlap_offsets = torch.zeros(self.max_overlaps)
        overlap_regions_ref = torch.zeros(self.max_overlaps, 4)
        overlap_regions_ovl = torch.zeros(self.max_overlaps, 4)

        for i in range(num_ovl):
            meta = ovl_result['metadata'][i]
            overlap_positions[i] = torch.tensor(meta['position'], dtype=torch.float32)
            overlap_directions[i] = meta['direction_id']
            overlap_offsets[i] = meta['offset']
            overlap_regions_ref[i] = torch.tensor(meta['region_in_ref'], dtype=torch.float32)
            overlap_regions_ovl[i] = torch.tensor(meta['region_in_ovl'], dtype=torch.float32)

        # Build return dict
        result = {
            # Standard fields (sb_model.set_input compatible)
            'A': ref_he_t,                                         # [3, 256, 256]
            'B': ref_ihc_t,                                        # [3, 256, 256]
            'A_paths': self.A_paths[image_idx],
            'B_paths': self.B_paths[image_idx],

            # Grid images (512×512, [0,1] for logging only)
            'grid_he': grid_he_t,                                      # [3, 512, 512]
            'grid_ihc': grid_ihc_t,                                    # [3, 512, 512]

            # Sheaf conditioning (populated by external cache)
            # Spatial mode: [VFM_DIM, 16, 16], Vector mode: [cond_dim]
            'sheaf_cond': self._get_sheaf_cond(
                image_idx, grid_idx, aug_params),

            # CLS-token conditioning (image- or grid-level).
            # Dual-key lookup: tuple key (neighborhood mode) takes priority;
            # falls back to int key (global mode);
            # final fallback is a zero vector (cache miss or CLS disabled).
            # The cache content is only read when the model was built with
            # --use_global_cls (otherwise global_proj is never instantiated).
            'global_cls': self._global_cls_cache.get(
                (image_idx, grid_idx),
                self._global_cls_cache.get(
                    image_idx, torch.zeros(1536))),

            # Overlap spatial metadata (fixed-size, padded)
            'overlap_valid': overlap_valid,                        # [max_ovl]
            'overlap_positions': overlap_positions,                # [max_ovl, 2]
            'overlap_directions': overlap_directions,              # [max_ovl]
            'overlap_offsets': overlap_offsets,                     # [max_ovl]
            'overlap_regions_ref': overlap_regions_ref,            # [max_ovl, 4]
            'overlap_regions_ovl': overlap_regions_ovl,            # [max_ovl, 4]

            # Scalars
            'ref_position': torch.tensor([ref_x, ref_y], dtype=torch.float32),
            'grid_idx': grid_idx,
            'image_idx': image_idx,
            'num_overlaps': num_ovl,
        }

        # Adjacent patches for sheaf loss
        any_sheaf = (getattr(self.opt, 'lambda_sheaf', 0) > 0 or
                     getattr(self.opt, 'lambda_sheaf_feat', 0) > 0)
        use_cocycle = getattr(self.opt, 'lambda_sheaf_cocycle', 0) > 0

        if any_sheaf or use_cocycle:
            # Try adjacent pair first (for cocycle); fall back to single adj
            adj1, adj2 = None, None
            if use_cocycle:
                pair = self._get_adjacent_pair(
                    grid_he, grid_ihc, ref_x, ref_y, rng)
                if pair is not None:
                    adj1 = pair['adj1']
                    adj2 = pair['adj2']

            # Fall back to single adj if pair not found
            if adj1 is None:
                adj1 = self._get_adjacent_patch(
                    grid_he, grid_ihc, ref_x, ref_y, rng)

            # adj1 (pairwise sheaf loss)
            if adj1 is not None:
                result['adj_A'] = self.ref_transform(adj1['he'])
                result['adj_B'] = self.ref_transform(adj1['ihc'])
                result['adj_offset_x'] = adj1['offset_x']
                result['adj_offset_y'] = adj1['offset_y']
                result['adj_sheaf_cond'] = self._shift_spatial_map_for_adj(
                    result['sheaf_cond'], adj1['offset_x'], adj1['offset_y'])
                result['has_adj'] = 1
            else:
                result['adj_A'] = torch.zeros(3, REF_SIZE, REF_SIZE)
                result['adj_B'] = torch.zeros(3, REF_SIZE, REF_SIZE)
                result['adj_offset_x'] = 0
                result['adj_offset_y'] = 0
                result['adj_sheaf_cond'] = torch.zeros_like(result['sheaf_cond'])
                result['has_adj'] = 0

            # adj2 (cocycle loss — triple overlap consistency)
            if adj2 is not None:
                result['adj2_A'] = self.ref_transform(adj2['he'])
                result['adj2_B'] = self.ref_transform(adj2['ihc'])
                result['adj2_offset_x'] = adj2['offset_x']
                result['adj2_offset_y'] = adj2['offset_y']
                result['adj2_sheaf_cond'] = self._shift_spatial_map_for_adj(
                    result['sheaf_cond'], adj2['offset_x'], adj2['offset_y'])
                result['has_adj2'] = 1
            else:
                result['adj2_A'] = torch.zeros(3, REF_SIZE, REF_SIZE)
                result['adj2_B'] = torch.zeros(3, REF_SIZE, REF_SIZE)
                result['adj2_offset_x'] = 0
                result['adj2_offset_y'] = 0
                result['adj2_sheaf_cond'] = torch.zeros_like(result['sheaf_cond'])
                result['has_adj2'] = 0

        # Raw overlap patches (only when building cache)
        if self.return_overlap_patches:
            from torchvision import transforms
            ovl_to_tensor = transforms.ToTensor()  # [0, 1], no normalization
            ovl_tensor = torch.zeros(self.max_overlaps, 3, OVL_SIZE, OVL_SIZE)
            for i, patch in enumerate(ovl_result['patches']):
                ovl_tensor[i] = ovl_to_tensor(patch)
            result['overlap_patches'] = ovl_tensor                 # [max_ovl, 3, 224, 224]

        return result

    # Sheaf conditioning retrieval
    def _get_sheaf_cond(self, image_idx, grid_idx, aug_params=None):
        """Retrieve sheaf conditioning, applying grid augmentation if needed.

        Args:
            image_idx, grid_idx: cache lookup keys
            aug_params: dict from _augment_grid (None = no augmentation)

        Returns:
            tensor: [cond_dim] (vector mode) or [1536, 16, 16] (spatial mode)
        """
        if self._sheaf_spatial:
            default = torch.zeros(1536, 16, 16)
        else:
            default = torch.zeros(self.sheaf_cond_dim)

        cond = self._sheaf_cache.get((image_idx, grid_idx), default)

        # Apply same augmentation to spatial map for alignment
        if self._sheaf_spatial and aug_params is not None:
            cond = self._apply_aug_to_spatial_map(cond.clone(), aug_params)

        return cond

    # Spatial map shift for adjacent patch
    @staticmethod
    def _shift_spatial_map_for_adj(spatial_map, offset_x, offset_y,
                                    token_px=16):
        """Shift spatial conditioning map to align with adjacent patch coords.

        When adj patch is offset by (dx, dy) pixels from ref, the spatial map
        must be shifted by (-dx//token_px, -dy//token_px) tokens so that each
        position in adj receives the correct conditioning.

        The shifted-out region (outside overlap) is filled with zeros,
        which is correct because sheaf loss only compares the overlap region.

        Args:
            spatial_map: [C, H, W] or [cond_dim] tensor
            offset_x, offset_y: adj offset in pixels relative to ref
            token_px: VFM token size in pixels (default 16)

        Returns:
            shifted spatial map (same shape)
        """
        # Vector mode: no spatial structure to shift
        if spatial_map.dim() == 1:
            return spatial_map.clone()

        # [C, H, W] spatial mode
        shift_x = offset_x // token_px  # e.g., 80 // 16 = 5
        shift_y = offset_y // token_px

        if shift_x == 0 and shift_y == 0:
            return spatial_map.clone()

        shifted = torch.zeros_like(spatial_map)
        C, H, W = spatial_map.shape

        # Compute valid source and destination ranges
        # src: region in ref spatial map that maps to adj's coordinate system
        # dst: corresponding region in shifted map
        src_y0 = max(0, shift_y)
        src_y1 = min(H, H + shift_y)
        src_x0 = max(0, shift_x)
        src_x1 = min(W, W + shift_x)

        dst_y0 = max(0, -shift_y)
        dst_y1 = min(H, H - shift_y)
        dst_x0 = max(0, -shift_x)
        dst_x1 = min(W, W - shift_x)

        if dst_y1 > dst_y0 and dst_x1 > dst_x0:
            shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = \
                spatial_map[:, src_y0:src_y1, src_x0:src_x1]

        return shifted

    # Adjacent patch for sheaf loss
    def _get_adjacent_patch(self, grid_he, grid_ihc, ref_x, ref_y, rng):
        """Get an adjacent overlapping patch for pixel sheaf loss.

        Selects a random valid direction (cardinal) and returns a 256×256 patch
        shifted by self.stride from the ref position.

        Returns:
            dict with 'he', 'ihc' (PIL), 'offset_x', 'offset_y'
            or None if no valid direction exists
        """
        stride = self.stride  # 80
        directions = [(stride, 0), (-stride, 0), (0, stride), (0, -stride)]
        rng.shuffle(directions)

        for dx, dy in directions:
            adj_x = ref_x + dx
            adj_y = ref_y + dy
            if (0 <= adj_x and adj_x + REF_SIZE <= GRID_SIZE and
                    0 <= adj_y and adj_y + REF_SIZE <= GRID_SIZE):
                adj_he = grid_he.crop(
                    (adj_x, adj_y, adj_x + REF_SIZE, adj_y + REF_SIZE))
                adj_ihc = grid_ihc.crop(
                    (adj_x, adj_y, adj_x + REF_SIZE, adj_y + REF_SIZE))
                return {
                    'he': adj_he, 'ihc': adj_ihc,
                    'offset_x': dx, 'offset_y': dy,
                }
        return None

    def _get_adjacent_pair(self, grid_he, grid_ihc, ref_x, ref_y, rng):
        """Get two adjacent patches forming a triple overlap with ref.

        Selects perpendicular cardinal pairs (e.g., E+S) that guarantee
        a large triple overlap region for cocycle consistency loss.

        Returns:
            dict with 'adj1' and 'adj2' (each has 'he','ihc','offset_x','offset_y')
            or None if no valid pair exists
        """
        stride = self.stride
        # All perpendicular cardinal pairs (both orderings for uniform adj1 sampling)
        pairs = [
            ((stride, 0), (0, stride)),      # E + S
            ((stride, 0), (0, -stride)),     # E + N
            ((-stride, 0), (0, stride)),     # W + S
            ((-stride, 0), (0, -stride)),    # W + N
            ((0, stride), (stride, 0)),      # S + E
            ((0, stride), (-stride, 0)),     # S + W
            ((0, -stride), (stride, 0)),     # N + E
            ((0, -stride), (-stride, 0)),    # N + W
        ]
        rng.shuffle(pairs)

        for (dx1, dy1), (dx2, dy2) in pairs:
            a1x, a1y = ref_x + dx1, ref_y + dy1
            a2x, a2y = ref_x + dx2, ref_y + dy2
            if (0 <= a1x and a1x + REF_SIZE <= GRID_SIZE and
                    0 <= a1y and a1y + REF_SIZE <= GRID_SIZE and
                    0 <= a2x and a2x + REF_SIZE <= GRID_SIZE and
                    0 <= a2y and a2y + REF_SIZE <= GRID_SIZE):
                return {
                    'adj1': {
                        'he': grid_he.crop((a1x, a1y, a1x+REF_SIZE, a1y+REF_SIZE)),
                        'ihc': grid_ihc.crop((a1x, a1y, a1x+REF_SIZE, a1y+REF_SIZE)),
                        'offset_x': dx1, 'offset_y': dy1,
                    },
                    'adj2': {
                        'he': grid_he.crop((a2x, a2y, a2x+REF_SIZE, a2y+REF_SIZE)),
                        'ihc': grid_ihc.crop((a2x, a2y, a2x+REF_SIZE, a2y+REF_SIZE)),
                        'offset_x': dx2, 'offset_y': dy2,
                    },
                }
        return None

    # Reference selection
    def _select_reference(self, full_image, gx, gy, rng):
        """Select tissue-rich 256×256 position within 512×512 grid.

        Uses deterministic RNG for consistency across DataLoader workers.

        Args:
            full_image: PIL Image (1024×1024)
            gx, gy: grid origin in image coordinates
            rng: seeded Random instance

        Returns:
            (ref_x, ref_y) position within grid (0-indexed, grid-local)
        """
        max_offset = GRID_SIZE - REF_SIZE  # 256

        for _ in range(self.max_ref_attempts):
            rx = rng.randint(0, max_offset)
            ry = rng.randint(0, max_offset)

            # Crop from full image
            crop = full_image.crop((gx + rx, gy + ry,
                                    gx + rx + REF_SIZE, gy + ry + REF_SIZE))
            crop_np = np.array(crop)

            if is_valid_tissue_patch(crop_np, min_tissue_ratio=self.min_tissue_ratio):
                return rx, ry

        # Fallback: grid center
        return max_offset // 2, max_offset // 2

    def _select_reference_from_grid(self, grid_img, rng):
        """Select tissue-rich 256×256 position within a 512×512 grid PIL image.

        Args:
            grid_img: PIL Image (512×512, possibly augmented)
            rng: seeded Random instance

        Returns:
            (ref_x, ref_y) position within grid (0-indexed)
        """
        max_offset = GRID_SIZE - REF_SIZE

        for _ in range(self.max_ref_attempts):
            rx = rng.randint(0, max_offset)
            ry = rng.randint(0, max_offset)

            crop = grid_img.crop((rx, ry, rx + REF_SIZE, ry + REF_SIZE))
            crop_np = np.array(crop)

            if is_valid_tissue_patch(crop_np, min_tissue_ratio=self.min_tissue_ratio):
                return rx, ry

        return max_offset // 2, max_offset // 2

    # Overlapped patch extraction (grid-local)
    def _extract_overlapped_from_grid(self, grid_img, ref_x, ref_y, rng):
        """Extract 224×224 overlapped patches from a grid PIL image.

        Same logic as _extract_overlapped but operates on grid-local coordinates.

        Args:
            grid_img: PIL Image (512×512, possibly augmented)
            ref_x, ref_y: reference position within grid
            rng: seeded Random instance

        Returns:
            dict with 'patches' (list of PIL Images) and 'metadata' (list of dicts)
        """
        patches = []
        metadata = []

        ref_cx = ref_x + REF_SIZE // 2
        ref_cy = ref_y + REF_SIZE // 2

        def _try_extract(direction_id, dx, dy, jitter_offset=0):
            ocx = ref_cx + dx
            ocy = ref_cy + dy
            ox = ocx - OVL_SIZE // 2
            oy = ocy - OVL_SIZE // 2

            if not (0 <= ox and ox + OVL_SIZE <= GRID_SIZE and
                    0 <= oy and oy + OVL_SIZE <= GRID_SIZE):
                return False

            region = compute_overlap_region(ref_x, ref_y, ox, oy)
            if region is None:
                return False

            patch = grid_img.crop((ox, oy, ox + OVL_SIZE, oy + OVL_SIZE))
            patches.append(patch)
            metadata.append({
                'direction_id': direction_id,
                'offset': self.stride,
                'jitter': jitter_offset,
                'position': (ox, oy),
                'center': (ocx, ocy),
                'region_in_ref': region['ref_crop'],
                'region_in_ovl': region['ovl_crop'],
            })
            return True

        # Cardinal directions (N, S, E, W): center + jitter
        cardinal_dirs = [DIR_N, DIR_S, DIR_E, DIR_W]
        for direction_id in cardinal_dirs:
            main_dx, main_dy = _direction_displacement(direction_id, self.stride)
            _try_extract(direction_id, main_dx, main_dy)
            for j in range(self.num_jitter_vh):
                jitter = rng.randint(-self.jitter_range, self.jitter_range)
                if direction_id in (DIR_N, DIR_S):
                    _try_extract(direction_id, main_dx + jitter, main_dy,
                                 jitter_offset=jitter)
                else:
                    _try_extract(direction_id, main_dx, main_dy + jitter,
                                 jitter_offset=jitter)

        # Diagonal directions (NE, NW, SE, SW): 1 each
        for direction_id in [DIR_NE, DIR_NW, DIR_SE, DIR_SW]:
            dx, dy = _direction_displacement(direction_id, self.stride)
            _try_extract(direction_id, dx, dy)

        return {'patches': patches, 'metadata': metadata}

    # Legacy methods (for cache computation from full image)
    def _extract_overlapped(self, full_image, gx, gy, ref_x, ref_y, rng):
        """Extract 224×224 overlapped patches surrounding the reference patch.

        8-direction surrounding extraction:
          - Cardinal (N,S,E,W): 1 center patch + lateral jitter patches per dir
          - Diagonal (NE,NW,SE,SW): 1 patch per dir (no jitter)
        All patches at stride distance from ref center (same overlap amount).
        V/H jitter: lateral offset perpendicular to the main direction,
        covering more of the ref boundary.

        Args:
            full_image: PIL Image (1024×1024)
            gx, gy: grid origin in image coordinates
            ref_x, ref_y: reference position within grid
            rng: seeded Random instance

        Returns:
            dict with 'patches' (list of PIL Images) and 'metadata' (list of dicts)
        """
        patches = []
        metadata = []

        ref_cx = ref_x + REF_SIZE // 2
        ref_cy = ref_y + REF_SIZE // 2

        def _try_extract(direction_id, dx, dy, jitter_offset=0):
            """Try to extract a patch at (ref_center + dx, dy). Returns True if valid."""
            ocx = ref_cx + dx
            ocy = ref_cy + dy
            ox = ocx - OVL_SIZE // 2
            oy = ocy - OVL_SIZE // 2

            # Bounds check within grid
            if not (0 <= ox and ox + OVL_SIZE <= GRID_SIZE and
                    0 <= oy and oy + OVL_SIZE <= GRID_SIZE):
                return False

            # Overlap check with reference
            region = compute_overlap_region(ref_x, ref_y, ox, oy)
            if region is None:
                return False

            patch = full_image.crop((gx + ox, gy + oy,
                                    gx + ox + OVL_SIZE, gy + oy + OVL_SIZE))
            patches.append(patch)
            metadata.append({
                'direction_id': direction_id,
                'offset': self.stride,
                'jitter': jitter_offset,
                'position': (ox, oy),
                'center': (ocx, ocy),
                'region_in_ref': region['ref_crop'],
                'region_in_ovl': region['ovl_crop'],
            })
            return True

        # Cardinal directions (N, S, E, W): center + jitter
        cardinal_dirs = [DIR_N, DIR_S, DIR_E, DIR_W]
        for direction_id in cardinal_dirs:
            # Main displacement at stride distance
            main_dx, main_dy = _direction_displacement(direction_id, self.stride)

            # 1) Center patch (no jitter)
            _try_extract(direction_id, main_dx, main_dy)

            # 2) Jitter patches: lateral offset perpendicular to main direction
            for j in range(self.num_jitter_vh):
                # Deterministic jitter offset from RNG
                jitter = rng.randint(-self.jitter_range, self.jitter_range)

                if direction_id in (DIR_N, DIR_S):
                    # Vertical direction → jitter along x-axis (left/right)
                    _try_extract(direction_id, main_dx + jitter, main_dy,
                                 jitter_offset=jitter)
                else:
                    # Horizontal direction → jitter along y-axis (up/down)
                    _try_extract(direction_id, main_dx, main_dy + jitter,
                                 jitter_offset=jitter)

        # Diagonal directions (NE, NW, SE, SW): 1 patch each, no jitter
        diagonal_dirs = [DIR_NE, DIR_NW, DIR_SE, DIR_SW]
        for direction_id in diagonal_dirs:
            dx, dy = _direction_displacement(direction_id, self.stride)
            _try_extract(direction_id, dx, dy)

        return {'patches': patches, 'metadata': metadata}

    # Cache management
    def set_sheaf_cache(self, cache_dict):
        """Inject pre-computed sheaf conditioning embeddings.

        Called from training loop after external VFM+LRM computation.

        Args:
            cache_dict: {(image_idx, grid_idx): tensor[sheaf_cond_dim]}
        """
        self._sheaf_cache = cache_dict

    def set_global_cls_cache(self, cls_dict):
        """Set global CLS token cache (image-level, grid-independent).

        Args:
            cls_dict: {image_idx: tensor[1536]}
        """
        self._global_cls_cache = cls_dict

    def set_preset_lookup(self, lookup):
        """Set preset lookup table for deterministic grid/ref selection.

        When set, __getitem__ uses these values instead of RNG-based selection.

        Args:
            lookup: {image_idx: (grid_idx, (ref_x, ref_y))} or None to clear
        """
        self._preset_lookup = lookup

    def clear_sheaf_cache(self):
        """Clear cached sheaf embeddings (e.g., at epoch boundary)."""
        self._sheaf_cache = {}
        self._preset_lookup = None

    def get_all_overlap_patches(self, image_idx, grid_idx):
        """Extract overlap patches for a specific (image, grid) pair.

        Convenience method for external cache computation.
        Uses the same deterministic RNG as __getitem__ (effective_epoch).
        NOTE: No augmentation applied here — cache is computed on original images.

        Returns:
            dict with 'patches' (list of PIL), 'metadata' (list of dict),
            'ref_position' (tuple)
        """
        eff_epoch = (self.current_epoch // self.cache_refresh_freq) * self.cache_refresh_freq
        rng = random.Random(image_idx * 1000 + grid_idx * 100 + eff_epoch * 7)

        he_img = Image.open(self.A_paths[image_idx]).convert('RGB')
        gx, gy = GRID_ORIGINS[grid_idx]
        grid_he = he_img.crop((gx, gy, gx + GRID_SIZE, gy + GRID_SIZE))

        ref_x, ref_y = self._select_reference_from_grid(grid_he, rng)
        ovl_result = self._extract_overlapped_from_grid(grid_he, ref_x, ref_y, rng)

        return {
            'patches': ovl_result['patches'],
            'metadata': ovl_result['metadata'],
            'ref_position': (ref_x, ref_y),
        }
