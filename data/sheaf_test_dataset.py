import os
import math
import torch
import random
import hashlib

import numpy as np

from PIL import Image
from data.base_dataset import BaseDataset
from data.tissue_filter import is_valid_tissue_patch

# Constants
IMAGE_SIZE = 1024
REF_SIZE = 256        # Generator crop size (= uniform crop size)
OVL_SIZE = 224        # VFM input size
NUM_REFS_PER_ROW = IMAGE_SIZE // REF_SIZE  # 4
NUM_REFS_PER_IMAGE = NUM_REFS_PER_ROW ** 2  # 16

# 8 directions
DIR_N = 0; DIR_S = 1; DIR_E = 2; DIR_W = 3
DIR_NE = 4; DIR_NW = 5; DIR_SE = 6; DIR_SW = 7

_DIR_VECTORS = {
    DIR_N: (0, -1), DIR_S: (0, +1), DIR_E: (+1, 0), DIR_W: (-1, 0),
    DIR_NE: (+1, -1), DIR_NW: (-1, -1), DIR_SE: (+1, +1), DIR_SW: (-1, +1),
}


def _direction_displacement(direction_id, distance):
    ux, uy = _DIR_VECTORS[direction_id]
    if direction_id in (DIR_NE, DIR_NW, DIR_SE, DIR_SW):
        d = int(distance / math.sqrt(2))
        return (ux * d, uy * d)
    else:
        return (ux * distance, uy * distance)


def compute_overlap_region(ref_x, ref_y, ovl_x, ovl_y,
                           ref_size=REF_SIZE, ovl_size=OVL_SIZE):
    o_left = max(ref_x, ovl_x)
    o_top = max(ref_y, ovl_y)
    o_right = min(ref_x + ref_size, ovl_x + ovl_size)
    o_bottom = min(ref_y + ref_size, ovl_y + ovl_size)

    if o_left >= o_right or o_top >= o_bottom: return None
    
    return {'ref_crop': (o_left - ref_x, o_top - ref_y, o_right - ref_x, o_bottom - ref_y),
            'ovl_crop': (o_left - ovl_x, o_top - ovl_y, o_right - ovl_x, o_bottom - ovl_y),}


class SheafTestDataset(BaseDataset):
    """BCI test dataset with uniform 4×4 tiling and cross-grid overlap extraction.

    Each __getitem__ returns one ref patch (256²) + surrounding overlapped
    patches (224²) extracted from the FULL 1024×1024 image (not grid-limited).

    Returns 16 items per image (4×4 uniform crop).
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # sheaf_stride, sheaf_num_jitter_vh, sheaf_jitter_range, sheaf_test_cache_dir
        # defined in base_options.py
        parser.add_argument('--min_tissue_ratio', type=float, default=0.0, help='No tissue filtering at inference (process all patches)')
        parser.add_argument('--return_overlap_patches', action='store_true')

        # Dataset structure selectors (consumed by _load_paths)
        parser.add_argument('--stain', type=str, default='her2', choices=['her2', 'ki67', 'er', 'pr'], help='IHC stain subdirectory under image/psi/{he,ihc}/{stain}')
        parser.add_argument('--train_split_mode', type=str, default='bci', choices=['bci', 'mist'], help='Split policy selector (mirrors sheaf train dataset)')
        parser.add_argument('--img_ext', type=str, default='.png', help='Image file extension')
        
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.stride = getattr(opt, 'sheaf_stride', 80)
        self.sheaf_cond_dim = getattr(opt, 'sheaf_cond_dim', 256)
        self.return_overlap_patches = getattr(opt, 'return_overlap_patches', False)

        # Global CLS cache: {image_idx: tensor[1536]}
        self._global_cls_cache = {}

        # Jitter matching training (deterministic via hash-based RNG)
        self.num_jitter_vh = getattr(opt, 'sheaf_num_jitter_vh', 2)
        self.jitter_range = getattr(opt, 'sheaf_jitter_range', 32)

        # max_overlaps: cardinal × (1 + jitter) + diagonal × 1
        self.patches_per_cardinal = 1 + self.num_jitter_vh
        self.max_overlaps = 4 * self.patches_per_cardinal + 4

        # Load test image paths
        self._load_paths(opt)

        # Transform: [-1, 1] for generator
        from torchvision import transforms
        self.ref_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Sheaf conditioning cache
        self._sheaf_cache = {}
        self._preset_lookup = None
        self._sheaf_spatial = getattr(opt, 'sheaf_spatial', False)

        is_rank0 = not getattr(opt, 'use_ddp', False) or getattr(opt, 'rank', 0) == 0
        if is_rank0:
            print(f"[SheafTestDataset] {self.num_images} test images, "
                  f"{NUM_REFS_PER_IMAGE} patches/image, "
                  f"stride={self.stride}, cross-grid overlap")

    def _load_paths(self, opt):
        import pandas as pd
        stain = getattr(opt, 'stain', 'her2')
        img_ext = getattr(opt, 'img_ext', '.png')
        train_split_mode = getattr(opt, 'train_split_mode', 'bci')
        dir_A = os.path.join(opt.dataroot, 'image', 'psi', 'he', stain)
        dir_B = os.path.join(opt.dataroot, 'image', 'psi', 'ihc', stain)
        labels_path = os.path.join(opt.dataroot, 'label', 'psi', stain, 'labels.csv')

        df = pd.read_csv(labels_path)
        # infer_split (config): 'train' | 'val' | 'test' | 'trainval' | 'train,val' | 'all' …
        infer_split_env = str(getattr(opt, 'infer_split', '') or '').strip()
        if infer_split_env:
            sp = infer_split_env.lower()
            if sp == 'all':
                pass  # keep full df
            elif sp == 'trainval':
                df = df[df['split'].isin(['train', 'val'])]
            elif ',' in sp:
                splits = [s.strip() for s in sp.split(',') if s.strip()]
                df = df[df['split'].isin(splits)]
            else:
                df = df[df['split'] == sp]
            test_key = sp  # for diagnostic
        else:
            # BCI 'test' split convention vs MIST 'val' split convention.
            test_key = 'val' if train_split_mode == 'mist' else 'test'
            df = df[df['split'] == test_key]

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
            raise ValueError(
                f"No test images found (stain={stain!r} "
                f"ext={img_ext!r} split_key={test_key!r} "
                f"dataroot={opt.dataroot!r})")

    def __len__(self):
        return self.num_images * NUM_REFS_PER_IMAGE

    def __getitem__(self, index):
        image_idx = index // NUM_REFS_PER_IMAGE
        patch_idx = index % NUM_REFS_PER_IMAGE

        # Uniform 4×4 tiling position
        row = patch_idx // NUM_REFS_PER_ROW
        col = patch_idx % NUM_REFS_PER_ROW
        ref_x = col * REF_SIZE  # image coordinates
        ref_y = row * REF_SIZE

        # Load full images
        he_img = Image.open(self.A_paths[image_idx]).convert('RGB')
        ihc_img = Image.open(self.B_paths[image_idx]).convert('RGB')

        # Extract ref patch
        ref_he = he_img.crop((ref_x, ref_y, ref_x + REF_SIZE, ref_y + REF_SIZE))
        ref_ihc = ihc_img.crop((ref_x, ref_y, ref_x + REF_SIZE, ref_y + REF_SIZE))

        # Extract overlapped patches (cross-grid: from FULL image)
        ovl_result = self._extract_overlapped_cross_grid(
            he_img, ref_x, ref_y)

        # Transforms
        ref_he_t = self.ref_transform(ref_he)
        ref_ihc_t = self.ref_transform(ref_ihc)

        # Grid tensors for logging (512×512 grid containing this ref)
        from torchvision import transforms
        grid_to_tensor = transforms.ToTensor()
        gx = (ref_x // 512) * 512
        gy = (ref_y // 512) * 512
        grid_he = he_img.crop((gx, gy, gx + 512, gy + 512))
        grid_ihc = ihc_img.crop((gx, gy, gx + 512, gy + 512))
        grid_he_t = grid_to_tensor(grid_he)
        grid_ihc_t = grid_to_tensor(grid_ihc)

        # Pack overlaps (fixed-size, padded)
        num_ovl = len(ovl_result['patches'])
        overlap_valid = torch.zeros(self.max_overlaps)
        overlap_valid[:num_ovl] = 1.0
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

        result = {
            'A': ref_he_t,
            'B': ref_ihc_t,
            'A_paths': self.A_paths[image_idx],
            'B_paths': self.B_paths[image_idx],
            'grid_he': grid_he_t,
            'grid_ihc': grid_ihc_t,
            'sheaf_cond': self._sheaf_cache.get(
                (image_idx, patch_idx),
                torch.zeros(1536, 16, 16) if getattr(self, '_sheaf_spatial', False)
                else torch.zeros(self.sheaf_cond_dim)),
            'global_cls': self._global_cls_cache.get(
                image_idx, torch.zeros(1536)),
            'overlap_valid': overlap_valid,
            'overlap_positions': overlap_positions,
            'overlap_directions': overlap_directions,
            'overlap_offsets': overlap_offsets,
            'overlap_regions_ref': overlap_regions_ref,
            'overlap_regions_ovl': overlap_regions_ovl,
            'ref_position': torch.tensor([ref_x, ref_y], dtype=torch.float32),
            'patch_idx': patch_idx,
            'image_idx': image_idx,
            'num_overlaps': num_ovl,
        }

        if self.return_overlap_patches:
            from torchvision import transforms
            ovl_to_tensor = transforms.ToTensor()
            ovl_tensor = torch.zeros(self.max_overlaps, 3, OVL_SIZE, OVL_SIZE)
            for i, patch in enumerate(ovl_result['patches']):
                ovl_tensor[i] = ovl_to_tensor(patch)
            result['overlap_patches'] = ovl_tensor

        return result

    def _extract_overlapped_cross_grid(self, full_image, ref_x, ref_y):
        """Extract overlapped patches from FULL 1024×1024 image (cross-grid).

        Unlike training (grid-limited), inference extracts from the entire
        image. Only actual image boundaries limit extraction.
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

            # Bounds check against FULL IMAGE (not grid)
            if not (0 <= ox and ox + OVL_SIZE <= IMAGE_SIZE and
                    0 <= oy and oy + OVL_SIZE <= IMAGE_SIZE):
                return False

            region = compute_overlap_region(ref_x, ref_y, ox, oy)
            if region is None:
                return False

            patch = full_image.crop((ox, oy, ox + OVL_SIZE, oy + OVL_SIZE))
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

        # Cardinal (N, S, E, W): center + jitter (matching training)
        for direction_id in [DIR_N, DIR_S, DIR_E, DIR_W]:
            main_dx, main_dy = _direction_displacement(direction_id, self.stride)
            _try_extract(direction_id, main_dx, main_dy)

            # Deterministic jitter (seeded by ref position + direction)
            # Clamp jitter to valid range so edge/corner patches always
            # extract maximum overlapped patches.
            for j in range(self.num_jitter_vh):
                seed = int(hashlib.md5(
                    f"{ref_x}_{ref_y}_{direction_id}_{j}".encode()
                ).hexdigest()[:8], 16)
                jitter_rng = random.Random(seed)
                jitter = jitter_rng.randint(-self.jitter_range, self.jitter_range)

                half_ovl = OVL_SIZE // 2  # 112
                if direction_id in (DIR_N, DIR_S):
                    # Jitter on dx (horizontal)
                    jcx = ref_cx + main_dx + jitter
                    min_j = max(-self.jitter_range, half_ovl - (ref_cx + main_dx))
                    max_j = min(self.jitter_range, (IMAGE_SIZE - half_ovl) - (ref_cx + main_dx))
                    jitter = max(min_j, min(jitter, max_j))
                    _try_extract(direction_id, main_dx + jitter, main_dy,
                                 jitter_offset=jitter)
                else:
                    # Jitter on dy (vertical)
                    jcy = ref_cy + main_dy + jitter
                    min_j = max(-self.jitter_range, half_ovl - (ref_cy + main_dy))
                    max_j = min(self.jitter_range, (IMAGE_SIZE - half_ovl) - (ref_cy + main_dy))
                    jitter = max(min_j, min(jitter, max_j))
                    _try_extract(direction_id, main_dx, main_dy + jitter,
                                 jitter_offset=jitter)

        # Diagonal (NE, NW, SE, SW)
        for direction_id in [DIR_NE, DIR_NW, DIR_SE, DIR_SW]:
            dx, dy = _direction_displacement(direction_id, self.stride)
            _try_extract(direction_id, dx, dy)

        return {'patches': patches, 'metadata': metadata}

    def set_sheaf_cache(self, cache_dict):
        self._sheaf_cache = cache_dict

    def set_global_cls_cache(self, cls_dict):
        """Set global CLS token cache (image-level).

        Args:
            cls_dict: {image_idx: tensor[1536]}
        """
        self._global_cls_cache = cls_dict

    def set_preset_lookup(self, lookup):
        self._preset_lookup = lookup

    def clear_sheaf_cache(self):
        self._sheaf_cache = {}
        self._preset_lookup = None

    def get_all_overlap_patches(self, image_idx, patch_idx):
        """Extract overlap patches for cache computation."""
        row = patch_idx // NUM_REFS_PER_ROW
        col = patch_idx % NUM_REFS_PER_ROW
        ref_x = col * REF_SIZE
        ref_y = row * REF_SIZE

        he_img = Image.open(self.A_paths[image_idx]).convert('RGB')
        ovl_result = self._extract_overlapped_cross_grid(he_img, ref_x, ref_y)

        return {
            'patches': ovl_result['patches'],
            'metadata': ovl_result['metadata'],
            'ref_position': (ref_x, ref_y),
        }
