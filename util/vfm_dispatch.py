import os
import json
import timm
import torch

import torch.nn.functional as F

from dataclasses import dataclass
from models.sheaf_modules import load_gigapath
from timm.layers import SwiGLUPacked


TARGET_TOKEN_GRID = 14
TARGET_NUM_TOKENS = TARGET_TOKEN_GRID * TARGET_TOKEN_GRID  # 196

@dataclass
class VFMSpec:
    """Static description of a VFM's geometry/layout for the wrapper."""
    name: str                 # 'gigapath' | 'uni2h' | 'virchow2'
    img_size: int             # VFM input resolution (typically 224)
    patch_size: int           # ViT patch size (16 for gigapath, 14 for uni2h/virchow2)
    embed_dim: int            # 1536 (gigapath / uni2h) or 1280 (virchow2)
    num_prefix_tokens: int    # 1 + register tokens: gigapath 1, virchow2 5, uni2h 9

    @property
    def native_grid(self) -> int:
        return self.img_size // self.patch_size


class VFMWrapper:
    """Uniform interface across pathology VFMs.

    `extract(x)` runs a single VFM forward pass and splits the output into
    the CLS token and the per-patch token grid. If the native grid differs
    from `TARGET_TOKEN_GRID` (=14), the patch tokens are bilinearly
    resampled in the spatial dim so that downstream code can keep its
    14x14 / token_px=16 assumption.
    """

    def __init__(self, model: torch.nn.Module, spec: VFMSpec):
        self.model = model
        self.spec = spec

    @torch.no_grad()
    def extract(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, img_size, img_size] (already ImageNet-normalized).
        Returns:
            cls_tokens:   [B, D]
            patch_tokens: [B, TARGET_NUM_TOKENS=196, D]
        """
        feats = self.model.forward_features(x)        # [B, num_prefix + N_native, D]
        D = self.spec.embed_dim

        if feats.shape[-1] != D: raise RuntimeError(f"VFM '{self.spec.name}' returned embed_dim={feats.shape[-1]}, expected {D} per VFMSpec.")

        cls = feats[:, 0]                              # [B, D]
        patch = feats[:, self.spec.num_prefix_tokens:] # [B, N_native, D]

        native = self.spec.native_grid

        if patch.shape[1] != native * native: raise RuntimeError(f"VFM '{self.spec.name}' patch-token count ({patch.shape[1]}) != native_grid^2 ({native*native}). Check num_prefix_tokens={self.spec.num_prefix_tokens}.")
        
        B = patch.shape[0]
        patch = patch.reshape(B, native, native, D)

        if native != TARGET_TOKEN_GRID:
            patch_bcwh = patch.permute(0, 3, 1, 2).contiguous()
            patch_bcwh = F.interpolate(patch_bcwh, size=(TARGET_TOKEN_GRID, TARGET_TOKEN_GRID), mode='bilinear', align_corners=False)
            patch = patch_bcwh.permute(0, 2, 3, 1).contiguous()

        patch = patch.reshape(B, TARGET_NUM_TOKENS, D)

        return cls, patch


def _try_load_local_config(model_path: str):
    """Optional config.json next to pytorch_model.bin. None if absent."""
    cfg_path = os.path.join(model_path, 'config.json')
    
    if not os.path.exists(cfg_path): return None
    with open(cfg_path) as f: return json.load(f)


def _load_state_dict_strict_or_warn(model, weights_path):
    """Load state dict with diagnostic on missing/unexpected/shape-mismatched
    keys. Aborts if any shape mismatch is detected (the loaded model would
    be unusable); proceeds with strict=False if only key-set differences."""
    sd = torch.load(weights_path, map_location='cpu', weights_only=True)
    
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']

    model_sd = model.state_dict()
    matched, unexpected, mismatched = {}, [], []

    for k, v in sd.items():
        if k not in model_sd: unexpected.append(k)
        elif model_sd[k].shape != v.shape: mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else: matched[k] = v
    
    missing = [k for k in model_sd if k not in matched]
    model.load_state_dict(matched, strict=False)

    if missing or unexpected or mismatched:
        print(f"  [warn] state_dict load: missing={len(missing)} unexpected={len(unexpected)} shape-mismatched={len(mismatched)}")
        if missing: print(f"    first missing keys: {missing[:5]}")
        if unexpected: print(f"    first unexpected keys: {unexpected[:5]}")
        if mismatched:
            print(f"    first shape-mismatched keys (key, ckpt_shape, model_shape):")
            for k, c, m in mismatched[:8]: print(f"      {k}: ckpt={c}  model={m}")
            raise RuntimeError(f"Aborting: {len(mismatched)} shape mismatches (see above). Model architecture does not match weights.")


def _build_gigapath(model_path: str, device: str) -> VFMWrapper:
    """Reuse the existing `load_gigapath` from sheaf_modules so the GigaPath
    path is byte-identical to the legacy training/inference code."""
    model = load_gigapath(model_path, device)
    cfg = _try_load_local_config(model_path) or {}
    margs = cfg.get('model_args', {})
    
    spec = VFMSpec(name='gigapath',
                   img_size=int(margs.get('img_size', 224)),
                   patch_size=int(margs.get('patch_size', 16)),
                   embed_dim=int(margs.get('embed_dim', 1536)),
                   num_prefix_tokens=1,)

    return VFMWrapper(model, spec)


def _build_uni(model_path: str, device: str) -> VFMWrapper:
    """UNI (v1): ViT-L/16 DINOv2, embed_dim=1024, single CLS token, no
    registers, default GELU MLP. Native grid 14x14 at 224 input (same as
    Prov-GigaPath, so no token-grid resampling needed).

    Per the official `config.json`:
      architecture=vit_large_patch16_224, patch_size=16, embed_dim=1024,
      init_values=1.0, num_features=1024, global_pool='token'.
    """
    cfg = _try_load_local_config(model_path) or {}
    img_size    = int(cfg.get('img_size',    224))
    patch_size  = int(cfg.get('patch_size',  16))
    embed_dim   = int(cfg.get('embed_dim',   cfg.get('num_features', 1024)))
    depth       = int(cfg.get('depth',       24))
    num_heads   = int(cfg.get('num_heads',   16))
    mlp_ratio   = float(cfg.get('mlp_ratio', 4.0))
    init_values = cfg.get('init_values',     1.0)

    model = timm.create_model("vit_large_patch16_224",
                              pretrained=False, num_classes=0,
                              img_size=img_size, patch_size=patch_size,
                              embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                              mlp_ratio=mlp_ratio, init_values=init_values,
                              dynamic_img_size=bool(cfg.get('dynamic_img_size', True)),)
    
    _load_state_dict_strict_or_warn(model, os.path.join(model_path, 'pytorch_model.bin'))
    model = model.to(device).eval()
    spec = VFMSpec(name='uni',
                   img_size=img_size, patch_size=patch_size,
                   embed_dim=embed_dim, num_prefix_tokens=1,)

    return VFMWrapper(model, spec)


def _build_uni2h(model_path: str, device: str) -> VFMWrapper:
    """UNI2-h: ViT-H/14 DINOv2, embed_dim=1536, CLS + 8 register tokens,
    SwiGLU-packed FFN. Native grid 16x16 at 224 input.
    Output token order: [CLS; register x 8; patches]."""
    cfg = _try_load_local_config(model_path) or {}
    img_size    = int(cfg.get('img_size',    224))
    patch_size  = int(cfg.get('patch_size',  14))
    embed_dim   = int(cfg.get('embed_dim',   1536))
    depth       = int(cfg.get('depth',       24))
    num_heads   = int(cfg.get('num_heads',   24))
    mlp_ratio   = float(cfg.get('mlp_ratio', 16.0 / 3.0))
    init_values = cfg.get('init_values',     1e-5)
    reg_tokens  = int(cfg.get('reg_tokens',  8))

    create_kwargs = dict(pretrained=False, num_classes=0,
                         img_size=img_size, patch_size=patch_size,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, init_values=init_values,
                         no_embed_class=True, reg_tokens=reg_tokens,)
    
    try:
        create_kwargs['mlp_layer'] = SwiGLUPacked
        create_kwargs['act_layer'] = torch.nn.SiLU
    
    except Exception as e: print(f"  [warn] SwiGLUPacked unavailable ({e!r}); falling back to default MLP.")

    model = timm.create_model("vit_huge_patch14_224", **create_kwargs)
    _load_state_dict_strict_or_warn(model, os.path.join(model_path, 'pytorch_model.bin'))
    model = model.to(device).eval()
    
    spec = VFMSpec(name='uni2h',
                   img_size=img_size, patch_size=patch_size,
                   embed_dim=embed_dim, num_prefix_tokens=1 + reg_tokens,)

    return VFMWrapper(model, spec)


def _build_virchow2(model_path: str, device: str) -> VFMWrapper:
    """Virchow2: ViT-H/14 with SwiGLU FFN and 4 register tokens.
    embed_dim=1280, depth=32, num_heads=16. Native grid 16x16 at 224.
    Output token order: [CLS; register x 4; patches]."""
    cfg = _try_load_local_config(model_path) or {}
    img_size    = int(cfg.get('img_size',    224))
    patch_size  = int(cfg.get('patch_size',  14))
    embed_dim   = int(cfg.get('embed_dim',   1280))
    depth       = int(cfg.get('depth',       32))
    num_heads   = int(cfg.get('num_heads',   16))
    mlp_ratio   = float(cfg.get('mlp_ratio', 5.3375))
    init_values = cfg.get('init_values',     1e-5)
    reg_tokens  = int(cfg.get('reg_tokens',  4))

    create_kwargs = dict(pretrained=False, num_classes=0,
                         img_size=img_size, patch_size=patch_size,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, init_values=init_values,
                         reg_tokens=reg_tokens, global_pool='',)

    try:
        create_kwargs['mlp_layer'] = SwiGLUPacked
        create_kwargs['act_layer'] = torch.nn.SiLU

    except Exception: pass

    model = timm.create_model("vit_huge_patch14_224", **create_kwargs)
    _load_state_dict_strict_or_warn(model, os.path.join(model_path, 'pytorch_model.bin'))
    model = model.to(device).eval()

    spec = VFMSpec(name='virchow2',
                   img_size=img_size, patch_size=patch_size,
                   embed_dim=embed_dim,
                   num_prefix_tokens=1 + reg_tokens,)

    return VFMWrapper(model, spec)


_VFM_BUILDERS = {'gigapath': _build_gigapath,
                 'uni':      _build_uni,
                 'uni2h':    _build_uni2h,
                 'virchow2': _build_virchow2,}


def _canonical_vfm_name(name: str) -> str:
    key = name.strip().lower().replace('-', '').replace('_', '')
    # looks not good but.. idk
    aliases = {'gigapath': 'gigapath', 'provgigapath': 'gigapath',
               'uni': 'uni', 'uni1': 'uni', 'unimass100k': 'uni',
               'uni2h': 'uni2h', 'uni2': 'uni2h', 'unih': 'uni2h',
               'virchow2': 'virchow2', 'virchow': 'virchow2',}

    canonical = aliases.get(key)
    if canonical is None: raise ValueError(f"Unknown VFM name: {name!r}. Choices: {list(_VFM_BUILDERS)}.")
    
    return canonical


def load_vfm(name: str, model_path: str, device: str) -> VFMWrapper:
    """Dispatch a VFM name to the matching builder and return a VFMWrapper."""
    canonical = _canonical_vfm_name(name)
    
    return _VFM_BUILDERS[canonical](model_path, device)
