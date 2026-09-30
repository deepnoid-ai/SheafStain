import os
import torch

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torchvision.utils as vutils    

from PIL import Image

try:
    import wandb
    HAS_WANDB = True

except ImportError: 
    HAS_WANDB = False


def tensor2numpy(tensor, denorm=True):
    """Convert tensor to numpy image (H, W, C) in [0, 255].

    Args:
        tensor: [C, H, W] or [B, C, H, W] tensor in [-1, 1] (if denorm=True)
        denorm: if True, converts from [-1,1] to [0,255]
    """
    if tensor.dim() == 4: tensor = tensor[0]
    img = tensor.detach().cpu().float()
    
    if denorm: img = (img + 1.0) / 2.0  # [-1,1] → [0,1]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    
    return (img * 255).astype(np.uint8)


def make_patch_grid(patches_tensor, valid_mask, grid_rows=4, grid_cols=4, patch_size=224):
    """Arrange patches into a single grid image. Fill invalid slots with white.

    Args:
        patches_tensor: [max_ovl, 3, H, W] raw patches (in [0,1], no denorm)
        valid_mask: [max_ovl] float tensor (1.0=valid, 0.0=pad)
        grid_rows, grid_cols: grid layout
        patch_size: display size per patch

    Returns:
        numpy array (grid_H, grid_W, 3) uint8
    """
    n_slots = grid_rows * grid_cols
    grid_img = np.ones((grid_rows * patch_size, grid_cols * patch_size, 3), dtype=np.uint8) * 255  # white background

    for idx in range(min(n_slots, patches_tensor.shape[0])):
        if valid_mask[idx] < 0.5: continue

        patch = patches_tensor[idx].detach().cpu().float()
        patch = patch.clamp(0, 1).permute(1, 2, 0).numpy()
        patch = (patch * 255).astype(np.uint8)

        # Resize if needed
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size: patch = np.array(Image.fromarray(patch).resize((patch_size, patch_size), Image.BILINEAR))

        row = idx // grid_cols
        col = idx % grid_cols

        grid_img[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patch

    return grid_img


def cond_to_heatmap(cond_vector, size=16):
    """Convert conditioning vector to 2D heatmap image.

    Args:
        cond_vector: [cond_dim] tensor
        size: reshape target (size × size). If cond_dim != size², pads/truncates.

    Returns:
        numpy array (size, size, 3) uint8 heatmap
    """
    vec = cond_vector.detach().cpu().float().numpy()
    target_len = size * size

    # Pad or truncate
    if len(vec) < target_len: vec = np.pad(vec, (0, target_len - len(vec)), constant_values=0)
    elif len(vec) > target_len: vec = vec[:target_len]

    mat = vec.reshape(size, size)

    # Normalize to [0, 1]
    vmin, vmax = mat.min(), mat.max()
    if vmax - vmin > 1e-8: mat = (mat - vmin) / (vmax - vmin)
    else: mat = np.zeros_like(mat)

    # Apply colormap
    cmap = cm.get_cmap('viridis')
    heatmap = (cmap(mat)[:, :, :3] * 255).astype(np.uint8)

    # Upscale for visibility
    heatmap_pil = Image.fromarray(heatmap).resize((size * 16, size * 16), Image.NEAREST)

    return np.array(heatmap_pil)


class WandBLogger:
    """WandB logging wrapper for SheafStain training.

    Args:
        opt: training options
        enabled: override to disable (e.g., non-rank-0 in DDP)
    """

    def __init__(self, opt, enabled=True):
        self.opt = opt
        self.enabled = enabled and HAS_WANDB

        if not self.enabled:
            if not HAS_WANDB and enabled: print("[WandBLogger] wandb not installed. Logging disabled.")

            return

        try:
            # Authenticate: env var (Docker/K8s) > file (local) > netrc
            api_key = os.environ.get('WANDB_API_KEY', '')

            if api_key:
                wandb.login(key=api_key)
                print("[WandBLogger] Authenticated via WANDB_API_KEY env var")
            
            else:
                api_key_file = getattr(opt, 'wandb_api_key_file', None)
                if api_key_file is not None and os.path.exists(api_key_file):
                    with open(api_key_file, 'r') as f: api_key = f.read().strip()
                    wandb.login(key=api_key)
                    print(f"[WandBLogger] Authenticated via {api_key_file}")

            # Initialize wandb
            config = {k: v for k, v in vars(opt).items() if not k.startswith('_') and k != 'visualizer'}

            wandb.init(project=getattr(opt, 'wandb_project', 'SheafStain'),
                       name=getattr(opt, 'name', 'experiment'),
                       config=config,
                       dir=os.path.join(opt.checkpoints_dir, opt.name),
                       resume='allow',
                       settings=wandb.Settings(init_timeout=300),)

            print(f"[WandBLogger] Initialized: project={wandb.run.project}, name={wandb.run.name}")

        except Exception as e:
            self.enabled = False
            print(f"[WandBLogger] Setup failed ({type(e).__name__}: {e}). Logging disabled; training continues. Set WANDB_API_KEY or wandb_api_key_file, or run `wandb login`, to enable it.")

    def log_scalars(self, epoch, losses, step):
        """Log scalar values (losses).

        Args:
            epoch: current epoch
            losses: OrderedDict of {loss_name: value}
            step: global step counter
        """
        if not self.enabled: return

        log_dict = {'epoch': epoch}
        for name, value in losses.items(): log_dict[f'loss/{name}'] = value

        # Total G and D losses (if available)
        if 'G' in losses: log_dict['loss/G_total'] = losses['G']

        # Two-phase tag: phase=1 (no VFM-FM) -> phase=2 (VFM-FM on) at
        # --vfm_fm_start_epoch. Tag is omitted if the option is not set
        # (i.e. legacy single-phase runs).
        fm_start = getattr(self.opt, 'vfm_fm_start_epoch', None)
        lam_fm = getattr(self.opt, 'lambda_vfm_fm', 0)

        if fm_start is not None and lam_fm and lam_fm > 0: log_dict['phase'] = 1 if epoch < fm_start else 2

        wandb.log(log_dict, step=step)

    def log_images(self, epoch, model, data, step):
        """Log images to wandb.

        Args:
            epoch: current epoch
            model: SBModel instance
            data: batch dict from SheafDataset
            step: global step counter
        """
        if not self.enabled: return

        images = {}

        # 1. Grid: Source H&E + Target IHC (512×512 side-by-side)
        if 'grid_he' in data and 'grid_ihc' in data:
            grid_he = data['grid_he']
            grid_ihc = data['grid_ihc']

            if grid_he.dim() == 4:
                grid_he = grid_he[0]
                grid_ihc = grid_ihc[0]

            he_img = tensor2numpy(grid_he, denorm=False)
            ihc_img = tensor2numpy(grid_ihc, denorm=False)
            grid_pair = np.concatenate([he_img, ihc_img], axis=1)
            
            images['images/grid'] = wandb.Image(grid_pair, caption=f'Left: H&E grid (512²), Right: IHC grid (epoch {epoch})')

        # 2. Reference Patch: H&E + IHC (side-by-side)
        if 'A' in data and 'B' in data:
            ref_he = tensor2numpy(data['A'])
            ref_ihc = tensor2numpy(data['B'])
            ref_pair = np.concatenate([ref_he, ref_ihc], axis=1)
            
            images['images/ref_patch'] = wandb.Image(ref_pair, caption='Left: H&E ref (256²), Right: IHC ref')

        # 3. Overlapped Patches: 4×4 grid (white fill for empty slots)
        if 'overlap_patches' in data and 'overlap_valid' in data:
            patches = data['overlap_patches']
            valid = data['overlap_valid']
            
            if patches.dim() == 5:
                patches = patches[0]
                valid = valid[0]
            
            grid_img = make_patch_grid(patches, valid)
            n_valid = int(valid.sum().item())

            images['images/ovl_patch'] = wandb.Image(grid_img, caption=f'Overlapped patches ({n_valid}/{patches.shape[0]} valid)')

        # 4. Generated Image (fake_B + real_B side-by-side)
        if hasattr(model, 'fake_B'):
            gen_img = tensor2numpy(model.fake_B)
            if hasattr(model, 'real_B'):
                target_img = tensor2numpy(model.real_B)
                gen_pair = np.concatenate([gen_img, target_img], axis=1)

                images['generated/fake_vs_target'] = wandb.Image(gen_pair, caption=f'Left: Generated, Right: Target (epoch {epoch})')
            
            else: images['generated/fake_B'] = wandb.Image(gen_img, caption=f'Generated IHC (epoch {epoch})')

        # 5. Conditioning Heatmap (disabled for now)
        # if 'sheaf_cond' in data:
        #     cond = data['sheaf_cond']
        #     if cond.dim() == 2:
        #         cond = cond[0]
        #     if cond.abs().sum() > 0:
        #         heatmap = cond_to_heatmap(cond, size=16)
        #         images['conditioning/heatmap'] = wandb.Image(
        #             heatmap, caption='Sheaf conditioning (16×16 reshape)')

        if images: wandb.log(images, step=step)

    def log_test_images(self, epoch, model, step):
        """Log test-phase multi-step generation results.

        Args:
            epoch: current epoch
            model: SBModel instance (after test forward pass)
            step: global step counter
        """
        if not self.enabled: return

        images = {}
        T = self.opt.num_timesteps

        # Input
        if hasattr(model, 'real_A'):
            images['test/input_HE'] = wandb.Image(tensor2numpy(model.real_A), caption='Input H&E')

        # Each timestep
        step_imgs = []
        for t in range(1, T + 1):
            attr = f'fake_{t}'
            if hasattr(model, attr):
                img = tensor2numpy(getattr(model, attr))
                step_imgs.append(img)
                images[f'test/step_{t}'] = wandb.Image(img, caption=f'Step {t}/{T}')

        # All steps side-by-side
        if step_imgs:
            combined = np.concatenate(step_imgs, axis=1)
            images['test/all_steps'] = wandb.Image(combined, caption=f'Steps 1→{T} (epoch {epoch})')

        # Ground truth
        if hasattr(model, 'real_B'): images['test/target_IHC'] = wandb.Image(tensor2numpy(model.real_B), caption='Target IHC (GT)')
        if images: wandb.log(images, step=step)

    def log_eval_metrics(self, epoch, metrics, step):
        """Log evaluation metrics (FID, SSIM, etc.) at epoch level.

        Args:
            epoch: current epoch
            metrics: dict of {metric_name: value}
            step: global step counter
        """
        if not self.enabled: return

        log_dict = {'epoch': epoch}
        for name, value in metrics.items(): log_dict[f'eval/{name}'] = value

        wandb.log(log_dict, step=step)

    def finish(self):
        """Finalize wandb run."""
        if self.enabled: wandb.finish()
