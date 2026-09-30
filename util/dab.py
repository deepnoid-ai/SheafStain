import torch

import torch.nn.functional as F


class DABExtractor:
    """Extract DAB stain intensity from IHC images using color deconvolution.

    Uses the Ruifrok & Johnston H-DAB stain matrix with softplus smoothing
    for differentiable training loss computation.
    """

    def __init__(self, device='cuda'):
        self.device = device

        # Standard H-DAB stain matrix (Ruifrok & Johnston)
        # Each row is a stain vector in RGB optical density space
        self.stain_matrix = torch.tensor([[0.268, 0.570, 0.776],  # DAB (brown)
                                          [0.650, 0.704, 0.286],  # Hematoxylin (blue)
                                         ], device=device, dtype=torch.float32)

        # Pseudo-inverse for deconvolution: [3, 2]
        self.deconv_matrix = torch.linalg.pinv(self.stain_matrix.T)

    def rgb_to_od(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Convert RGB [0,1] to optical density: OD = -log10(I/I0)."""
        rgb_images = rgb_images.clamp(1e-6, 1.0)
        
        return -torch.log10(rgb_images + 1e-6)

    def extract_dab_intensity(self, images: torch.Tensor, normalize: str = "max") -> torch.Tensor:
        """Extract DAB stain intensity from IHC images.

        Args:
            images: [B, 3, H, W] RGB images in [-1, 1] or [0, 1]
            normalize: "none", "max", or "meanstd"

        Returns:
            dab_intensity: [B, H, W] DAB intensity map
        """
        B, C, H, W = images.shape
        assert C == 3, "Input must be RGB images"

        # Auto-convert [-1, 1] -> [0, 1] if needed
        if images.min() < 0: images = (images + 1.0) / 2.0

        od = self.rgb_to_od(images)
        od_flat = od.permute(0, 2, 3, 1).reshape(-1, 3)

        # Ensure deconv_matrix is on same device as input
        deconv_matrix = self.deconv_matrix.to(od_flat.device)

        # Deconvolve: concentrations = OD @ M_inv^T
        concentrations = od_flat @ deconv_matrix.T
        dab_flat = concentrations[:, 0]  # DAB channel

        dab_intensity = dab_flat.reshape(B, H, W)

        # Softplus for smooth gradients (beta=5.0 for sharper transition)
        dab = F.softplus(dab_intensity, beta=5.0)

        if normalize == "max" or normalize is True:
            mx = dab.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            dab = dab / mx

        elif normalize == "meanstd":
            mean = dab.mean(dim=(1, 2), keepdim=True)
            std = dab.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            dab = (dab - mean) / std

        elif normalize == "none" or normalize is False: pass
        else: raise ValueError(f"Unknown normalization: {normalize}")

        return dab
