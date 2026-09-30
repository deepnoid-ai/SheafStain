import os
import torch

import numpy as np
import util.util as util
import torch.nn.functional as F

from . import networks
from .base_model import BaseModel
from .patchnce import PatchNCELoss
from .sheaf_modules import load_gigapath
from .dab_loss import dab_intensity_loss
from .fourier_loss import fourier_edge_loss
from util.vfm_dispatch import load_vfm, TARGET_TOKEN_GRID
from torch.utils.checkpoint import checkpoint as _grad_checkpoint


class SBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--mode', type=str, default="sb", choices=['FastCUT', 'fastcut', 'sb'])

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_SB', type=float, default=0.1, help='weight for SB loss')
        
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch', type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--lmda', type=float, default=0.1)
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance', type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--lambda_sheaf_feat', type=float, default=0.0, help='Weight for multi-scale feature-space sheaf loss. 0=disabled (pixel sheaf only)')
        parser.add_argument('--sb_cost', type=str, default='vfm_cosine', choices=['pixel_mse', 'vfm_cosine'],
                            help='SB transport cost metric: pixel_mse (UNSB original) or vfm_cosine (VL-SB, Prov-GigaPath cosine).')
        
        # NOTE: --lambda_vfm_fm is defined in options/base_options.py alongside
        # other auxiliary loss weights (lambda_dab_intensity, lambda_fourier_edge, etc.)

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.mode.lower() == "sb": parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.mode.lower() == "fastcut": parser.set_defaults(nce_idt=False, lambda_NCE=10.0, flip_equivariance=True, n_epochs=150, n_epochs_decay=50)
        else: raise ValueError(opt.mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # DDP static_graph safety flags (set per-iteration in set_input)
        self._adj_is_dummy = False
        self._adj2_is_dummy = False

        # VL-SB / VFM-D: VFM (Prov-GigaPath) handle. Set eagerly below in the
        # isTrain branch to avoid DDP static_graph issues with lazy loading.
        self._vfm = None
        # Cache for fake_B's VFM patch grid, populated in compute_G_loss and
        # reused by compute_D_loss to avoid a redundant ViT-giant forward.
        self._cached_fake_vfm_grid = None

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'SB']
        if getattr(opt, 'lambda_sheaf', 0) > 0: self.loss_names += ['sheaf']
        if getattr(opt, 'lambda_sheaf_feat', 0) > 0: self.loss_names += ['sheaf_feat']
        if getattr(opt, 'lambda_sheaf_cocycle', 0) > 0: self.loss_names += ['sheaf_cocycle']
        if getattr(opt, 'lambda_fourier_edge', 0) > 0: self.loss_names += ['fourier_edge']
        
        # DAB extractor: needed by lambda_dab_intensity AND/OR lambda_vfm_fm.
        # We initialize once if either is active.
        _needs_dab_extractor = (getattr(opt, 'lambda_dab_intensity', 0) > 0
                                or getattr(opt, 'lambda_vfm_fm', 0) > 0)

        if getattr(opt, 'lambda_dab_intensity', 0) > 0: self.loss_names += ['dab_intensity']
        if getattr(opt, 'lambda_vfm_fm', 0) > 0: self.loss_names += ['vfm_fm']

        if _needs_dab_extractor:
            from util.dab import DABExtractor
            self.dab_extractor = DABExtractor(device='cpu')

        self.visual_names = ['real_A','real_A_noisy', 'fake_B', 'real_B']

        if self.opt.phase == 'test':
            self.visual_names = ['real']

            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain: self.model_names = ['G', 'F', 'D','E']
        else: self.model_names = ['G'] # during inference only load G

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # netE (energy network) takes a 6-channel concat (X_t, X_1) and uses
            # input2 for a contrastive negative pair (also 6 channels). Its input2
            # semantics is incompatible with vfm_cond's VFM-grid input, so when
            # netD = 'vfm_cond' we force netE back to basic_cond.
            netE_type = 'basic_cond' if opt.netD == 'vfm_cond' else opt.netD
            self.netE = networks.define_D(opt.output_nc*4, opt.ndf, netE_type, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # VL-SB / VFM-D: eager VFM (Prov-GigaPath) load (P1, DDP-safe).
            # Loading inside forward would change the computation graph between
            # iterations under DDP static_graph; loading here makes the graph
            # consistent across all iterations.
            needs_vfm = (getattr(opt, 'sb_cost', 'pixel_mse') == 'vfm_cosine'
                         or opt.netD == 'vfm_cond'
                         or getattr(opt, 'lambda_vfm_fm', 0) > 0)

            if needs_vfm:
                # Dispatch on --vfm_name (gigapath / uni2h / virchow2). For the
                # legacy default 'gigapath', this is byte-identical to the prior
                # `load_gigapath(...)` path. UNI2-h/Virchow2 use proper timm arch
                # init (SwiGLU, reg_tokens, no_embed_class) via the dispatcher.
                _vfm_name = getattr(opt, 'vfm_name', 'gigapath')
                _vfm_path = getattr(opt, 'vfm_model_path', None)

                assert _vfm_path, 'vfm_model_path is not set (configure it in config.yaml or pass --vfm_model_path).'

                _wrapper = load_vfm(_vfm_name, _vfm_path, device=self.device)
                self._vfm = _wrapper.model
                self._vfm_spec = _wrapper.spec

                for p in self._vfm.parameters(): p.requires_grad_(False)

                # L2: precompute ImageNet normalization stats once.
                self._vfm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                self._vfm_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                
                # torch.compile on the function we actually call (forward_features).
                # Default mode + auto-dynamic; safe with autocast and checkpoint.
                # Falls back to uncompiled if torch.compile is unavailable or fails.
                self._vfm_features_fn = self._vfm.forward_features
                
                try: self._vfm_features_fn = torch.compile(self._vfm.forward_features)
                except Exception as e: print(f"[SheafStain] torch.compile on VFM failed, using uncompiled: {e}")

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers: self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)
            
    def _dab_to_rgb(self, dab_intensity):
        """Render DAB intensity map as a brown-tinted RGB image via Beer-Lambert.

        Used by VFM-FM (DAB-channel feature distillation): instead of feeding the
        raw 1-channel DAB intensity (out-of-distribution for an RGB-trained VFM),
        we reconstruct the RGB appearance of an IHC sample with DAB chromogen
        only (no hematoxylin), which matches what Prov-GigaPath saw during
        dual-stain pretraining. Beer-Lambert: I = I_0 * 10^(-OD), with OD =
        intensity * stain_DAB.

        Args:
            dab_intensity: [B, H, W] DAB intensity map in [0, 1] (DABExtractor
                output with normalize='max').

        Returns:
            [B, 3, H, W] RGB image in [-1, 1] (matches generator output / VFM
            input convention used elsewhere in this model).
        """
        # Ruifrok H-DAB stain vector for DAB (brown), in optical-density space.
        stain_dab = torch.tensor([0.268, 0.570, 0.776], device=dab_intensity.device, dtype=dab_intensity.dtype,).view(1, 3, 1, 1)
        od = dab_intensity.unsqueeze(1) * stain_dab          # [B, 3, H, W]
        rgb01 = torch.pow(10.0, -od).clamp(0.0, 1.0)         # Beer-Lambert
        
        return rgb01 * 2.0 - 1.0                              # → [-1, 1]

    def _vfm_normalize(self, x):
        """Common preprocessing for VFM forward: denorm + ImageNet stats + 224x224 resize."""
        x = (x + 1.0) * 0.5
        x = (x - self._vfm_mean) / self._vfm_std
        if x.shape[-1] != 224 or x.shape[-2] != 224: x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        
        return x

    def _vfm_features_inner(self, x):
        """Internal: bf16 autocasted VFM forward_features. Returns fp32.

        ``self._vfm_features_fn`` is the (optionally torch.compile-wrapped)
        ``forward_features`` callable bound at ``__init__`` time.
        """
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16): feats = self._vfm_features_fn(x)
        
        return feats.float()

    def _vfm_features_call(self, x):
        """Single entry point for VFM forward_features.

        - torch.compile: handled by ``self._vfm_features_fn`` (set in __init__).
        - bf16 autocast: applied inside ``_vfm_features_inner``.

        Note: gradient checkpointing was previously enabled on the grad path
        but caused a torch.compile + use_reentrant=False backward-replay
        BackendCompilerFailed (ViT-giant attention's size-16 uint8 head-index
        constant not faked under FakeTensorMode) when the very first VFM call
        in compute_G_loss is grad-enabled (sb_cost=pixel_mse + vfm_fm path).
        Removed for stability; per-GPU memory cost is ~1 GB per grad-on call
        on ViT-giant (bf16, batch 3), well within H200 budget.
        """
        return self._vfm_features_inner(x)

    def _vfm_split_tokens(self, feats):
        """Spec-aware split of `forward_features` output into (cls, grid).

        Handles:
          - num_prefix_tokens: GigaPath/UNI2-h=1 (CLS only), Virchow2=5 (CLS+4 reg).
          - native grid: GigaPath=14 (no resample), UNI2-h/Virchow2=16 -> bilinear
            resample to 14 so downstream code keeps its 14x14 / token_px=16
            convention (the same convention the spatial presets store).

        Args:
            feats: [B, num_prefix + N_native, D] (raw forward_features output)

        Returns:
            (cls, grid):
                cls:  [B, D]
                grid: [B, D, 14, 14]
        """
        spec = self._vfm_spec
        D = spec.embed_dim
        cls = feats[:, 0]               
                               # [B, D]
        patch = feats[:, spec.num_prefix_tokens:, :]           # [B, N_native, D]
        native = spec.native_grid
        B = patch.shape[0]
        
        patch = patch.reshape(B, native, native, D)
        grid = patch.permute(0, 3, 1, 2).contiguous()          # [B, D, native, native]
        
        if native != TARGET_TOKEN_GRID:
            grid = F.interpolate(grid, size=(TARGET_TOKEN_GRID, TARGET_TOKEN_GRID), mode='bilinear', align_corners=False)

        return cls, grid

    def _vfm_forward(self, x):
        """VL-SB: VFM CLS forward.

        Args:
            x: image tensor [B, 3, H, W] in [-1, 1] range.

        Returns:
            CLS embedding [B, vfm_dim] from frozen VFM (gigapath/uni2h/virchow2).
        """
        if self._vfm is None: raise RuntimeError("VFM is not loaded. Set --sb_cost vfm_cosine or --netD vfm_cond at training time.")
        
        feats = self._vfm_features_call(self._vfm_normalize(x))
        cls, _ = self._vfm_split_tokens(feats)
        
        return cls

    def _vfm_forward_spatial(self, x):
        """VFM-D: VFM patch-token grid.

        Returns:
            Patch-token grid [B, vfm_dim, 14, 14] (CLS/registers excluded,
            native 16x16 resampled to 14x14 for UNI2-h/Virchow2).
        """
        if self._vfm is None: raise RuntimeError("VFM is not loaded. Set --sb_cost vfm_cosine or --netD vfm_cond at training time.")
        
        feats = self._vfm_features_call(self._vfm_normalize(x))
        _, grid = self._vfm_split_tokens(feats)
        
        return grid

    def _vfm_forward_combined(self, x):
        """P2: Single VFM forward returning BOTH CLS and patch grid.

        Used by compute_G_loss when both VL-SB and VFM-D are active to avoid
        a redundant ViT-giant forward on the same input.

        Returns:
            (cls, grid):
                cls:  [B, vfm_dim]
                grid: [B, vfm_dim, 14, 14]
        """
        if self._vfm is None: raise RuntimeError("VFM is not loaded. Set --sb_cost vfm_cosine or --netD vfm_cond at training time.")
        feats = self._vfm_features_call(self._vfm_normalize(x))
        
        return self._vfm_split_tokens(feats)

    def data_dependent_initialize(self, data,data2):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        # Under DDP, each rank already receives its local batch shard,
        # so we use the full local batch without further division.
        if self.use_ddp: bs_per_gpu = data["A"].size(0)
        else: bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        
        self.set_input(data,data2)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        
        if self.opt.isTrain:    
            self.compute_G_loss().backward()
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()  
            
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        
        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()
        
        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample': self.optimizer_F.zero_grad()
        
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        
        if self.opt.netF == 'mlp_sample': self.optimizer_F.step()       
        
    def set_input(self, input, input2=None):

        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        
        if input2 is not None:
            self.real_A2 = input2['A' if AtoB else 'B'].to(self.device)
            self.real_B2 = input2['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # Sheaf conditioning (from SheafDataset / SheafEmbeddingCache)
        # Note: zero tensors are kept as-is (NOT converted to None).
        # DDP static_graph requires consistent forward pass structure.
        # Zero conditioning = no effect with additive injection (0 + x = x).
        if 'sheaf_cond' in input: self.sheaf_cond = input['sheaf_cond'].to(self.device)
        else: self.sheaf_cond = None

        # Adjacent patch for pixel sheaf loss
        raw_has_adj = input.get('has_adj', 0)
        if isinstance(raw_has_adj, torch.Tensor):
            if raw_has_adj.numel() > 1 and raw_has_adj.unique().numel() > 1: raw_has_adj = 0  # Mixed batch
            else: raw_has_adj = raw_has_adj[0].item()
        
        any_adj_needed = (self.opt.lambda_sheaf > 0
                          or getattr(self.opt, 'lambda_sheaf_feat', 0) > 0
                          or getattr(self.opt, 'lambda_sheaf_cocycle', 0) > 0)
        bs = self.real_A.size(0)
        self._adj_is_dummy = False
        
        if raw_has_adj and any_adj_needed:
            # Real adj data
            self.has_adj = 1
            self.adj_A = input['adj_A'].to(self.device)
            self.adj_B = input['adj_B'].to(self.device)
        
            ox = input['adj_offset_x']
            oy = input['adj_offset_y']
        
            self.adj_offset_x = ox.to(self.device) if isinstance(ox, torch.Tensor) \
                else torch.tensor([ox], device=self.device)
            self.adj_offset_y = oy.to(self.device) if isinstance(oy, torch.Tensor) \
                else torch.tensor([oy], device=self.device)
        
            if 'adj_sheaf_cond' in input: self.adj_sheaf_cond = input['adj_sheaf_cond'].to(self.device)
            else: self.adj_sheaf_cond = torch.zeros_like(self.sheaf_cond)
        
        elif not raw_has_adj and any_adj_needed:
            # Dummy adj for DDP static_graph safety:
            # netG forward count must be constant across all iterations.
            # dummy data keeps graph structure identical; loss is zeroed via
            # adj_fake.sum() * 0.0 (maintains DDP parameter tracking).
            self.has_adj = 1
            self._adj_is_dummy = True
            
            self.adj_A = torch.zeros_like(self.real_A)
            self.adj_B = torch.zeros_like(self.real_B)
            
            self.adj_offset_x = torch.zeros(bs, device=self.device)
            self.adj_offset_y = torch.zeros(bs, device=self.device)
            
            self.adj_sheaf_cond = torch.zeros_like(self.sheaf_cond)
        
        else:
            # No sheaf losses active (e.g., M1 vanilla baseline)
            self.has_adj = 0
            self.adj_A = None

        # Global CLS conditioning (image-level, 100% valid, independent of spatial preset)
        # Note: zero tensors kept as-is (DDP static_graph safety).
        # Zero CLS = no effect with additive injection.
        if 'global_cls' in input: self.global_cls = input['global_cls'].to(self.device)
        else: self.global_cls = None

        # Adjacent patch 2 for cocycle loss
        raw_has_adj2 = input.get('has_adj2', 0)
        if isinstance(raw_has_adj2, torch.Tensor):
            if raw_has_adj2.numel() > 1 and raw_has_adj2.unique().numel() > 1: raw_has_adj2 = 0
            else: raw_has_adj2 = raw_has_adj2[0].item()
        
        lambda_cc = getattr(self.opt, 'lambda_sheaf_cocycle', 0)
        
        self._adj2_is_dummy = False
        if raw_has_adj2 and lambda_cc > 0:
            # Real adj2 data
            self.has_adj2 = 1
            self.adj2_A = input['adj2_A'].to(self.device)
        
            ox2 = input['adj2_offset_x']
            oy2 = input['adj2_offset_y']
        
            self.adj2_offset_x = ox2.to(self.device) if isinstance(ox2, torch.Tensor) \
                else torch.tensor([ox2], device=self.device)
            self.adj2_offset_y = oy2.to(self.device) if isinstance(oy2, torch.Tensor) \
                else torch.tensor([oy2], device=self.device)
        
            if 'adj2_sheaf_cond' in input: self.adj2_sheaf_cond = input['adj2_sheaf_cond'].to(self.device)
            else: self.adj2_sheaf_cond = torch.zeros_like(self.sheaf_cond)
        
        elif not raw_has_adj2 and lambda_cc > 0:
            # Dummy adj2 for DDP static_graph safety
            self.has_adj2 = 1
            self._adj2_is_dummy = True
            self.adj2_A = torch.zeros_like(self.real_A)
            self.adj2_offset_x = torch.zeros(bs, device=self.device)
            self.adj2_offset_y = torch.zeros(bs, device=self.device)
            self.adj2_sheaf_cond = torch.zeros_like(self.sheaf_cond)
        
        else:
            # Cocycle loss disabled
            self.has_adj2 = 0
            self.adj2_A = None
            self.adj2_sheaf_cond = None

    def forward(self):
        # ── Test-only fast path ──
        # Skip SB dynamics + training forward entirely.
        # Test path is self-contained: starts from self.real_A, recomputes
        # all variables, runs multi-step inference independently.
        # Saves ~1.4s per forward() call (SB dynamics is unused in test mode).
        if (self.opt.phase == 'test') and (not self.isTrain):
            self.real = self.real_A
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1), times])
            times = torch.tensor(times).float().to(self.device)
            self.times = times
            
            bs = self.real.size(0)
            time_idx_init = (torch.randint(T, size=[1]).to(self.device) * torch.ones(size=[1]).to(self.device)).long()

            self.time_idx = time_idx_init
            self.timestep = times[time_idx_init]

            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1, 1, 1, 1)
                        scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)

                    Xt = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    z = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1 = self.netG(Xt, time_idx, z, sheaf_cond=self.sheaf_cond, global_cls=self.global_cls)
                    setattr(self, "fake_"+str(t+1), Xt_1)

                self.fake_B = Xt_1
            
            return

        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1),times])
        times = torch.tensor(times).float().to(self.device)
        self.times = times
        bs =  self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1]).to(self.device) * torch.ones(size=[1]).to(self.device)).long()
        self.time_idx = time_idx
        self.timestep     = times[time_idx]
        
        # Determine if adj SB paths are needed
        # Note: sheaf_cond is always a tensor when sheaf losses are active
        # (zero→None conversion removed for DDP static_graph safety).
        # Non-sheaf runs have adj_A=None, so use_adj_sb=False regardless.
        any_sheaf_loss = (getattr(self.opt, 'lambda_sheaf', 0) > 0 or getattr(self.opt, 'lambda_sheaf_feat', 0) > 0)
        any_cocycle = getattr(self.opt, 'lambda_sheaf_cocycle', 0) > 0
        use_adj_sb = (self.has_adj and (any_sheaf_loss or any_cocycle) and self.adj_A is not None and self.isTrain)
        use_adj2_sb = (getattr(self, 'has_adj2', 0) and any_cocycle and getattr(self, 'adj2_A', None) is not None and self.isTrain)
        gc = self.global_cls  # shorthand for global_cls

        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item()+1):

                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1,1,1,1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                
                Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_1     = self.netG(Xt, time_idx, z, sheaf_cond=self.sheaf_cond, global_cls=gc)

                Xt2       = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_12    = self.netG(Xt2, time_idx, z, sheaf_cond=self.sheaf_cond, global_cls=gc)

                if self.opt.nce_idt:
                    XtB = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1B = self.netG(XtB, time_idx, z, sheaf_cond=self.sheaf_cond, global_cls=gc)

                # Adj patch 1: same SB dynamics as ref
                if use_adj_sb:
                    Xt_adj  = self.adj_A if (t == 0) else (1-inter) * Xt_adj + inter * Xt_1_adj.detach() + (scale * tau).sqrt() * torch.randn_like(Xt_adj).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1_adj = self.netG(Xt_adj, time_idx, z, sheaf_cond=self.adj_sheaf_cond, global_cls=gc)

                # Adj patch 2: for cocycle consistency
                if use_adj2_sb:
                    Xt_adj2 = self.adj2_A if (t == 0) else (1-inter) * Xt_adj2 + inter * Xt_1_adj2.detach() + (scale * tau).sqrt() * torch.randn_like(Xt_adj2).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1_adj2 = self.netG(Xt_adj2, time_idx, z, sheaf_cond=self.adj2_sheaf_cond, global_cls=gc)

            if self.opt.nce_idt: self.XtB = XtB.detach()
            
            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()
            
            self.adj_A_noisy = Xt_adj.detach() if use_adj_sb else None
            self.adj2_A_noisy = Xt_adj2.detach() if use_adj2_sb else None
                      
        
        # ── Training forward path ──
        # Skip in test-only mode: z_in=[2*bs] causes AdaIN dimension
        # mismatch with batch>1. Test path (below) handles inference
        # independently, so this block's output is unused in test mode.
        # Double condition (phase + isTrain) for maximum safety.
        is_test_only = (self.opt.phase == 'test') and (not self.isTrain)
        if not is_test_only:
            z_in    = torch.randn(size=[2*bs,4*self.opt.ngf]).to(self.real_A.device)
            z_in2    = torch.randn(size=[bs,4*self.opt.ngf]).to(self.real_A.device)
            self.z_ref = z_in[:bs].detach()  # save for sheaf loss adj patch
            
            """Run forward pass"""
            self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
            self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_noisy

            if self.opt.flip_equivariance:
                self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)

                if self.flipped_for_equivariance:
                    self.real = torch.flip(self.real, [3])
                    self.realt = torch.flip(self.realt, [3])

            # Sheaf cond for nce_idt: duplicate for concat batch (real_A + real_B)
            sheaf_cond_full = self.sheaf_cond
            global_cls_full = gc

            if self.opt.nce_idt and self.opt.isTrain:
                if sheaf_cond_full is not None: sheaf_cond_full = torch.cat([self.sheaf_cond, self.sheaf_cond], dim=0)
                if global_cls_full is not None: global_cls_full = torch.cat([gc, gc], dim=0)

            self.fake = self.netG(self.realt, self.time_idx, z_in, sheaf_cond=sheaf_cond_full, global_cls=global_cls_full)
            self.fake_B2 = self.netG(self.real_A_noisy2, self.time_idx, z_in2, sheaf_cond=self.sheaf_cond, global_cls=gc)
            self.fake_B = self.fake[:self.real_A.size(0)]

            if self.opt.nce_idt: self.idt_B = self.fake[self.real_A.size(0):]

        if self.opt.phase == 'test':
            self.real = self.real_A  # needed for compatibility (replaces line 374)
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1),times])
            times = torch.tensor(times).float().to(self.device)
            self.times = times
            bs =  self.real.size(0)
            time_idx = (torch.randint(T, size=[1]).to(self.device) * torch.ones(size=[1]).to(self.device)).long()
            self.time_idx = time_idx
            self.timestep     = times[time_idx]
            visuals = []
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):                    
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1,1,1,1)
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)

                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1     = self.netG(Xt, time_idx, z, sheaf_cond=self.sheaf_cond, global_cls=self.global_cls)
                    setattr(self, "fake_"+str(t+1), Xt_1)
                
                self.fake_B = Xt_1 # Set fake_B for backward compatibility

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        bs =  self.real_A.size(0)

        fake = self.fake_B.detach()
        std = torch.rand(size=[1]).item() * self.opt.std

        if self.opt.netD == 'vfm_cond': # VFM-D: feed Prov-GigaPath spatial features alongside the image.
            with torch.no_grad():
                vfm_real = self._vfm_forward_spatial(self.real_B)

                # P2: reuse fake_B's VFM grid cached in compute_G_loss (already detached).
                if self._cached_fake_vfm_grid is not None: vfm_fake = self._cached_fake_vfm_grid
                else: vfm_fake = self._vfm_forward_spatial(fake)
            
            pred_fake = self.netD(fake, self.time_idx, vfm_fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            self.pred_real = self.netD(self.real_B, self.time_idx, vfm_real)
        
        else:
            pred_fake = self.netD(fake, self.time_idx)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            self.pred_real = self.netD(self.real_B, self.time_idx)
        
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        
        return self.loss_D
    
    def compute_E_loss(self):
        """Calculate GAN loss for the discriminator"""        
        
        bs =  self.real_A.size(0)        
        XtXt_1 = torch.cat([self.real_A_noisy,self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2,self.fake_B2.detach()], dim=1)
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() +temp + temp**2
        
        return self.loss_E

    def compute_G_loss(self):
        bs =  self.real_A.size(0)
        tau = self.opt.tau

        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        std = torch.rand(size=[1]).item() * self.opt.std

        # P2: Compute fake_B's VFM features ONCE (CLS for VL-SB, grid for VFM-D)
        # using a single forward when both are needed; gradients flow through.
        sb_uses_vfm = (self.opt.lambda_SB > 0.0 and getattr(self.opt, 'sb_cost', 'pixel_mse') == 'vfm_cosine')
        d_uses_vfm = (self.opt.lambda_GAN > 0.0 and self.opt.netD == 'vfm_cond')
        fake_cls, fake_grid = None, None

        if sb_uses_vfm and d_uses_vfm: fake_cls, fake_grid = self._vfm_forward_combined(self.fake_B)
        elif sb_uses_vfm: fake_cls = self._vfm_forward(self.fake_B)
        elif d_uses_vfm: fake_grid = self._vfm_forward_spatial(self.fake_B)
        
        # Cache detached fake grid so compute_D_loss can reuse it (saves one VFM forward).
        self._cached_fake_vfm_grid = fake_grid.detach() if fake_grid is not None else None

        if self.opt.lambda_GAN > 0.0:
            if self.opt.netD == 'vfm_cond': pred_fake = self.netD(fake, self.time_idx, fake_grid) # VFM-D: gradient flows through fake -> VFM (frozen) -> projection -> D.
            else: pred_fake = self.netD(fake, self.time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN

        else: self.loss_G_GAN = 0.0
        
        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)

            bs = self.opt.batch_size

            ET_XY    = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
            self.loss_SB = -(self.opt.num_timesteps-self.time_idx[0])/self.opt.num_timesteps*self.opt.tau*ET_XY

            if sb_uses_vfm:
                # VL-SB: domain-aware transport cost via Prov-GigaPath cosine distance.
                with torch.no_grad(): phi_xt = self._vfm_forward(self.real_A_noisy)
                
                phi_xt_n = F.normalize(phi_xt.flatten(1), dim=1)
                phi_fake_n = F.normalize(fake_cls.flatten(1), dim=1)
                self.loss_SB += self.opt.tau * (1.0 - (phi_xt_n * phi_fake_n).sum(dim=1)).mean()
            
            # Original UNSB pixel-space MSE.
            else: self.loss_SB += self.opt.tau*torch.mean((self.real_A_noisy-self.fake_B)**2)
        
        if self.opt.lambda_NCE > 0.0: self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else: self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else: loss_NCE_both = self.loss_NCE
        
        # ── Sheaf loss warmup (shared across all sheaf losses) ──
        epoch = getattr(self.opt, 'current_epoch', 0)
        warmup = self.opt.sheaf_warmup_epochs
        rampup = self.opt.sheaf_rampup_epochs
        
        if epoch < warmup: sheaf_weight = 0.0
        elif rampup > 0: sheaf_weight = min(1.0, (epoch - warmup) / rampup)
        else: sheaf_weight = 1.0

        gc = self.global_cls  # shorthand

        # ── Generate adj patches (shared by pixel/feature/cocycle losses) ──
        adj_fake = None
        any_adj_loss = (self.opt.lambda_sheaf > 0 or
                        getattr(self.opt, 'lambda_sheaf_feat', 0) > 0 or
                        getattr(self.opt, 'lambda_sheaf_cocycle', 0) > 0)

        # Always generate adj/adj2 when available (static graph for DDP).
        # sheaf_weight controls loss magnitude only, not forward execution.
        if any_adj_loss and self.adj_A_noisy is not None:
            from models.sheaf_loss import pixel_sheaf_loss, compute_overlap_crop
            adj_fake = self.netG(self.adj_A_noisy, self.time_idx, self.z_ref, sheaf_cond=self.adj_sheaf_cond, global_cls=gc)

        adj2_fake = None
        lambda_cc = getattr(self.opt, 'lambda_sheaf_cocycle', 0)

        if lambda_cc > 0 and self.adj2_A_noisy is not None:
            adj2_fake = self.netG(self.adj2_A_noisy, self.time_idx, self.z_ref, sheaf_cond=self.adj2_sheaf_cond, global_cls=gc)

        # ── Pixel sheaf loss (pairwise boundary consistency) ──
        self.loss_sheaf = 0.0
        if self.opt.lambda_sheaf > 0 and adj_fake is not None:
            if self._adj_is_dummy: self.loss_sheaf = adj_fake.sum() * 0.0 # Dummy: graph-connected zero (DDP parameter tracking)
            else:
                from models.sheaf_loss import pixel_sheaf_loss, compute_overlap_crop
                W = self.fake_B.shape[3]
                total_sheaf = 0.0
                valid_count = 0

                for b in range(bs):
                    ox = self.adj_offset_x[b].item()
                    oy = self.adj_offset_y[b].item()
                    _, _, valid = compute_overlap_crop(W, ox, oy)
                    
                    if not valid: continue
                    total_sheaf += pixel_sheaf_loss(self.fake_B[b:b+1], adj_fake[b:b+1], ox, oy, mode='l1', alpha=getattr(self.opt, 'sheaf_mean_alpha', 1.0))
                    valid_count += 1

                if valid_count > 0: self.loss_sheaf = (total_sheaf / valid_count) * sheaf_weight
                else: self.loss_sheaf = adj_fake.sum() * 0.0 # All overlaps invalid: graph-connected zero fallback

        # Multi-scale feature sheaf loss
        self.loss_sheaf_feat = 0.0
        lambda_sf = getattr(self.opt, 'lambda_sheaf_feat', 0)
        
        if lambda_sf > 0 and adj_fake is not None and self._adj_is_dummy: self.loss_sheaf_feat = adj_fake.sum() * 0.0
        elif lambda_sf > 0 and adj_fake is not None:
            from models.sheaf_loss import feature_sheaf_loss, compute_overlap_crop
            
            z_feat = self.z_ref
            feat_ref = self.netG(self.fake_B.detach(), self.time_idx * 0, z_feat,
                                 sheaf_cond=self.sheaf_cond, global_cls=gc,
                                 layers=self.nce_layers, encode_only=True)

            adj_fake_det = adj_fake.detach()
            feat_adj = self.netG(adj_fake_det, self.time_idx * 0, z_feat,
                                 sheaf_cond=self.adj_sheaf_cond, global_cls=gc,
                                 layers=self.nce_layers, encode_only=True)
            
            crop_size = self.opt.crop_size
            total_sf = 0.0
            valid_sf = 0
            
            for b in range(bs):
                ox = self.adj_offset_x[b].item()
                oy = self.adj_offset_y[b].item()
                _, _, valid = compute_overlap_crop(crop_size, ox, oy)
                
                if not valid: continue
                
                feat_ref_b = [f[b:b+1] for f in feat_ref]
                feat_adj_b = [f[b:b+1] for f in feat_adj]
                total_sf += feature_sheaf_loss(feat_ref_b, feat_adj_b, ox, oy, input_size=crop_size, mode='l1')
                valid_sf += 1

            if valid_sf > 0: self.loss_sheaf_feat = (total_sf / valid_sf) * sheaf_weight
            else: self.loss_sheaf_feat = adj_fake.sum() * 0.0

        # Cocycle gluing loss (triple-overlap consistency)
        # Two components per the Čech cocycle condition:
        #   1. ref ↔ adj2 pairwise (direct supervision)
        #   2. adj1 ↔ adj2 cocycle (transitivity check)
        self.loss_sheaf_cocycle = 0.0
        if lambda_cc > 0 and adj_fake is not None and adj2_fake is not None and (self._adj_is_dummy or self._adj2_is_dummy):
            self.loss_sheaf_cocycle = adj_fake.sum() * 0.0 + adj2_fake.sum() * 0.0 # Dummy: graph-connected zero for both adj paths

        elif lambda_cc > 0 and adj_fake is not None and adj2_fake is not None:
            from models.sheaf_loss import pixel_sheaf_loss, compute_overlap_crop
            W = self.fake_B.shape[3]
            total_cc = 0.0
            valid_cc = 0

            for b in range(bs):
                ox1 = self.adj_offset_x[b].item()
                oy1 = self.adj_offset_y[b].item()
                ox2 = self.adj2_offset_x[b].item()
                oy2 = self.adj2_offset_y[b].item()

                # 1. ref ↔ adj2 pairwise
                _, _, valid_pw2 = compute_overlap_crop(W, ox2, oy2)
                if valid_pw2:
                    total_cc += pixel_sheaf_loss(self.fake_B[b:b+1], adj2_fake[b:b+1], ox2, oy2, mode='l1', alpha=getattr(self.opt, 'sheaf_mean_alpha', 1.0))
                    valid_cc += 1

                # 2. adj1 ↔ adj2 cocycle (relative offset)
                rel_ox = ox2 - ox1
                rel_oy = oy2 - oy1

                _, _, valid_rel = compute_overlap_crop(W, rel_ox, rel_oy)
                if valid_rel:
                    total_cc += pixel_sheaf_loss(adj_fake[b:b+1], adj2_fake[b:b+1], rel_ox, rel_oy, mode='l1', alpha=getattr(self.opt, 'sheaf_mean_alpha', 1.0))
                    valid_cc += 1

            if valid_cc > 0: self.loss_sheaf_cocycle = (total_cc / valid_cc) * sheaf_weight
            else: self.loss_sheaf_cocycle = adj_fake.sum() * 0.0 + adj2_fake.sum() * 0.0 # All overlaps invalid: graph-connected zero fallback

        # Fourier Edge Loss (high-frequency preservation)
        self.loss_fourier_edge = 0.0
        if self.opt.lambda_fourier_edge > 0:
            self.loss_fourier_edge = fourier_edge_loss(self.fake_B, self.real_B) * self.opt.lambda_fourier_edge

        # DAB Intensity Loss (staining fidelity)
        self.loss_dab_intensity = 0.0
        if self.opt.lambda_dab_intensity > 0 and hasattr(self, 'dab_extractor'):
            self.loss_dab_intensity = dab_intensity_loss(self.fake_B, self.real_B, self.dab_extractor) * self.opt.lambda_dab_intensity

        # DAB-VFM Feature Matching (KD-style distillation in DAB-only domain)
        # Extract DAB intensity (differentiable color deconvolution) → render as
        # brown-tinted RGB via Beer-Lambert → feed into Prov-GigaPath. The
        # student (G's DAB pattern) is supervised by the teacher (real IHC's DAB
        # pattern) in the dual-stain VFM's feature space, providing chromogen-
        # specific feedback that justifies the dual-stain pretraining.
        # Phase-2 gating: VFM-FM only activates from --vfm_fm_start_epoch.
        # For epoch < start, effective weight is 0 (skip forward to save compute).
        _vfm_fm_start = getattr(self.opt, 'vfm_fm_start_epoch', 1)
        _vfm_fm_active = epoch >= _vfm_fm_start

        self.loss_vfm_fm = 0.0
        if (getattr(self.opt, 'lambda_vfm_fm', 0) > 0 and _vfm_fm_active and hasattr(self, 'dab_extractor') and self._vfm is not None):
            dab_fake = self.dab_extractor.extract_dab_intensity(self.fake_B)
            rgb_dab_fake = self._dab_to_rgb(dab_fake)
            phi_dab_fake = self._vfm_forward_spatial(rgb_dab_fake)  # gradient flows
           
            with torch.no_grad():
                dab_real = self.dab_extractor.extract_dab_intensity(self.real_B)
                rgb_dab_real = self._dab_to_rgb(dab_real)
                phi_dab_real = self._vfm_forward_spatial(rgb_dab_real)
           
            self.loss_vfm_fm = F.l1_loss(phi_dab_fake, phi_dab_real)
        
        lambda_vfm_fm_eff = self.opt.lambda_vfm_fm if _vfm_fm_active else 0.0

        self.loss_G = self.loss_G_GAN + self.opt.lambda_SB*self.loss_SB + \
                      self.opt.lambda_NCE*loss_NCE_both + \
                      self.opt.lambda_sheaf * self.loss_sheaf + \
                      lambda_sf * self.loss_sheaf_feat + \
                      lambda_cc * self.loss_sheaf_cocycle + \
                      self.loss_fourier_edge + \
                      self.loss_dab_intensity + \
                      lambda_vfm_fm_eff * self.loss_vfm_fm
        
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        z    = torch.randn(size=[self.real_A.size(0),4*self.opt.ngf]).to(self.real_A.device)
        feat_q = self.netG(tgt, self.time_idx*0, z, sheaf_cond=self.sheaf_cond, global_cls=self.global_cls, layers=self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance: feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.time_idx*0, z, sheaf_cond=self.sheaf_cond, global_cls=self.global_cls, layers=self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers