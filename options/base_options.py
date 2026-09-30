import os
import sys
import data
import yaml
import torch
import models
import argparse

from util import util


class BaseOptions():
    def __init__(self, cmd_line=None):
        self.initialized = False
        self.cmd_line = None

        if cmd_line is not None: self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default='placeholder', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--config', type=str, default='config.yaml', help='Path to the unified YAML config (local paths + optional overrides). See config.yaml.')
        parser.add_argument('--easy_label', type=str, default='experiment_name', help='Interpretable name')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        # model parameters
        parser.add_argument('--model', type=str, default='sb', help='chooses which model to use.')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--num_timesteps', type=int, default=5, help='# of discrim filters in the first conv layer')
        parser.add_argument('--embedding_dim', type=int, default=512, help='# of output image channels: 3 for RGB and 1 for grayscale')
        
        parser.add_argument('--netD', type=str, default='basic_cond', choices=['basic', 'basic_cond', 'vfm_cond', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'], help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator. vfm_cond extends basic_cond with Prov-GigaPath spatial features at input.')
        parser.add_argument('--netE', type=str, default='basic_cond', choices=['basic', 'basic_cond', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2', 'patchstylegan2'], help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks_cond', choices=['resnet_9blocks', 'resnet_9blocks_cond', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], help='specify generator architecture')
        
        parser.add_argument('--embedding_type', type=str, default='positional', choices=['fourier', 'positional'], help='specify generator architecture')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--style_dim', type=int, default=512, help='only used if netD==n_layers')
        parser.add_argument('--n_mlp', type=int, default=3, help='only used if netD==n_layers')
        
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
        parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
        
        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        
        parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--std', type=float, default=0.25, help='Scale of Gaussian noise added to data')
        parser.add_argument('--tau', type=float, default=0.01, help='Entropy parameter')
        parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
        
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='sheaf', help='chooses how datasets are loaded. [sheaf | sheaf_test]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--random_scale_max', type=float, default=3.0, help='(used for single image translation) Randomly scale the image by the specified factor as data augmentation.')
        
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # parameters related to StyleGAN2-based networks
        parser.add_argument('--stylegan2_G_num_downsampling', default=1, type=int, help='Number of downsampling layers used by StyleGAN2Generator')

        # DDP / seed
        parser.add_argument('--seed', type=int, default=1024, help='random seed for reproducibility')

        # Debug / quick test
        parser.add_argument('--debug_mode', action='store_true', help='Quick debug run: uses only 10 images, reduces epochs to 2, disables HTML saving')

        # WandB logging
        parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging')
        parser.add_argument('--wandb_project', type=str, default='SheafStain', help='WandB project name')
        parser.add_argument('--wandb_api_key_file', type=str, default=None, help='Path to text file containing WandB API key (one line, key only)')

        # ── Sheaf conditioning parameters ──
        parser.add_argument('--sheaf_cond_dim', type=int, default=0, help='Sheaf conditioning dimension. 0=disabled (vanilla SB), >0=enabled')
        parser.add_argument('--vfm_embed_dim', type=int, default=1536, help='Embedding dim D of VFM tokens stored in spatial presets. '
                                                                            'Determines the input dim of the generator\'s conditioning '
                                                                            'projection layers. 1536 = Prov-GigaPath / UNI2-h, '
                                                                            '1280 = Virchow2. Must match preset[\'vfm_embed_dim\'].')
        parser.add_argument('--sheaf_injection_mode', type=str, default='additive', choices=['additive', 'adaln', 'gating'], help='How to inject sheaf conditioning into ResNet blocks')
        parser.add_argument('--sheaf_agg_mode', type=str, default='agnostic', choices=['agnostic', 'direction'], help='Aggregation mode: agnostic (mean pool) or direction-aware')
        parser.add_argument('--sheaf_zero_init', action='store_true', default=True, help='Zero-init conditioning layers (model freely learns to use cond)')
        parser.add_argument('--sheaf_no_zero_init', dest='sheaf_zero_init', action='store_false', help='Xavier-init conditioning layers (force model to use cond)')
        parser.add_argument('--sheaf_spatial', action='store_true', default=False, help='Use spatially-varying conditioning (fiber bundle). '
                                                                                        'Each position in the feature map receives position-dependent '
                                                                                        'conditioning from VFM tokens. Requires spatial presets.')
        parser.add_argument('--sheaf_cache_batch_size', type=int, default=64, help='Batch size for VFM inference during cache computation')
        parser.add_argument('--sheaf_cache_refresh_freq', type=int, default=5, help='Recompute sheaf cache every N epochs. '
                                                                                    'Ref positions stay fixed between refreshes. '
                                                                                    '1=every epoch (original), 5=recommended')
        parser.add_argument('--sheaf_preset_selection', type=str, default='roundrobin', choices=['roundrobin', 'random', 'stratified'], help='Preset selection: roundrobin (sequential), '
                                                                                                                                             'random (uniform random), '
                                                                                                                                             'stratified (shuffled cycles, all presets used before repeating)')
        parser.add_argument('--shuffle_epoch', type=int, default=5, help='Epoch interval for loading a new preset. Between shuffles, resample_subset provides diversity.')
        parser.add_argument('--vfm_name', type=str, default='gigapath', choices=['gigapath', 'uni', 'uni2h', 'virchow2'], help='VFM family used for the VFM-FM teacher and spatial-cond presets. '
                                                                                                                               'Must match the VFM used to generate --sheaf_preset_dir. '
                                                                                                                               'gigapath: ViT-G/14, D=1536, 14x14 native. '
                                                                                                                               'uni:      ViT-L/16, D=1024, 14x14 native (Mass-100K DINOv2). '
                                                                                                                               'uni2h:    ViT-H/14, D=1536, 16x16 native (resampled to 14). '
                                                                                                                               'virchow2: ViT-H/14, D=1280, 16x16 native + 4 reg tokens.')
        parser.add_argument('--vfm_model_path', type=str, default=None, help='Path to VFM weights directory (must contain pytorch_model.bin; '
                                                                             'optional config.json next to it for architecture overrides).')
        parser.add_argument('--lrm_checkpoint', type=str, default=None, help='Path to pre-trained LRM checkpoint (.pt). None=raw VFM only (no sheaf correction)')
        parser.add_argument('--sheaf_token_strategy', type=str, default='overlap_interp', choices=['overlap_interp', 'full_interp', 'overlap_subsample'], help='LRM token strategy: '
                                                                                                                                                               'overlap_interp (B, recommended), '
                                                                                                                                                               'full_interp (A, simplest), '
                                                                                                                                                               'overlap_subsample (C, no interpolation)')
        parser.add_argument('--sheaf_stride', type=int, default=80, help='Stride for overlapped patch extraction (pixels)')
        parser.add_argument('--sheaf_num_jitter_vh', type=int, default=2, help='Number of jitter patches per cardinal direction (0=center only)')
        parser.add_argument('--sheaf_jitter_range', type=int, default=32, help='Jitter range in pixels for cardinal overlapped patches')
        parser.add_argument('--sheaf_test_cache_dir', type=str, default=None, help='Directory with precomputed test cache (.pt files). None=compute on-the-fly')
        parser.add_argument('--sheaf_preset_dir', type=str, default=None, help='Directory containing precomputed sheaf cache presets')
        parser.add_argument('--lambda_sheaf', type=float, default=0.0, help='Sheaf loss weight. 0=disabled, >0=enabled')
        parser.add_argument('--lambda_sheaf_cocycle', type=float, default=0.0, help='Cocycle gluing loss weight (triple-overlap consistency). 0=pairwise only, >0=adds Cech cocycle condition')
        parser.add_argument('--lambda_fourier_edge', type=float, default=0.0, help='Fourier edge loss weight (high-freq preservation). 0=disabled, >0=enabled')
        parser.add_argument('--lambda_dab_intensity', type=float, default=0.0, help='DAB intensity matching loss weight (p90 score). 0=disabled, >0=enabled')
        parser.add_argument('--lambda_vfm_fm', type=float, default=0.0, help='DAB-channel VFM Feature Matching loss weight '
                                                                             '(KD-style distillation in the IHC chromogen domain via '
                                                                             'Beer-Lambert reconstructed DAB-only RGB through Prov-GigaPath). '
                                                                             '0=disabled, suggested 0.5~1.0 when enabled.')
        parser.add_argument('--vfm_fm_start_epoch', type=int, default=1, help='Epoch at which VFM-FM loss activates (phase-2 trigger). '
                                                                              'For epoch < vfm_fm_start_epoch, lambda_vfm_fm is masked to 0; '
                                                                              'for epoch >= vfm_fm_start_epoch, full lambda_vfm_fm applies. '
                                                                              'Default 1 = active from start (no phase gating).')
        parser.add_argument('--sheaf_mean_alpha', type=float, default=1.0, help='Weight of pixel L1 relative to mean-diff in sheaf loss. 1.0=both equal, 0=mean-only, large=L1-dominant')
        parser.add_argument('--sheaf_warmup_epochs', type=int, default=0, help='Epochs before sheaf loss activation. '
                                                                               'Note: adj forward always runs regardless of warmup '
                                                                               '(static_graph=True requires constant graph structure)')
        parser.add_argument('--sheaf_rampup_epochs', type=int, default=10, help='Epochs for sheaf loss alpha ramp-up (0->1)')
        parser.add_argument('--use_global_cls', action='store_true', default=False, help='Enable CLS-token conditioning '
                                                                                         '(image- or grid-level tone harmonization). '
                                                                                         'Source selected by --cls_token_type '
                                                                                         '(default: global). When False, no CLS '
                                                                                         'projection is instantiated and no injection '
                                                                                         'occurs at model forward.')
        parser.add_argument('--cls_token_type', type=str, default='global', choices=['global', 'neighborhood'], help="CLS source when --use_global_cls is set. "
                                                                                                                     "'global': whole-image 16-tile VFM CLS mean "
                                                                                                                     "(keyed by img_idx). "
                                                                                                                     "'neighborhood': per-grid overlap-patch "
                                                                                                                     "open-cover VFM CLS mean ("
                                                                                                                     "keyed by (img_idx, grid_idx)).")
        parser.add_argument('--global_cls_path', type=str, default=None, help='Path to precomputed global CLS file (.pt). '
                                                                              'If None and --use_global_cls, falls back to '
                                                                              '{dataroot}/global_cls.pt. Only applies when '
                                                                              '--cls_token_type=global.')

        self.initialized = True

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None: opt, _ = parser.parse_known_args()
        else: opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        
        if self.cmd_line is None: opt, _ = parser.parse_known_args()  # parse again with new defaults
        else: opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        
        if self.cmd_line is None: return parser.parse_args()
        else: return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default: comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def _apply_config(self, opt):
        """Load the unified YAML config (opt.config) and apply it onto opt.

        Precedence: explicit CLI flag > config.yaml > built-in argparse default.
        Every key in the YAML sets opt.<key>; required local paths are asserted
        (no environment fallback, no default path).
        """
        cfg_path = opt.config
        assert os.path.exists(cfg_path), f"config file not found: '{cfg_path}'. Copy config.yaml, fill in your local paths, and pass --config <file> (default: config.yaml)."
        with open(cfg_path) as f: cfg = yaml.safe_load(f) or {}
        
        assert isinstance(cfg, dict), f"{cfg_path}: top level must be a mapping (key: value)."
        
        # Flags given on the command line win, even when their value equals the default.
        cli = self.cmd_line if self.cmd_line is not None else sys.argv[1:]
        explicit = {a[2:].split('=', 1)[0].replace('-', '_') for a in cli if a.startswith('--')}

        for key, val in cfg.items():
            if val is None or key in explicit: continue
            setattr(opt, key, val)
 
        # required local paths (no fallback)
        assert getattr(opt, 'vfm_model_path', None), f"vfm_model_path is required: set it in '{cfg_path}' or pass --vfm_model_path."
        if opt.isTrain:
            assert getattr(opt, 'sheaf_preset_dir', None), f"sheaf_preset_dir is required for training: set it in '{cfg_path}' or pass --sheaf_preset_dir."

    def parse(self, rank=0):
        """Parse our options, create checkpoints directory suffix, and set up gpu device.

        Parameters:
            rank (int) -- process rank for DDP (0 for single-GPU or non-distributed)
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # Merge the unified YAML config (config.yaml) onto opt.
        self._apply_config(opt)

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # Only rank 0 prints and saves options
        if rank == 0: self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []

        for str_id in str_ids:
            id = int(str_id)
            if id >= 0: opt.gpu_ids.append(id)

        # For DDP, torch.cuda.set_device is handled in train.py setup_ddp()
        # For non-DDP, set it here
        if 'LOCAL_RANK' not in os.environ:
            if len(opt.gpu_ids) > 0: torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        return self.opt
