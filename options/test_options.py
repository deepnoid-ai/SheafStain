from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=0, help='how many test images to run (0 = all)')
        parser.add_argument('--num_workers', type=int, default=1, help='Total number of GPU workers for parallel inference')
        parser.add_argument('--worker_id', type=int, default=0, help='This worker ID (0-indexed). Each worker processes images[worker_id::num_workers]')

        # inference options (overlap + tissue filtering)
        parser.add_argument('--test_stride', type=int, default=256, help='Stride for test tiling. 256=no overlap (4x4), 128=50%% overlap (7x7), 192=25%% overlap (5x5)')
        parser.add_argument('--tissue_threshold', type=float, default=0.1, help='Min tissue ratio. Patches below this are treated as background (skip generator). Only used with --skip_background.')
        parser.add_argument('--skip_background', action='store_true', help='Skip generator for background-dominant patches. Apply tone matching instead.')

        # inference options (sheafified global conditioning)
        parser.add_argument('--sanity_check', action='store_true', help='Run sanity check: compare global vs local conditioning maps. Exits after check (no full inference).')
        parser.add_argument('--sanity_n_samples', type=int, default=10, help='Number of images for sanity check (default: 10)')

        # inference options (sheaf-guided bridge generation)
        parser.add_argument('--bridge_blend_width', type=int, default=32, help='Pixels on each side of seam for bridge blending. 32px = 2 spatial conditioning cells.')
        parser.add_argument('--bridge_compat_threshold', type=float, default=0.0, help='VFM spatial map cosine similarity threshold for '
                                                                                       'selective bridge generation. '
                                                                                       '0.0 = generate all bridges (default), '
                                                                                       '>0 = skip bridges where compatibility exceeds threshold.')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        
        self.isTrain = False
      
        return parser
