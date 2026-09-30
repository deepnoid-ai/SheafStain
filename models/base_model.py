import os
import torch

import torch.distributed as dist

from . import networks
from collections import OrderedDict
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.use_ddp = getattr(opt, 'use_ddp', False)
        self.local_rank = getattr(opt, 'local_rank', 0)
        self.rank = getattr(opt, 'rank', 0)

        # Device: use local_rank for DDP, gpu_ids[0] for single-GPU
        if self.gpu_ids:
            self.device = torch.device('cuda:{}'.format(self.local_rank if self.use_ddp else self.gpu_ids[0]))
        else:
            self.device = torch.device('cpu')

        if opt.preprocess != 'scale_width': torch.backends.cudnn.benchmark = True
        
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0

        # Loss EMA tracking (per-loss exponential moving averages)
        # Near-zero cost per iteration; enables smoothed loss trajectories
        # for training-dynamics analysis (see manuscript app:training_dynamics).
        self.loss_ema = OrderedDict()
        self.ema_beta = getattr(opt, 'loss_ema_beta', 0.99)
        self._ema_initialized = False

    def _is_rank0(self):
        """Check if current process is rank 0."""
        return self.rank == 0

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers"""
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)

        if self._is_rank0():
            self.print_networks(opt.verbose)

    def parallelize(self):
        """Wrap networks with DDP or DataParallel."""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if self.use_ddp:
                    net = net.to(self.device)
                    net = torch.nn.parallel.DistributedDataParallel(
                        net,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=True,
                        broadcast_buffers=False,
                        static_graph=True,
                    )
                else:
                    net = torch.nn.DataParallel(net, self.opt.gpu_ids)
                setattr(self, 'net' + name, net)

    def data_dependent_initialize(self, data):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        if self._is_rank0():
            lr = self.optimizers[0].param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        self._update_loss_ema(errors_ret)
        return errors_ret

    def _update_loss_ema(self, losses):
        """Update per-loss exponential moving averages in-place.

        On the first call, seeds each EMA with the current raw value to avoid
        long warm-up bias. Subsequent calls apply standard EMA update:
            ema <- beta * ema + (1 - beta) * current
        Cost is O(n_losses) per call (negligible vs. forward/backward pass).
        """
        if not self._ema_initialized:
            for k, v in losses.items():
                self.loss_ema[k] = float(v)
            self._ema_initialized = True
            return
        beta = self.ema_beta
        for k, v in losses.items():
            if k in self.loss_ema:
                self.loss_ema[k] = beta * self.loss_ema[k] + (1.0 - beta) * float(v)
            else:
                self.loss_ema[k] = float(v)

    def get_loss_ema(self):
        """Return smoothed (EMA) loss values for logging / dynamics analysis."""
        return OrderedDict(self.loss_ema)

    def save_networks(self, epoch):
        """Save all the networks to the disk. Only called on rank 0."""
        if not self._is_rank0():
            return

        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # Unwrap DDP/DP module
                    net_module = net.module if hasattr(net, 'module') else net
                    # Copy state_dict to CPU without moving the model itself
                    # (moving the model breaks DDP's registered hooks/buffers)
                    state_dict = {k: v.cpu().clone() for k, v in net_module.state_dict().items()}
                    torch.save(state_dict, save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk."""
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
                else:
                    load_dir = self.save_dir

                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net' + name)
                # Unwrap DDP or DataParallel
                if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                    net = net.module
                if self._is_rank0():
                    print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture"""
        if not self._is_rank0():
            return
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}
