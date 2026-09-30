import os
import time
import torch
import datetime

import numpy as np
import torch.distributed as dist

from data import create_dataset
from models import create_model
from models.sheaf_modules import SheafEmbeddingCache

from util.visualizer import Visualizer
from util.wandb_logger import WandBLogger
from options.train_options import TrainOptions

def setup_ddp():
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=4))
    local_rank = int(os.environ['LOCAL_RANK'])
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()


def is_rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


if __name__ == '__main__':
    use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if use_ddp: local_rank, rank, world_size = setup_ddp()
    else: local_rank, rank, world_size = 0, 0, 1

    opt = TrainOptions().parse(rank=rank)  # get training options (rank-aware)
    opt.local_rank = local_rank
    opt.rank = rank
    opt.world_size = world_size
    opt.use_ddp = use_ddp

    # Override gpu_ids for DDP: each process uses only its local_rank GPU
    if use_ddp: opt.gpu_ids = [local_rank]

    # Seed for reproducibility (data augmentation varies per rank for diversity)
    seed = opt.seed + rank if hasattr(opt, 'seed') else 1024 + rank
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
 
    # Debug mode: limit data and epochs for quick testing
    if getattr(opt, 'debug_mode', False):
        opt.max_dataset_size = 10
        opt.n_epochs = 2
        opt.n_epochs_decay = 0
        opt.print_freq = 1
        opt.display_freq = 1
        opt.save_epoch_freq = 1
        
        if is_rank0(): print("[DEBUG MODE] max_dataset_size=10, epochs=2")

    dataset = create_dataset(opt)   # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    dataset_size = len(dataset)     # get the number of images in the dataset.

    model = create_model(opt)       # create a model given opt.model and other options

    if is_rank0(): print('The number of training images = %d' % dataset_size)

    # Sheaf Embedding Cache
    sheaf_cache = None
    use_sheaf = getattr(opt, 'sheaf_cond_dim', 0) > 0

    if use_sheaf:
        sheaf_cache = SheafEmbeddingCache(opt)
        if is_rank0(): print(f'[Sheaf] Conditioning enabled: dim={opt.sheaf_cond_dim}, injection={opt.sheaf_injection_mode}, agg={opt.sheaf_agg_mode}')

    # Global CLS
    # Separate file takes priority; if absent, the preset provides global_cls
    # during epoch-loop preset loading (load_spatial_preset_from_dir).
    if getattr(opt, 'use_global_cls', False) and sheaf_cache is not None:
        cls_path = getattr(opt, 'global_cls_path', None)
        
        if cls_path is None: cls_path = os.path.join(opt.dataroot, 'global_cls.pt')
        if os.path.exists(cls_path): sheaf_cache.load_global_cls(cls_path)
        elif is_rank0(): print(f"[Sheaf] global_cls.pt not found at {cls_path}. Will use global_cls from the preset if available.")

    # Visualizer only on rank 0
    if is_rank0():
        visualizer = Visualizer(opt)
        opt.visualizer = visualizer

    else:
        visualizer = None
        opt.visualizer = None

    # WandB Logger (rank 0 only)
    wandb_logger = None

    if getattr(opt, 'use_wandb', False) and is_rank0(): wandb_logger = WandBLogger(opt, enabled=True)

    total_iters = 0                 # the total number of training iterations
    optimize_time = 0.1
    times = []

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        opt.current_epoch = epoch  # for sheaf loss warm-up

        # Two-phase boundary log (one-shot, rank0).
        _fm_start = getattr(opt, 'vfm_fm_start_epoch', None)
        _lam_fm = getattr(opt, 'lambda_vfm_fm', 0)

        if is_rank0() and _fm_start and _lam_fm and _lam_fm > 0 and epoch == _fm_start:
            print(f"[SheafStain] === Phase 2 ACTIVATED at epoch {epoch}: VFM-FM loss enabled (lambda_vfm_fm={_lam_fm}) ===")

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        if is_rank0() and visualizer is not None: visualizer.reset()

        dataset.set_epoch(epoch)
        dataset2.set_epoch(epoch)

        # Load sheaf conditioning from precomputed presets
        if use_sheaf and sheaf_cache is not None:
            shuffle_epoch = getattr(opt, 'shuffle_epoch', 5)
            preset_selection = getattr(opt, 'sheaf_preset_selection', 'roundrobin')
            need_refresh = (epoch % shuffle_epoch == 0) or (epoch == opt.epoch_count)

            if need_refresh:
                # All ranks load presets independently (DDP: each rank has its
                # own dataset object, so each must load its own copy)
                preset_lookup = sheaf_cache.load_spatial_preset_from_dir(opt.sheaf_preset_dir, epoch=epoch, selection=preset_selection)

                dataset.dataset.set_preset_lookup(preset_lookup)
                dataset2.dataset.set_preset_lookup(preset_lookup)

                dataset.dataset.set_sheaf_cache(sheaf_cache.get_cache_dict())
                dataset2.dataset.set_sheaf_cache(sheaf_cache.get_cache_dict())

                # Global CLS (image-level, from preset or separate file)
                global_cls_dict = sheaf_cache.get_global_cls_dict()
                if global_cls_dict:
                    dataset.dataset.set_global_cls_cache(global_cls_dict)
                    dataset2.dataset.set_global_cls_cache(global_cls_dict)
                if use_ddp: dist.barrier()

        for i, (data, data2) in enumerate(zip(dataset, dataset2)):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0: t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            if len(opt.gpu_ids) > 0: torch.cuda.synchronize()
            optimize_start_time = time.time()

            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data, data2)
                model.setup(opt)
                model.parallelize()

            model.set_input(data, data2)
            model.optimize_parameters()

            if len(opt.gpu_ids) > 0: torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            # Rank-0 only: display, print, save
            if is_rank0() and visualizer is not None:
                if total_iters % opt.display_freq == 0:
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    # WandB image logging
                    if wandb_logger is not None: wandb_logger.log_images(epoch, model, data, step=total_iters)

                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                    if opt.display_id is None or opt.display_id > 0: visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                    # WandB scalar logging (raw + EMA-smoothed trajectories)
                    if wandb_logger is not None:
                        wandb_logger.log_scalars(epoch, losses, step=total_iters)
                        try:
                            losses_ema = model.get_loss_ema()
                            wandb_logger.log_scalars(epoch, {f'ema_{k}': v for k, v in losses_ema.items()}, step=total_iters)
                        except AttributeError: pass  # older checkpoints without EMA API

            if total_iters % opt.save_latest_freq == 0:
                if is_rank0():
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    print(opt.name)

                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                    if use_sheaf and sheaf_cache is not None: sheaf_cache.save_aggregator(os.path.join(opt.checkpoints_dir, opt.name), save_suffix)
                
                if use_ddp: dist.barrier()  # all ranks wait for rank 0 to finish saving

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            if is_rank0():
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))

                model.save_networks('latest')
                model.save_networks(epoch)

                if use_sheaf and sheaf_cache is not None:
                    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
                    sheaf_cache.save_aggregator(ckpt_dir, 'latest')
                    sheaf_cache.save_aggregator(ckpt_dir, epoch)

            if use_ddp: dist.barrier()

        if is_rank0(): print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        model.update_learning_rate()

    # Finalize WandB and DDP
    if wandb_logger is not None: wandb_logger.finish()
    if use_ddp: cleanup_ddp()
