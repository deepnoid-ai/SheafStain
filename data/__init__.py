import importlib
import torch.utils.data

import torch.distributed as dist

from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader. This is the main interface between this package and 'train.py'/'test.py'
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)

        use_ddp = getattr(opt, 'use_ddp', False)

        if use_ddp and opt.isTrain:
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=opt.world_size,
                rank=opt.rank,
                shuffle=not opt.serial_batches,
            )
            shuffle = False  # sampler handles shuffling
        else:
            self.sampler = None
            shuffle = not opt.serial_batches

        is_rank0 = not use_ddp or opt.rank == 0
        if is_rank0:
            print("dataset [%s] was created" % type(self.dataset).__name__)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=shuffle,
            sampler=self.sampler,
            num_workers=int(opt.num_threads),
            drop_last=True if opt.isTrain else False,
            pin_memory=True if len(opt.gpu_ids) > 0 else False,
        )

    def set_epoch(self, epoch):
        self.dataset.current_epoch = epoch
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
