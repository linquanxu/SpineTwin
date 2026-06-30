import os
import os.path as osp
import time
import argparse

import torch
import torch.distributed as dist

from utils.logger import get_logger
from utils.pyt_utils import parse_devices, all_reduce_tensor, extant_file

# try:
#     from apex.parallel import DistributedDataParallel, SyncBatchNorm
# except ImportError:
#     raise ImportError(
#         "Please install apex from https://www.github.com/nvidia/apex .")


logger = get_logger()
def _make_worker_seed_fn(base_seed: int, rank: int = 0):
    def seed_worker(worker_id):
        worker_seed = (base_seed + worker_id + rank * 9973) % (2**32)
        import numpy as _np, random as _random, torch as _torch
        _np.random.seed(worker_seed)
        _random.seed(worker_seed)
        _torch.manual_seed(worker_seed)
        try:
            import albumentations as A
            A.set_seed(int(worker_seed))
            from monai.utils import set_determinism as _monai_set_det
            _monai_set_det(seed=int(worker_seed))
        except Exception:
            pass
    return seed_worker
def _make_generator(base_seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(base_seed))
    return g
    
class Engine(object):
    def __init__(self, custom_parser=None):
        logger.info(
            "PyTorch Version {}".format(torch.__version__))
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        self.continue_state_object = self.args.continue_fpath

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1
            print("WORLD_SIZE is %d" % (int(os.environ['WORLD_SIZE'])))
        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
            self.devices = [i for i in range(self.world_size)]
        else:
            gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            self.devices =  [i for i in range(len(gpus.split(',')))]


        self.base_seed = getattr(self.args, 'random_seed', 42) if hasattr(self, 'args') else 42
        self.local_rank = getattr(self.args, 'local_rank', 0) if hasattr(self, 'args') else 0
        rank_seed = int(self.base_seed) + int(self.local_rank)
        self._dl_worker_init_fn = _make_worker_seed_fn(rank_seed, self.local_rank)
        self._dl_generator = _make_generator(rank_seed)

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        # p.add_argument('--local_rank', default=0, type=int,
        #                help='process rank on node')

    def data_parallel(self, model):
        # if self.distributed:
        #     model = DistributedDataParallel(model)
        # else:
        model = torch.nn.DataParallel(model)
        return model


    def get_train_loader(self, train_dataset, collate_fn=None, drop_last=False):
        base_seed = getattr(self, 'base_seed', 42)
        local_rank = getattr(self, 'local_rank', 0)
        worker_init_fn = getattr(self, '_dl_worker_init_fn', _make_worker_seed_fn(base_seed, local_rank))
        generator = getattr(self, '_dl_generator', _make_generator(base_seed))


        train_sampler = None
        is_shuffle = True
        batch_size = self.args.batch_size

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=self.world_size,
            rank=local_rank,
            shuffle=True,
            seed=base_seed)
            batch_size = self.args.batch_size // self.world_size
            is_shuffle = False

        dl_kwargs = dict(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            drop_last=drop_last,
            shuffle=is_shuffle,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        if self.args.num_workers and self.args.num_workers > 0:
            dl_kwargs.update(dict(
                persistent_workers=True,
                prefetch_factor=3,
            ))


        train_loader = torch.utils.data.DataLoader(**dl_kwargs)
        return train_loader, train_sampler

    def get_test_loader(self, test_dataset, batch_size,collate_fn=None):
        test_sampler = None
        is_shuffle = False
        batch_size = batch_size
        worker_init_fn = getattr(self, '_dl_worker_init_fn', _make_worker_seed_fn(self.base_seed, self.local_rank))
        generator = getattr(self, '_dl_generator', _make_generator(self.base_seed))
        if self.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset)
            batch_size = self.args.batch_size // self.world_size

        dl_kwargs = dict(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            drop_last=False,
            shuffle=is_shuffle,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        if self.args.num_workers and self.args.num_workers > 0:
            dl_kwargs.update(dict(
                persistent_workers=True,
                prefetch_factor=3, 
            ))
        test_loader = torch.utils.data.DataLoader(**dl_kwargs)

        return test_loader, test_sampler

    def get_val_loader(self, test_dataset, batch_size):
        test_sampler = None
        is_shuffle = False
        batch_size = batch_size

        if self.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset)
            batch_size = self.args.batch_size // self.world_size

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                       batch_size=batch_size,
                                       num_workers=self.args.num_workers,
                                       drop_last=False,
                                       shuffle=is_shuffle,
                                       pin_memory=True,
                                       sampler=test_sampler)

        return test_loader, test_sampler

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            return torch.mean(tensor)


    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False
