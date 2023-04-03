from enum import Enum, auto
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from prefetch_generator import BackgroundGenerator

from torch.utils.data import  DataLoader
from utils.utils import *




class DataLoader_(DataLoader):
    # ref: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#issuecomment-495090086
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    


def create_dataloader(cfg, mode, rank,dataset,my_collate_fn=None):
    assert mode in ['train','dev','eval_key','eval_dev','eval_test','eval_train']
    if cfg.dataloader.use_background_generator:
        data_loader = DataLoader_ #set to False when using ddp
    else:
        data_loader = DataLoader
    # dataset = Dataset_(cfg, mode)
    train_use_shuffle = True
    sampler = None
    if cfg.dist.gpus > 0 and cfg.dataloader.divide_dataset_per_gpu:
        sampler = DistributedSampler(dataset, cfg.dist.gpus, rank)
        train_use_shuffle = False
    if mode=='train':
        if my_collate_fn is not None:
            return data_loader(
            dataset=dataset,
            batch_size=cfg.dataloader.train_batch_size,
            shuffle=train_use_shuffle,
            sampler=sampler,
            num_workers=cfg.dataloader.train_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=my_collate_fn
        ),sampler
        return data_loader(
            dataset=dataset,
            batch_size=cfg.dataloader.train_batch_size,
            shuffle=train_use_shuffle,
            sampler=sampler,
            num_workers=cfg.dataloader.train_num_workers,
            pin_memory=True,
            drop_last=True,
        ),sampler
    elif mode=='dev':
        if my_collate_fn is not None:
            return   data_loader(
            dataset=dataset,
            batch_size=cfg.dataloader.test_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=cfg.dataloader.test_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=my_collate_fn
        ),sampler
            
        return data_loader(
            dataset=dataset,
            batch_size=cfg.dataloader.test_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=cfg.dataloader.test_num_workers,
            pin_memory=True,
            drop_last=True,
        ),sampler
    elif mode in ['eval_key','eval_test',"eval_dev",'eval_train']:
        if my_collate_fn is not None:
            return data_loader(
            dataset=dataset,
            batch_size=cfg.dataloader.test_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=cfg.dataloader.test_num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=my_collate_fn,
        ),sampler
        return data_loader(
            dataset=dataset,
            batch_size=cfg.dataloader.test_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=cfg.dataloader.test_num_workers,
            pin_memory=True,
            drop_last=False,
        ),sampler
    else:
        raise ValueError(f"invalid dataloader mode {mode}")

    

