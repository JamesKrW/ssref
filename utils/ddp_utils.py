import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
import os
import socket

def is_logging_process(eval=None):
    # print(dist.get_rank())
    if eval is not None:
        return False
    return (not dist.is_initialized()) or dist.get_rank() == 0

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
    
def setup(cfg, rank):
    os.environ["MASTER_ADDR"] = cfg.dist.master_addr
    os.environ["MASTER_PORT"] = cfg.dist.master_port
    timeout_sec = 10
    if cfg.dist.timeout is not None:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        timeout_sec = cfg.dist.timeout
    timeout = datetime.timedelta(seconds=timeout_sec)

    # initialize the process group
    dist.init_process_group(
        backend=cfg.dist.backend,
        init_method=cfg.dist.init_method, 
        rank=rank,
        world_size=cfg.dist.world_size,
        timeout=timeout,
    )


def cleanup():
    dist.destroy_process_group()


def distributed_run(fn, cfg):
    mp.spawn(fn, args=(cfg,), nprocs=cfg.dist.gpus, join=True)