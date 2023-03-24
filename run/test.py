import sys
sys.path.append("/share/data/mei-work/kangrui/github/ssref")
import datetime
import itertools
import os

import traceback


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from utils.utils import *
from configs.config import Config
from models.model_arch import SentenceEncoder,BertSiameseClassifier
from models.ddp_model import Model
from modules.dataloader import create_dataloader
from modules.dataset import TestDataset
import itertools
from tqdm import tqdm



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


def test_loop(rank, cfg):
    if cfg.device == torch.device('cuda') and cfg.dist.gpus != 0:
        cfg.device = torch.device("cuda", rank)
        # turn off background generator when distributed run is on
        cfg.dataloader.use_background_generator = False
        setup(cfg, rank)
        torch.cuda.set_device(cfg.device)
    if is_logging_process():
        cfg.logger=loadLogger(cfg.work_dir)
    # setup writer
   
    # make dataloader
    if is_logging_process():
        cfg.logger.info("Making dataloader...")
        # cfg.writer.add_scalar("Making train dataloader...")
    dataset = TestDataset(cfg, cfg.mode)
    data_loader = create_dataloader(cfg, cfg.mode, rank,dataset)


    # init Model
   
    #net_arch = DoubleBERT(cfg)
    #net_arch = SentenceEncoder(cfg)
    net_arch =BertSiameseClassifier(cfg)
    #net_arch=BertSiameseClassifier(cfg)
    model = Model(cfg, net_arch,None,None,None,rank)
    if cfg.usefinetuned:
        if is_logging_process():
            cfg.logger.info("load model.")
        model.load_network()
    if is_logging_process():
        cfg.logger.info("Starting testing.")
            

    try:
        embed=[]
        ids=[]
        if is_logging_process():
            pbar = tqdm(data_loader)
        else:
            pbar =data_loader
        with torch.no_grad():
            model.net.eval()
            for input,paper_ids in pbar:
                if cfg.mode=='eval_key':
                    output=model.get_key(input)
                elif cfg.mode in ['eval_dev','eval_test','eval_train']:
                    output=model.get_query(input)
                ids+=paper_ids
                embed.append(output)

        embed = torch.cat(embed, dim=0).detach().cpu().numpy()
        id2embed= {k: v for k, v in zip(ids, embed)}
        with open(osp.join(cfg.mode_dir,f'{cfg.mode}_{rank}_id2embed.pkl'),'wb') as f:
            pickle.dump(id2embed,f)

        
        if is_logging_process():
            cfg.logger.info("End of testing")
            # cfg.writer.add_scalar("End of Train")
          

    except Exception as e:
        if is_logging_process():
            cfg.logger.error(traceback.format_exc())
            # cfg.writer.add_scalar(e)
        else:
            traceback.print_exc()
    finally:
        if cfg.dist.gpus != 0:
            cleanup()

def setcfg(cfg):
    cfg.mode='eval_dev' #'key' assert mode in ['eval_key','eval_dev','eval_test','eval_train']

    save_dir='/share/data/mei-work/kangrui/github/ssref/result/pretrained_sbert'
    cfg.model.network_pth_path=osp.join(save_dir,'checkpoints/best.pt')
    cfg.usefinetuned=False
    cfg.dataloader.test_batch_size=1024
    cfg.num_workers=256


    # no need to change
    cfg.work_dir = osp.join(save_dir,"test_result")
    cfg.mode_dir = osp.join(cfg.work_dir,f"{cfg.mode}")
    cfg.dataset.test=Config()


def main():
    cfg=Config()
    cfg.init()
    setcfg(cfg)
    set_seed(cfg.seed)

    mkdir(cfg.work_dir)
    mkdir(cfg.mode_dir)

    if cfg.multi_gpu==0: 
        # single GPU
        cfg.dist.gpus=0
        cfg.dist.world_size=1
        test_loop(0, cfg)
    else:
        cfg.dist.gpus=torch.cuda.device_count()
        cfg.dist.world_size=cfg.dist.gpus
        print(cfg.dist.gpus)
        assert cfg.dist.gpus > 1
        distributed_run(test_loop, cfg)

    dist={}
    root_path=cfg.mode_dir
    for filename in os.listdir(root_path):
        if filename.startswith(f"{cfg.mode}"):
            full_path = os.path.join(root_path, filename)
            newdist=loadpickle(full_path)
            dist.update(newdist)
    with open(osp.join(cfg.work_dir,f'{cfg.mode}_id2embed.pkl'),'wb') as f:
        pickle.dump(dist,f)

    print("end of the end, embdedding generated")
    
if __name__ == "__main__":
    main()