import sys
sys.path.append("/share/data/mei-work/kangrui/github/ssref")
import datetime
import itertools
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import traceback


import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from utils.utils import *
from configs.config import Config
from models.model_arch import pair_sbert
from models.ddp_model import Model
from modules.dataloader import create_dataloader
from modules.dataset import ArxivDataset,SSDataset
from modules.loss import NTXent,SSloss
from transformers import get_scheduler
from torch.optim import AdamW
import itertools
from tqdm import tqdm
from torch.distributed import ReduceOp
import math
from calculate_topN_and_reference_f1 import calculate_recall
from transformers import AutoTokenizer
def test_model(cfg,model,test_loader):
    model.net.eval()
    test_loss = 0
    test_loop_len = 0
    with torch.no_grad():
        if is_logging_process():
            pbar = tqdm(test_loader, postfix=f"testing...")
        else:
            pbar = test_loader
        for model_input, model_target in pbar:
            output = model.inference(model_input)
            # print("model_target->",model_target)
            loss_v = model.loss_f(output, model_target.to(cfg.device))
            if cfg.dist.gpus > 0:
                # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                dist.all_reduce(loss_v)
                loss_v /= torch.tensor(float(cfg.dist.gpus))
            test_loss += loss_v.to("cpu").item()
            test_loop_len += 1

        test_loss /= test_loop_len

        if is_logging_process():
            cfg.logger.info(
                "Test Loss %.08f at (epoch: %d / step: %d)"
                % (test_loss, model.epoch + 1, model.step)
            )
    return test_loss

def test_recall(cfg,model,batch_size,tokenizer,query,key_id2embed,id2paper,device,retrieve_count=1000,logger=None):
    query_id2embed={} # embedding for query
    i=0
    pbar = tqdm(range(0,len(query),batch_size), postfix=f"generaten query embedding")
    model.net.eval()
    with torch.no_grad():
        for i in pbar:
            right=i
            left=min(i+batch_size,len(query))
            sentences=[paperid2abstract(paperid,id2paper) for paperid in query[right:left]]
            _input=tokenize(sentences,tokenizer,max_length=512)
                     
            output=model.get_query(_input) #B*L
            for idx,paperid in enumerate(query[right:left]):
                query_id2embed[paperid]=output[idx].cpu().numpy()
    
    calculate_recall(query_id2embed=query_id2embed,key_id2embed=key_id2embed,
                     id2paper=id2paper,device=device,retrieve_count=retrieve_count,logger=logger)



def train_model(cfg, model, train_loader,test_loader=None):
    model.net.train()

    if is_logging_process():
        pbar = tqdm(train_loader, postfix=f"training..., loss: {model.loss_v:.08f}")
    else:
        pbar = train_loader

    train_loss=0
    for model_input, model_target in pbar:
        model.optimize_parameters(model_input, model_target)
        loss = model.loss_v
        train_loss+=loss
        if is_logging_process():
            pbar.postfix = f"loss: {model.loss_v:.08f}"

        model.step += 1

        if is_logging_process() and (loss > 1e8 or math.isnan(loss)):
            cfg.logger.error("Loss exploded to %.02f at step %d!" % (loss, model.step))
            raise Exception("Loss exploded")

        if model.step % cfg.log_interval_train == 0:
            train_loss_avg = torch.zeros((1,), dtype=torch.float32).cuda().fill_(train_loss)
            if cfg.dist.gpus > 0:
                dist.all_reduce(train_loss_avg, op=ReduceOp.SUM)
            train_loss = train_loss_avg.item() / cfg.dist.world_size
            cur_loss = train_loss / cfg.log_interval_train
            if is_logging_process():
                cfg.logger.info("Train Loss %.08f at (epoch: %d / step: %d)"% (cur_loss, model.epoch + 1, model.step))
            train_loss=0
        dist.barrier()
        if model.step % cfg.log_interval_recall == 0 or model.step==1:
            if is_logging_process():
                print("begin eval recall")
                if cfg.id2paper is None:
                    cfg.id2paper=loadpickle(cfg.sstraining.id2paper)
                    cfg.key_id2embed=loadpickle(cfg.sstraining.key_id2embed)
                    cfg.query=loadpickle(cfg.sstraining.query)
                    print("load full data finished")
                dist.barrier()
                test_recall(cfg=cfg,model=model,batch_size=cfg.dataloader.test_batch_size,tokenizer=cfg.tokenizer.instance,
                            query=cfg.query,key_id2embed=cfg.key_id2embed,id2paper=cfg.id2paper,device=cfg.device,retrieve_count=1000,logger=cfg.logger)

            else:
                dist.barrier()
        if model.step % cfg.log_interval_test == 0 or model.step==1:
            test_model(cfg,model,test_loader)




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


def train_loop(rank, cfg):
    set_seed(cfg.seed+rank)
    cfg.tokenizer.instance=AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    if cfg.device == torch.device('cuda') and cfg.dist.gpus != 0:
        cfg.device = torch.device("cuda", rank)
        
        cfg.dataloader.use_background_generator = False
        setup(cfg, rank)
        torch.cuda.set_device(cfg.device)
    if is_logging_process():
        cfg.logger=loadLogger(cfg.work_dir)
        cfg.id2paper=None
        cfg.key_id2embed=None
        cfg.query=None
    
        # cfg.logger=loadLogger(cfg.work_dir)
        # cfg.id2paper=loadpickle(cfg.sstraining.id2paper)
        # cfg.key_id2embed=loadpickle(cfg.sstraining.key_id2embed)
        # cfg.query=loadpickle(cfg.sstraining.query)
    
   
    if is_logging_process():
        cfg.logger.info("Making train dataloader...")
       
    train_dataset=SSDataset(cfg,"train")
    test_dataset=SSDataset(cfg,"dev")

    train_loader = create_dataloader(cfg, "train", rank,train_dataset)
    if is_logging_process():
        cfg.logger.info("Making test dataloader...")
    test_loader = create_dataloader(cfg, "dev", rank,test_dataset)

    # init Model
    cfg.scheduler.num_training_steps = cfg.num_epoch * len(train_loader)

    
   
    net_arch = pair_sbert(cfg)
    loss_f = SSloss(cfg)
    optimizer=AdamW
    scheduler=get_scheduler
    model = Model(cfg, net_arch,loss_f,optimizer,scheduler,rank)

    
    if cfg.model.resume_state_path is not None:
        model.load_training_state()
    elif cfg.model.network_pth_path is not None:
        model.load_network()
    else:
        if is_logging_process():
            cfg.logger.info("Starting new training run.")
            

    try:
        if cfg.dist.gpus == 0 or cfg.dataloader.divide_dataset_per_gpu:
            epoch_step = 1
        else:
            epoch_step = cfg.dist.gpus
        for model.epoch in itertools.count(model.epoch + 1, epoch_step):
            if model.epoch > cfg.num_epoch:
                break
            train_model(cfg, model, train_loader,test_loader)
            if model.epoch % cfg.chkpt_interval == 0:
                if is_logging_process():
                    model.save_network()
                    model.save_training_state()

            test_rst=test_model(cfg, model, test_loader)
            if is_logging_process():
                if model.best_test_rst is None or test_rst<model.best_test_rst:
                    model.save_network(name='best')
                    model.save_training_state(name='best')
                    model.best_test_rst=test_rst

        if is_logging_process():
            cfg.logger.info("End of Train")
           

    except Exception as e:
        if is_logging_process():
            cfg.logger.error(traceback.format_exc())
            
        else:
            traceback.print_exc()
    finally:
        if cfg.dist.gpus != 0:
            cleanup()



def main():
    cfg=Config()
    cfg.init()
    mkdir(cfg.chkpt_dir)
    mkdir(cfg.work_dir)
    
    if not cfg.multi_gpu: 
        # single GPU
        cfg.dist.gpus=0
        cfg.dist.world_size=1
        train_loop(0, cfg)
    else:
        cfg.dist.gpus=torch.cuda.device_count()
        cfg.dist.world_size=cfg.dist.gpus
        print(cfg.dist.gpus)
        assert cfg.dist.gpus > 1
        distributed_run(train_loop, cfg)
if __name__ == "__main__":
    main()