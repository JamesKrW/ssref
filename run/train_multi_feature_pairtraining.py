import sys
sys.path.append("/share/data/mei-work/kangrui/github/ssref")
import itertools
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import traceback


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoModel


from utils.utils import *
from utils.ddp_utils import *
from configs.config_train_recall_multi_feature import Config
from models.ddp_model import Model
from modules.dataloader import create_dataloader
from transformers import get_scheduler
from torch.optim import AdamW
import itertools
from tqdm import tqdm
from torch.distributed import ReduceOp
import math
from torch.utils.data import Dataset
from torch import nn


class Mydataset(Dataset):

    def __init__(self, cfg,mode,id2txt):

        # get data
        assert mode in ['train','dev']
        if mode ==  'train':
            cfg.dataset.datafolder="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml"
        else:
            cfg.dataset.datafolder="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml-small"
        self.cite_pair=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_cite_pair_label.pkl"))
        self.id2txt=id2txt
        random.shuffle(self.cite_pair)

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        
        self.max_length = cfg.tokenizer.max_length
    def __len__(self):
        return len(self.cite_pair)

    def txt_generate(self,data):
        sep_token=self.tokenizer.sep_token
        para=""
        # para+=data['paperID']+'\n'
        para+=data['title']+sep_token
        authortxt=' and '.join([item['name'] for item in data['authors']])
        para+=authortxt+sep_token
        para+=data['abstract']
        return para
    
    def truncate_sentence(self,sentence, max_length):
        tokens = sentence.split()
        truncated_tokens = tokens[:max_length]
        truncated_sentence = " ".join(truncated_tokens)
        return truncated_sentence

    def __getitem__(self, index):
        data=self.cite_pair[index]
        query_text=self.txt_generate(self.id2txt[data['query']])
        key_text=self.txt_generate(self.id2txt[data['key']])
        label=data['label']

        query_length=int(self.max_length*0.5)

        query_text=self.truncate_sentence(query_text, query_length)

        inputs=self.tokenizer.encode_plus(
            query_text,
            key_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        data={
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels':torch.tensor(label, dtype=torch.float)
        }

        return data,torch.tensor(label, dtype=torch.long)

class Myloss(nn.Module):

    def __init__(self,cfg):
        super().__init__()

    def forward(self, output,target,mode=None):
        # print(output)
        loss=output.loss
        if mode is not None:
            if mode == 'test':
                logits=output.logits.flatten()
                pos_score=torch.sum(logits[torch.nonzero(target == 1).squeeze()])
                neg_score=torch.sum(logits[torch.nonzero(target == 0).squeeze()])
                pos_cnt=torch.sum((target == 1))
                neg_cnt=torch.sum((target == 0))
            return loss,pos_score,pos_cnt,neg_score,neg_cnt
        return loss


class Mymodel(nn.Module):
    #self supervised learning sbert
    def __init__(self, cfg):
        super().__init__()
        self.cross_encoder =  AutoModelForSequenceClassification.from_pretrained(cfg.model_arch.name)
        self.cross_encoder.gradient_checkpointing_enable()
       
    def forward(self, input,mode=None):
        output=self.cross_encoder(**input)
        return output

def test_model(cfg,model,test_loader):
    model.net.eval()
    test_loss = 0
    test_loop_len = 0
    test_pos_score=0
    test_neg_score=0
    test_pos_cnt=0
    test_neg_cnt=0
    with torch.no_grad():
        if is_logging_process():
            pbar = tqdm(test_loader, postfix=f"testing...")
        else:
            pbar = test_loader
        for model_input, model_target in pbar:
            output = model.inference(model_input)
            # print("model_target->",model_target)
            loss_v,positive_score,pos_cnt,negative_score,neg_cnt = model.loss_f(output, model_target.to(cfg.device),mode='test')
            if cfg.dist.gpus > 0:
                # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                dist.all_reduce(loss_v)
                dist.all_reduce(positive_score)
                dist.all_reduce(negative_score)
                dist.all_reduce(pos_cnt)
                dist.all_reduce(neg_cnt)
                loss_v /= torch.tensor(float(cfg.dist.gpus))
            test_pos_score+=positive_score.to("cpu").item()
            test_neg_score+=negative_score.to("cpu").item()
            test_pos_cnt+=pos_cnt.to("cpu").item()
            test_neg_cnt+=neg_cnt.to("cpu").item()
            test_loss += loss_v.to("cpu").item()
            test_loop_len += 1
        test_loss/=test_loop_len
        test_pos_score /= (test_pos_cnt+1e-6)
        test_neg_score /= (test_neg_cnt+1e-6)
        if is_logging_process():
            cfg.logger.info(
                "Test Loss %.08f at (epoch: %d / step: %d)"
                % (test_loss, model.epoch + 1, model.step)
            )
            cfg.logger.info(
                "Test pos_score %.08f at (epoch: %d / step: %d)"
                % (test_pos_score, model.epoch + 1, model.step)
            )
            cfg.logger.info(
                "Test neg_score %.08f at (epoch: %d / step: %d)"
                % (test_neg_score, model.epoch + 1, model.step)
            )
    return test_loss


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
        if model.step % cfg.log_interval_test == 0 or model.step==1:
            test_rst=test_model(cfg,model,test_loader)
            if is_logging_process():
                
                if model.best_test_rst is None or test_rst<model.best_test_rst:
                    model.save_network(name='best')
                    model.save_training_state(name='best')
                    model.best_test_rst=test_rst






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
   
    if is_logging_process():
        cfg.logger.info("Making train dataloader...")
       
    id2txt=loadpickle("/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml/full_data_abs_title_author_data.pkl")
    train_dataset=Mydataset(cfg,"train",id2txt)
    test_dataset=Mydataset(cfg,"dev",id2txt)

    train_loader,train_sampler = create_dataloader(cfg, "train", rank,train_dataset)
    if is_logging_process():
        cfg.logger.info("Making test dataloader...")
    test_loader,_ = create_dataloader(cfg, "dev", rank,test_dataset)

    # init Model
    cfg.scheduler.num_training_steps = cfg.num_epoch * len(train_loader)

    
   
    net_arch = Mymodel(cfg)
    loss_f = Myloss(cfg)
    optimizer=AdamW
    scheduler=get_scheduler
    model = Model(cfg=cfg, net_arch=net_arch,loss_f=loss_f,optimizer=optimizer,scheduler=scheduler,rank=rank,find_unused_parameters=False)

    
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
            train_sampler.set_epoch(model.epoch)
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
        port= 8888 
        while is_port_in_use(port):
            port += 1
        cfg.dist.master_port=str(port)
        cfg.dist.gpus=torch.cuda.device_count()
        cfg.dist.world_size=cfg.dist.gpus
        print(cfg.dist.gpus)
        assert cfg.dist.gpus > 1
        distributed_run(train_loop, cfg)
if __name__ == "__main__":
    main()