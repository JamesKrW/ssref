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
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from torch import nn
import numpy as np
import torch.nn.functional as F

class Mydataset(Dataset):

    def __init__(self, cfg,mode,id2txt,sample_num=5):

        # get data
        assert mode in ['train','dev']
        if mode ==  'train':
            cfg.dataset.datafolder="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml"
        else:
            cfg.dataset.datafolder="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml-small"
        self.cite_pair=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_cite_pair.pkl"))
        self.id2txt=id2txt
        random.shuffle(self.cite_pair)
       
        
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        self.tokenizer = tokenizer
        
        self.max_length = cfg.tokenizer.max_length
        self.sample_num=sample_num
        self.id=list(self.id2txt.keys())

    def __len__(self):
        return len(self.cite_pair)

    def txt_generate(self,data):
        para=""
        # para+=data['paperID']+'[SEP]'
        para+=data['title']+'[SEP]'
        authortxt=' and '.join([item['name'] for item in data['authors']])
        para+=authortxt+'[SEP]'
        para+=data['abstract']
        return para

    def __getitem__(self, index):
        # print(self.cite_pair[index])
        #positive pair
        query_id=self.cite_pair[index][0]
        key_id=self.cite_pair[index][1]

        key_id_list=[]
        key_ids=[]
        key_mask=[]
        key_token_type_ids=[]
        key_id_list.append(key_id)
        #negative pair
        i=0
        while i<self.sample_num:
            random_id = random.choice(self.id)
            if random_id not in self.id2txt[query_id]['references']:
                key_id_list.append(random_id)
                i+=1
        for key in key_id_list:
            key_text = self.txt_generate(self.id2txt[key])
            key_inputs = self.tokenizer.encode_plus(
            key_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
            key_ids.append(key_inputs['input_ids'])
            key_mask.append(key_inputs['attention_mask'])
            key_token_type_ids.append( key_inputs["token_type_ids"])
    
        query_text=self.txt_generate(self.id2txt[query_id])
        


        query_inputs = self.tokenizer.encode_plus(
            query_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        query_ids = [query_inputs['input_ids'] for i in range(self.sample_num+1)]
        query_mask = [query_inputs['attention_mask'] for i in range(self.sample_num+1)]
        query_token_type_ids = [query_inputs["token_type_ids"] for i in range(self.sample_num+1)]

        key={
            'ids': [torch.tensor(item, dtype=torch.long) for item in key_ids],
            'mask': [torch.tensor(item, dtype=torch.long) for item in key_mask],
            'token_type_ids': [torch.tensor(item, dtype=torch.long) for item in key_token_type_ids],
        }

        query={
            'ids': [torch.tensor(item, dtype=torch.long) for item in query_ids],
            'mask': [torch.tensor(item, dtype=torch.long) for item in query_mask],
            'token_type_ids': [torch.tensor(item, dtype=torch.long) for item in query_token_type_ids],
        }
        data={
            'key':key,
            'query':query
        }
        target=[0 for i in range(self.sample_num+1)]
        target[0]=1
        return data,torch.tensor(target, dtype=torch.long)

def my_collate_fn(batch):
    key_ids=[]
    key_mask=[]
    key_token_type_ids=[]

    query_ids=[]
    query_mask=[]
    query_token_type_ids=[]

    for item in batch:
        
       
        key=item[0]['key']

        for subitem in key['ids']:
            key_ids.append(subitem)
        for subitem in key['mask']:
            key_mask.append(subitem)
        for subitem in key['token_type_ids']:
            key_token_type_ids.append(subitem)
        

        query=item[0]['query']
        for subitem in query['ids']:
            query_ids.append(subitem)
        for subitem in query['mask']:
            query_mask.append(subitem)
        for subitem in query['token_type_ids']:
            query_token_type_ids.append(subitem)
    
    key_ids= torch.stack(key_ids, dim=0)
    key_mask= torch.stack(key_mask, dim=0)
    key_token_type_ids= torch.stack(key_token_type_ids, dim=0)
    query_ids= torch.stack(query_ids, dim=0)
    query_mask= torch.stack(query_mask, dim=0)
    query_token_type_ids= torch.stack(query_token_type_ids, dim=0)
    

    data={
            'key':{
                'ids': key_ids,
                'mask': key_mask,
                'token_type_ids': key_token_type_ids,
        
            },
            'query':{
                'ids': query_ids,
                'mask': query_mask,
                'token_type_ids': query_token_type_ids,
            }
        }
    
    target = []
    for item in batch:
        for t in item[1]:
            target.append(t)
    
    target = torch.tensor(target, dtype=torch.long)
    return data, target

class Myloss(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.device=cfg.device
       

    def forward(self, output,target,mode=None):
        query=output['query']
       
        key=output['key']
        rst=torch.einsum('ij,ij->i', query, key)
        sig_rst=torch.sigmoid(rst)

        pos_num=torch.sum(target == 1)
        neg_num=torch.sum(target == 0)
        loss1=torch.sum(-(1-target)*torch.log(1-sig_rst))
        loss2=torch.sum(-target*torch.log(sig_rst))
        if mode=='test':
            pos_mean=torch.mean(rst[torch.nonzero(target == 1).squeeze()])
            neg_mean=torch.mean(rst[torch.nonzero(target == 0).squeeze()])
            return (loss1/neg_num+loss2/pos_num),pos_mean,neg_mean
        return loss1/neg_num+loss2/pos_num

class SentenceEncoder(nn.Module):
    #self supervised learning sbert
    def __init__(self, model_arch_name,checkpoint_enable):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_arch_name)
        if checkpoint_enable: 
            self.pretrained_model.gradient_checkpointing_enable()
        
    def mean_pooling(self,model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids,attention_mask,token_type_ids=None,mode='query'):
        model_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        if mode=='query':
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    

class Mymodel(nn.Module):
    #self supervised learning sbert
    def __init__(self, cfg):
        super().__init__()
        self.key_encoder = SentenceEncoder(cfg.model_arch.name,checkpoint_enable=False)
        self.query_encoder = SentenceEncoder(cfg.model_arch.name,checkpoint_enable=False)
        for param in self.query_encoder.parameters():
                param.requires_grad = False
       
       

    def forward(self, input,mode=None):

        
        if mode!=None:
            s=input['ids']
            ms=input['mask']
            tks=input['token_type_ids']
            if mode=='query':
                 return self.query_encoder(s,ms,tks,'query')
            else:
                 return self.key_encoder(s,ms,tks,'key')
        else:
            ss=input['query']['ids']
            sms=input['query']['mask']
            stks=input['query']['token_type_ids']
            
            

            ts=input['key']['ids']
            tms=input['key']['mask']
            ttks=input['key']['token_type_ids']

            query=self.query_encoder(ss,sms,stks,'query')
            key=self.key_encoder(ts,tms,ttks,'key')
        

            return  {'key':key,'query':query}

def test_model(cfg,model,test_loader):
    model.net.eval()
    test_loss = 0
    test_loop_len = 0
    test_pos=0
    test_neg=0
    with torch.no_grad():
        if is_logging_process():
            pbar = tqdm(test_loader, postfix=f"testing...")
        else:
            pbar = test_loader
        for model_input, model_target in pbar:
            output = model.inference(model_input)
            # print("model_target->",model_target)
            loss_v,positive_score,negative_score = model.loss_f(output, model_target.to(cfg.device),mode='test')
            if cfg.dist.gpus > 0:
                # Aggregate loss_v from all GPUs. loss_v is set as the sum of all GPUs' loss_v.
                dist.all_reduce(loss_v)
                dist.all_reduce(positive_score)
                dist.all_reduce(negative_score)
                loss_v /= torch.tensor(float(cfg.dist.gpus))
                positive_score/=torch.tensor(float(cfg.dist.gpus))
                negative_score/=torch.tensor(float(cfg.dist.gpus))
            test_pos+=positive_score.to("cpu").item()
            test_neg+=negative_score.to("cpu").item()
            test_loss += loss_v.to("cpu").item()
            test_loop_len += 1

        test_loss /= test_loop_len
        test_pos/=test_loop_len
        test_neg/=test_loop_len
        if is_logging_process():
            cfg.logger.info(
                "Test Loss %.08f at (epoch: %d / step: %d)"
                % (test_loss, model.epoch + 1, model.step)
            )
            cfg.logger.info(
                "Test pos_score %.08f at (epoch: %d / step: %d)"
                % (test_pos, model.epoch + 1, model.step)
            )
            cfg.logger.info(
                "Test neg_score %.08f at (epoch: %d / step: %d)"
                % (test_neg, model.epoch + 1, model.step)
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
    
    id2txt=loadpickle("/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml/full_id_abs_title_cite_ref_author.pkl")
    train_dataset=Mydataset(cfg,"train",id2txt)
    test_dataset=Mydataset(cfg,"dev",id2txt)

    train_loader,_ = create_dataloader(cfg, "train", rank,train_dataset,my_collate_fn=my_collate_fn)
    if is_logging_process():
        cfg.logger.info("Making test dataloader...")
    test_loader,_ = create_dataloader(cfg, "dev", rank,test_dataset,my_collate_fn=my_collate_fn)

    # init Model
    cfg.scheduler.num_training_steps = cfg.num_epoch * len(train_loader)

    
   
    net_arch = Mymodel(cfg)
    loss_f = Myloss(cfg)
    optimizer=AdamW
    scheduler=get_scheduler
    model = Model(cfg=cfg, net_arch=net_arch,loss_f=loss_f,optimizer=optimizer,scheduler=scheduler,rank=rank)

    
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
    
    if cfg.multi_gpu!=1: 
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