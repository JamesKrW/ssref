import sys
sys.path.append("/share/data/mei-work/kangrui/github/ssref")
import datetime
import itertools
import os

import traceback


import torch
import torch.nn.functional as F
from utils.ddp_utils import *
from utils.utils import *
from configs.config import Config
from models.model_arch import SentenceEncoder,BertSiameseClassifier
from models.ddp_model import Model
from modules.dataloader import create_dataloader
from modules.dataset import TestDataset
import itertools
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModel
from torch import nn
from torch.utils.data import Dataset
def setcfg(cfg):
    cfg.mode='eval_dev' #'key' assert mode in ['eval_key','eval_dev','eval_test','eval_train']

    save_dir='/share/data/mei-work/kangrui/github/ssref/result/sentence-transformers_all-MiniLM-L6-v2/2023-04-03T04-11-59'
    cfg.model.network_pth_path=osp.join(save_dir,'checkpoints/best.pt')
    cfg.usefinetuned=True
    cfg.num_workers=2


    # no need to change
    cfg.work_dir = osp.join(save_dir,"test_result")
    cfg.mode_dir = osp.join(cfg.work_dir,f"{cfg.mode}")
    cfg.dataset.test=Config()

    cfg.dataset.test.datafolder="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml-small"
    # which contains dev.pkl,test.pkl
    cfg.dataset.test.full_data_path="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml/full_data_abs_title_author_data.pkl"
    #"/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml/full_data_txt.pkl"

class MyDataset(Dataset):

    def __init__(self, cfg,mode):
        print(mode)
        # get data
        assert mode in ['eval_key','eval_test','eval_dev','eval_train']
        
        if mode=='eval_key':
            self.id2txt=loadpickle(cfg.dataset.test.full_data_path)
            self.id=list(self.id2txt.keys())
        elif mode in ['eval_test','eval_dev','eval_train']:
            mode=mode.split('_')[1]
            self.id2txt=loadpickle(osp.join(cfg.dataset.test.datafolder,f"{mode}_filtered_abs_title_author_data.pkl"))
            self.id=loadpickle(osp.join(cfg.dataset.test.datafolder,f"{mode}_filtered.pkl"))
        
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        self.tokenizer = tokenizer
        
        self.max_length =cfg.tokenizer.max_length
        # self.empt=0

    def txt_generate(self,data):
        para=""
        # para+=data['paperID']+'[SEP]'
        para+=data['title']+'[SEP]'
        authortxt=' and '.join([item['name'] for item in data['authors']])
        para+=authortxt+'[SEP]'
        para+=data['abstract']
        return para
    
    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        # print(self.cite_pair[index])
      
        key_id=self.id[index]
        key_text = self.txt_generate(self.id2txt[key_id])
        
        _inputs = self.tokenizer.encode_plus(
            key_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        key_ids = _inputs['input_ids']
        key_mask = _inputs['attention_mask']
        key_token_type_ids = _inputs["token_type_ids"]

        key={
            'ids': torch.tensor(key_ids, dtype=torch.long),
            'mask': torch.tensor(key_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(key_token_type_ids, dtype=torch.long),
        }

        return key,key_id
    
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


def test_loop(rank, cfg):
    if cfg.device == torch.device('cuda') and cfg.dist.gpus != 0:
        cfg.device = torch.device("cuda", rank)
        # turn off background generator when distributed run is on
        cfg.dataloader.use_background_generator = False
        setup(cfg, rank)
        torch.cuda.set_device(cfg.device)
    if is_logging_process():
        cfg.logger=loadLogger(cfg.work_dir,cfg.mode)
    # setup writer
   
    # make dataloader
    if is_logging_process():
        cfg.logger.info("Making dataloader...")
        # cfg.writer.add_scalar("Making train dataloader...")
    dataset = MyDataset(cfg, cfg.mode)
    data_loader,_ = create_dataloader(cfg, cfg.mode, rank,dataset)



    net_arch =Mymodel(cfg)
    model = Model(cfg=cfg, net_arch=net_arch,rank=rank)
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