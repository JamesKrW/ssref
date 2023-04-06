import sys
sys.path.append("/share/data/mei-work/kangrui/github/ssref")
import itertools
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import traceback


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoModel

import sys
sys.path.append("/share/data/mei-work/kangrui/github/ssref")
import os

import traceback


import torch
import torch.nn.functional as F
from utils.ddp_utils import *
from utils.utils import *
from utils.test_utils import *
from configs.config_train_recall_multi_feature import Config
from models.ddp_model import Model
from modules.dataloader import create_dataloader
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModel
from torch import nn
from torch.utils.data import Dataset
from utils.test_utils import get_precision_recall_f1

from configs.config_train_recall_multi_feature import Config
from models.ddp_model import Model
from modules.dataloader import create_dataloader
import itertools
from tqdm import tqdm




class Mydataset(Dataset):

    def __init__(self, cfg,id2txt,pred_pair):

        # get data
        self.pred_pair=pred_pair
        self.id2txt=id2txt
        

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        
        self.max_length = cfg.tokenizer.max_length
    def __len__(self):
        return len(self.pred_pair)

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
        data=self.pred_pair[index]
        query_text=self.txt_generate(self.id2txt[data['query']])
        key_text=self.txt_generate(self.id2txt[data['key']])

        query_length=int(self.max_length*0.6)
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

        outputs={
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels':torch.tensor(1, dtype=torch.float)
        }

        return outputs,data['query']+','+data['key']


class Mymodel(nn.Module):
    #self supervised learning sbert
    def __init__(self, cfg):
        super().__init__()
        self.cross_encoder =  AutoModelForSequenceClassification.from_pretrained(cfg.model_arch.name)
        self.cross_encoder.gradient_checkpointing_enable()
       
    def forward(self, input,mode=None):
        output=self.cross_encoder(**input)
        return output



def test_loop(rank, cfg):
    set_seed(cfg.seed+rank)
   
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
       
    id2txt=loadpickle(cfg.dataset.test.full_data_path)
    test_dataset=Mydataset(cfg,id2txt,cfg.pred_pair)

    if is_logging_process():
        cfg.logger.info("Making test dataloader...")
    test_loader,_ = create_dataloader(cfg=cfg, mode='eval_test', rank=rank,dataset=test_dataset)
    net_arch = Mymodel(cfg)
    model = Model(cfg=cfg, net_arch=net_arch,rank=rank,find_unused_parameters=False,eval=True)
    model.load_network()
            
    try:
        model.net.eval()
        with torch.no_grad():
            if is_logging_process():
                pbar = tqdm(test_loader, postfix=f"testing...")
            else:
                pbar = test_loader
            
            scores=[]
            pairs=[]

            for model_input, pair in pbar:
                output = model.inference(model_input)

                scores.append(output.logits.flatten())
                pairs+=pair
            
                # print(pairs)
                # print(scores)
                # sys.exit()
            scores = torch.cat(scores, dim=0).flatten().detach().cpu().numpy()
            pair2scores= {str(k): v for k, v in zip(pairs, scores)}

            with open(osp.join(cfg.mode_dir,f'{cfg.mode}_{rank}_id2embed.pkl'),'wb') as f:
                pickle.dump(pair2scores,f)

            if is_logging_process():
                cfg.logger.info("End of testing")

    except Exception as e:
        if is_logging_process():
            cfg.logger.error(traceback.format_exc())
            
        else:
            traceback.print_exc()
    finally:
        if cfg.dist.gpus != 0:
            cleanup()


def get_score(cfg):
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
    with open(osp.join(cfg.work_dir,f'{cfg.mode}_pair2score.pkl'),'wb') as f:
        pickle.dump(dist,f)

    print("end of the end, embdedding generated")
    return dist


def setcfg(cfg,mode,save_dir_path):
    cfg.mode=mode #'key' assert mode in ['eval_key','eval_dev','eval_test','eval_train']

    save_dir=save_dir_path
    cfg.model.network_pth_path=osp.join(save_dir,'checkpoints/best.pt')
    cfg.model.resume_state_path=osp.join(save_dir,'checkpoints/best.state')
    cfg.usefinetuned=True
    cfg.num_workers=2


    # no need to change
    cfg.work_dir = osp.join(save_dir,"test_result")
    cfg.mode_dir = osp.join(cfg.work_dir,f"{cfg.mode}")
    cfg.dataset.test=Config()

    cfg.dataset.test.datafolder="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml-small"
    # which contains dev.pkl,test.pkl
    cfg.dataset.test.full_data_path="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml/full_data_abs_title_author_data.pkl"
    cfg.dataset.test.pred_path=f"/share/data/mei-work/kangrui/github/ssref/result/pretrained_pair_sbert/f1_result/{cfg.mode}_pred_1000.pkl"
    cfg.dataset.test.gold_path=f"/share/data/mei-work/kangrui/github/ssref/result/pretrained_pair_sbert/f1_result/{cfg.mode}_gold.pkl"


def rerank(candidate_set,pair2score,rerank_num=None,device=None):
    cnt=0
    total_mean=0
    mean_score=0
    pbar = tqdm(range(0,len(candidate_set),1), postfix=f"mean score:{mean_score}")
    query=list(candidate_set.keys())
    

    if rerank_num is None:
        for v in candidate_set.values():
            if rerank_num is None or rerank_num>len(v):
                rerank_num=len(v)
                
    rerank_candidate={}   
    for i in pbar:
        query_paper=query[i]
        
        score=[pair2score[query_paper+','+k] for k in candidate_set[query_paper]]
               
        
        
        
        score=torch.tensor(score).to(device)
        mean_score=torch.mean(score)
        total_mean+=mean_score
        top_k=torch.topk(score,k=rerank_num)[1]

#         print(pred_logits[0][top_k[-5:]])
        rerank=[candidate_set[query_paper][idx.cpu().item()] for idx in top_k]
        rerank_candidate[query_paper]=rerank
        pbar.postfix=f"mean score:{mean_score}"
    print("total_mean",total_mean/len(pbar))
    print("min rerank num->",rerank_num)
    return rerank_candidate

    
def main(save_dir,mode):
    cfg=Config()
    cfg.init()
    setcfg(cfg,mode,save_dir)

    candidate_set={}

    path="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml/full_id_abs_title_cite_ref_author.pkl"
    full_data=loadpickle(path)
    pred_set=loadpickle(cfg.dataset.test.pred_path)
    gold_set=loadpickle(cfg.dataset.test.gold_path)



    candidate_set={}
    pred_pair=[]
    for k,v in pred_set.items():
        if k not in full_data.keys():
            continue
        cur=set()
        for paper in v[:200]:
            if paper not in full_data.keys():
                continue
            data={'query':k,'key':paper}
            pred_pair.append(data)
            cur.add(paper)
            for ref in full_data[paper]["references"]:
                if ref not in full_data.keys():
                    continue
                data={'query':k,'key':ref}
                pred_pair.append(data)
                cur.add(ref)
        candidate_set[k]=list(cur)
    print(len(candidate_set))
    print(len(pred_pair))

    score_path=osp.join(osp.join(save_dir,'test_result'),f'{mode}_pair2score.pkl')
    if not osp.exists(score_path):
        cfg.pred_pair=pred_pair
        pair2score=get_score(cfg)
    else:
        pair2score=loadpickle(score_path)
       
    # print(pair2score.keys())
    score=0
    total=0
    for k,v in gold_set.items():
        for pid in v:
            if k+','+pid in pair2score.keys():
                score+=pair2score[k+','+pid]
                total+=1
    print(score/total)

    total_len=0
    for k,v in candidate_set.items():
        total_len+=len(v)
    print(total_len)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rerank_candidate=rerank(candidate_set,pair2score,rerank_num=None,device=device)
    show_rerank_rst(rerank_candidate,gold_set)

if __name__ == "__main__":
   save_dir='/share/data/mei-work/kangrui/github/ssref/result/cross-encoder_ms-marco-TinyBERT-L-2/2023-04-04T23-50-34'
   main(save_dir,mode='eval_test')