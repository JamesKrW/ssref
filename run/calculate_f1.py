import sys
sys.path.append("/share/data/mei-work/kangrui/github/ssref")
import datetime
import itertools
import os

import traceback


import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.test_utils import *
from utils.utils import *
from configs.config import Config
import itertools
from tqdm import tqdm


def setcfg(cfg):
    cfg.mode='eval_train' #'key' assert mode in ['eval_dev','eval_test','eval_train']
    save_dir="/share/data/mei-work/kangrui/github/ssref/result/pretrained_sbert"
    cfg.retrieve_count=200
    
    # basically no need to change
    cfg.dataset.test=Config()
    cfg.dataset.test.full_data="/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml/full_data_no_embed.pkl"
    cfg.src_dir = osp.join(save_dir,"test_result")
    cfg.work_dir = osp.join(save_dir,"f1_result")
    #"/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml/full_data_txt.pkl"



def main():
    # init working space and random state
    cfg=Config()
    cfg.init()
    setcfg(cfg)
    set_seed(cfg.seed)
    mkdir(cfg.work_dir)
    cfg.logger=loadLogger(cfg.work_dir,cfg.mode)

    # get full data
    with open(cfg.dataset.test.full_data, 'rb') as f: 
        id2paper = pickle.load(f)

    # get full data embedding 
    with open(os.path.join(cfg.src_dir, f'eval_key_id2embed.pkl'), 'rb') as f: 
        full_id2embed = pickle.load(f)
    # get test data embedding 
    with open(os.path.join(cfg.src_dir, f'{cfg.mode}_id2embed.pkl'), 'rb') as f: 
        test_id2embed = pickle.load(f)
    # get dev data embedding 
    

    print(f"sort ids")
    id2rank, rank2id=sort_ids(id2paper)

    print(f"build keys")
    keys=build_keys_bert(rank2id, full_id2embed)
    keys = keys.to(cfg.device)

    print(f"start search ...")
    all_pred, all_gold = list(), list()
    faiss_token = start_faiss()

    pre_dict={}
    gold_dict={}
    retrieve_count=cfg.retrieve_count
    for paperid in tqdm(list(test_id2embed.keys()), desc=f"Eval [{cfg.mode}]"): 
        gold = get_gold(id2paper[paperid])
        K = len(gold)
        if K < 1: continue
        all_gold.append(gold)
        cur_keys = keys[:id2rank[paperid]]
        query = torch.FloatTensor(test_id2embed[paperid]).unsqueeze(0)
        query = query.to(cfg.device)
        predtensor = knn(faiss_token, cur_keys, query, retrieve_count)
        predtensor = torch.squeeze(predtensor, 0).to('cpu')
        pred=[]
        for p in predtensor: 
            pred.append(rank2id[int(p)])
        all_pred.append(pred)

        pre_dict[paperid]=pred
        gold_dict[paperid]=gold

    print(f"compute eval metric ...")

    with open(osp.join(cfg.work_dir,f'{cfg.mode}_pred_{cfg.retrieve_count}.pkl'),'wb') as f:
        pickle.dump(pre_dict,f)

    with open(osp.join(cfg.work_dir,f'{cfg.mode}_gold.pkl'),'wb') as f:
        pickle.dump(gold_dict,f)

    all_pred=np.array(all_pred)
    all_gold=np.array(all_gold)
    for i in [64,128]:
        
        cfg.logger.info(f"num sampled:{i}")
        metric = get_precision_recall_f1(all_pred[:,:i], all_gold)
        cfg.logger.info(f"{metric}")

if __name__ == "__main__":
    main()