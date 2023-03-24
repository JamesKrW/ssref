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
    cfg.mode='eval_test' #'key' assert mode in ['eval_dev','eval_test']

    save_dir='/share/data/mei-work/kangrui/github/ssref/result/pretrained_sbert_mo'
    cfg.src_dir = osp.join(save_dir,"test_result")
    cfg.work_dir = osp.join(save_dir,"f1_result")
    cfg.work_dir = osp.join(save_dir,"f1_result")

    cfg.dataset.test=Config()
    cfg.dataset.test.full_data="/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml/full_data_no_embed.pkl"
    #"/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml/full_data_txt.pkl"
    cfg.retrieve_count=1000


def calculate_recall(query_id2embed,key_id2embed,id2paper,device,retrieve_count=1000,logger=None):
    print(f"sort ids")
    id2rank, rank2id=sort_ids(id2paper)
    keys=build_keys_bert(rank2id, key_id2embed)
    keys = keys.to(device)

    print(f"start search ...")
    all_pred, all_gold = list(), list()
    faiss_token = start_faiss()
    pre_dict={}
    gold_dict={}
    for paperid in tqdm(list(query_id2embed.keys()), desc=f"Eval recall"): 
        gold = get_gold(id2paper[paperid])
        K = len(gold)
        if K < 1: continue
        all_gold.append(gold)
        cur_keys = keys[:id2rank[paperid]]
        query = torch.FloatTensor(query_id2embed[paperid]).unsqueeze(0)
        query = query.to(device)
        predtensor = knn(faiss_token, cur_keys, query, retrieve_count)
        predtensor = torch.squeeze(predtensor, 0).to('cpu')
        pred=[]
        for p in predtensor: 
            pred.append(rank2id[int(p)])
        all_pred.append(pred)

        pre_dict[paperid]=pred
        gold_dict[paperid]=gold

    print(f"compute eval metric ...")
    i=100
    while i<retrieve_count:
        all_pred_topN=[]
        for paperlist in all_pred:
            topN=paperlist[:i]
            topNandReference=[]
            topNandReference+=list(topN)
            for paperid in list(topN):
                 topNandReference+=[item["paperId"] for item in id2paper[paperid]["references"]]
            all_pred_topN.append( topNandReference)
        all_pred_topN=np.array(all_pred_topN)
        metric = get_precision_recall_f1(all_pred_topN, all_gold)
        if logger is not None:
            logger.info(f"top{i}:\n{metric}")
        else:
            print(f"top{i}:\n{metric}")
        i*=2

def main():
    # init working space and random state
    cfg=Config()
    cfg.init()
    setcfg(cfg)
    set_seed(cfg.seed)
    mkdir(cfg.work_dir)
    cfg.logger=loadLogger(cfg.work_dir,f'top{cfg.retrieve_count}{cfg.mode}')

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
    

    calculate_recall(query_id2embed=test_id2embed,key_id2embed=full_id2embed,id2paper=id2paper,device=cfg.device,retrieve_count=1000,logger=cfg.logger)

if __name__ == "__main__":
    main()