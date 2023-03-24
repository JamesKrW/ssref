import torch
from torch.utils.data import Dataset
from utils import *
import pickle
from transformers import AutoTokenizer
import random

import faiss 
from faiss.contrib.torch_utils import torch_replacement_knn_gpu as knn_gpu
import torch
import torch 
import os, sys 
import pickle
from datetime import datetime
from tqdm import tqdm
import argparse
import numpy as np
all_splits = ['train', 'dev', 'test', 'extra']

def build_keys_bert(rank2id, id2embed): 
    print("length of id2embed",len(id2embed))
    print("length of rank2id",len(rank2id))
    # assert len(rank2id) == len(id2embed)

    D = len(id2embed[rank2id[0]])
    res = list()
    for i in range(len(rank2id)): 

        try: 
            vec = id2embed[rank2id[i]]
        except: 
            vec = [0.0 for _ in range(D)]

        res.append(vec)
    res=np.array(res)
    return torch.FloatTensor(res)

def build_keys(rank2id, id2paper): 

    assert len(rank2id) == len(id2paper)

    D = len(id2paper[rank2id[0]]['embedding']['vector'])
    res = list()
    for i in range(len(rank2id)): 

        try: 
            vec = id2paper[rank2id[i]]['embedding']['vector']
        except: 
            vec = [0.0 for _ in range(D)]

        res.append(vec)

    return torch.FloatTensor(res)

def evaluate(args): 

    # read data 
    print(f"load data ...")
    id2paper, paper_ids = load_arxivaiml()
    
    # sort paper IDs so earlier papers stay higher in matrix
    # rank = position in date-ranked order
    print(f"sort ids")
    id2rank, rank2id = sort_ids(id2paper)

    print(f"build keys")
    keys = build_keys(rank2id, id2paper)

    if args.cuda: keys = keys.to(device='cuda:0')

    print(f"start search ...")
    all_pred, all_gold = list(), list()
    faiss_token = start_faiss()

    """
    TODO: optimize inference speed 
    batch query 
    each batch has B consecutive papers 
    make K <= K + B 
    for each query, find its K older papers from K + B retrievals
    # of older papers >= K cuz newer papers <= B
    """

    for paperid in tqdm(paper_ids[args.split], desc=f"Eval [{args.split}]"): 
        # loop over each paper to eval
        gold = get_gold(id2paper[paperid])
        K = len(gold)
        
        # K = max(K, 1)
        if K < 1: continue # skip papers that have NO gold refs
        
        all_gold.append(gold)
        
        # get search results 
        # only search over papers published earlier
        cur_keys = keys[:id2rank[paperid]]
        query = torch.FloatTensor(id2paper[paperid]['embedding']['vector']).unsqueeze(0)
        if args.cuda: query = query.to(device='cuda:0')

        # 1 x K 
        predtensor = knn(faiss_token, cur_keys, query, K) 
        predtensor = torch.squeeze(predtensor, 0).to('cpu') # K 
        pred = set()
        for p in predtensor: 
                pred.add(rank2id[int(p)])            
        # except: 
        #     print(f"paperId = {paperid}")
        #     print(f"K = {K}")
        #     print(f"predtensor = {predtensor}")
        #     print(f"rank = {id2rank[paperid]}")
        
        all_pred.append(pred)

    # compute precision, recall, F1 
    print(f"compute eval metric ...")
    metric = get_precision_recall_f1(all_pred, all_gold)

    for k, v in metric.items(): 
        print(f"{k} : {v}")

    print(f"save results")
    # TODO: hand-coded for now, adjust it later
    out_path = os.path.join('/share/data/mei-work/hmei/ref-search/refsearch/run')
    with open(os.path.join(out_path, f'specter_{args.split}_results.pkl'), 'wb') as f: 
        pickle.dump(
            {
                'gold': all_gold, 'pred': all_pred
            }, f
        )
    


def main(): 

    parser = argparse.ArgumentParser(description='test S2 specter model')
    parser.add_argument('--dataset', type=str, default='arxivaiml',
                        choices=['arxivaiml'],
                        help='dataset name')
    parser.add_argument('--split', type=str, default='dev', 
                        choices=['dev', 'test', 'extra'])
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--root', default='../../', help='root path where we run program')
    args = parser.parse_args()

    args.cuda = True # must use GPU

    evaluate(args)


def get_date(date, year): 
    if date is None and year is None: 
        date = '1000-01-01' # must be very old
    elif date is None and year is not None: 
        date = f'{year}-01-01'
    else: 
        date = date 
    return datetime.strptime(date, "%Y-%m-%d")

def load_arxivaiml(): 

    global all_splits
    # location of the data corpus on TTIC Cluster 
    # change to your own data dir
    data_path = os.path.join('/home-nfs/hongyuan/mei-work/refsum-data/arxiv-aiml')

    ids = dict()
    for split in all_splits: 
        with open(os.path.join(data_path, f'{split}.pkl'), 'rb') as f: 
            ids[split] = pickle.load(f)
    
    with open(os.path.join(data_path, f'full_data.pkl'), 'rb') as f: 
        full_data = pickle.load(f)
    
    return full_data, ids

def sort_ids(id2paper): 
    # id2paper : {id : metadata}
    global all_splits

    # id and date
    idndate = list()

    for i, paper in id2paper.items(): 
        date = paper['publicationDate'] # "2015-11-16"
        year = paper['year'] # 2015
        date = get_date(date, year)

        idndate.append((i, date))

    # sort id based on date
    idndate.sort(key=lambda x: x[1], reverse=False)

    id2rank, rank2id = dict(), dict()
    for r, x in enumerate(idndate): 
        rank2id[r] = x[0]
        id2rank[x[0]] = r 

    return id2rank, rank2id


def get_precision_recall_f1(all_pred, all_gold): 
    # list of lists of strings 
    # all_pred : [{'a', 'b'}, {'c', 'd', 'e'}, ...]
    # all_gold : [{'a', 'b', 'c'}, {'d', 'e'}, ...]
    all_pred=[set(item) for item in list(all_pred)]
    all_gold=[set(item) for item in list(all_gold)]
    true_pos, false_pos, false_neg = 0, 0, 0 
    
    for pred, gold in zip(all_pred, all_gold): 
        intersect = pred & gold
        
        true_pos += len(intersect)
        false_pos += len(pred) - len(intersect)
        false_neg += len(gold) - len(intersect)

    p = 1.0 * true_pos / (true_pos + false_pos)
    r = 1.0 * true_pos / (true_pos + false_neg)

    return {
        'precision': p, 'recall': r, 
        'f1': 2.0 * p * r / (p + r), 
        'true_pos': true_pos, 
        'false_pos': false_pos, 
        'false_neg': false_neg, 
        'num_sample': len(all_pred)
    }


def get_gold(paper): 
    # paper includes all necessary meta data

    res = set()

    for ref in paper['references']: 
        refid = ref['paperId']
        if refid is not None: res.add(refid)
    
    return res

def start_faiss(): 
    res = faiss.StandardGpuResources()  # use a single GPU
    res.setDefaultNullStreamAllDevices()
    #res.setTempMemory(64 * 1024 * 1024)
    return res


def knn(token, keys, query, k): 

    D, I = knn_gpu(token, query, keys, k)

    return I # M x k 


# suppose we have three id to embedding dicts:key,test_query,dev_query {'paperid':embedding}