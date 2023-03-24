import torch
from torch.utils.data import Dataset
from utils.utils import *
import pickle
from transformers import AutoTokenizer
import random
    
    
class NXTENTDataset(Dataset):

    def __init__(self, cfg,mode):

        #get arxiv_dict
        arxiv_dict=loadjson(cfg.dataset.arxiv_json)
        self.data = arxiv_dict
        #get cite_pair
        with open(cfg.dataset.cited_pair,'rb') as f:
            cite_pair=pickle.load(f)
        

        self.mode = mode
        train_size=int(len(cite_pair)*cfg.dataset.ratio)
        if mode =="train":
           self.cite_pair=cite_pair[:train_size]
        elif mode=="test":
            self.cite_pair=cite_pair[train_size:]
        else:
            raise ValueError(f"invalid dataloader mode {mode}")
        random.shuffle(self.cite_pair)

       
        
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        self.tokenizer = tokenizer
        
        self.max_length = cfg.tokenizer.max_length

    def __len__(self):
        return len(self.cite_pair)

    def __getitem__(self, index):
        # print(self.cite_pair[index])
        query_id,key_id=self.cite_pair[index]
        key_text = self.data[key_id]['abstract'].strip().lower()
        query_text=self.data[query_id]['abstract'].strip().lower()
        # query_text=key_text
        key_inputs = self.tokenizer.encode_plus(
            key_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        key_ids = key_inputs['input_ids']
        key_mask = key_inputs['attention_mask']
        key_token_type_ids = key_inputs["token_type_ids"]

        key={
            'ids': torch.tensor(key_ids, dtype=torch.long),
            'mask': torch.tensor(key_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(key_token_type_ids, dtype=torch.long),
            'text': key_text
        }

        query_inputs = self.tokenizer.encode_plus(
            query_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        query_ids = query_inputs['input_ids']
        query_mask = query_inputs['attention_mask']
        query_token_type_ids = query_inputs["token_type_ids"]

        query={
            'ids': torch.tensor(query_ids, dtype=torch.long),
            'mask': torch.tensor(query_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(query_token_type_ids, dtype=torch.long),
            'text': query_text
        }
        data={
            'key':key,
            'query':query
        }

        return data,torch.tensor(1, dtype=torch.int64)
    
class ArxivDataset(Dataset):

    def __init__(self, cfg,mode):

        # get data
        assert mode in ['train','dev']
        self.cite_pair=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_cite.pkl"))
        self.id2txt=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_txt.pkl"))
        if mode in ['dev']:
            random.shuffle(self.cite_pair)
       
        
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        self.tokenizer = tokenizer
        
        self.max_length = cfg.tokenizer.max_length

    def __len__(self):
        return len(self.cite_pair)

    def __getitem__(self, index):
        # print(self.cite_pair[index])
        query_id,key_id=self.cite_pair[index]
        key_text = self.id2txt[key_id].strip().lower()
        query_text=self.id2txt[query_id].strip().lower()
        # query_text=key_text
        key_inputs = self.tokenizer.encode_plus(
            key_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        key_ids = key_inputs['input_ids']
        key_mask = key_inputs['attention_mask']
        key_token_type_ids = key_inputs["token_type_ids"]

        key={
            'ids': torch.tensor(key_ids, dtype=torch.long),
            'mask': torch.tensor(key_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(key_token_type_ids, dtype=torch.long),
            'text': key_text
        }

        query_inputs = self.tokenizer.encode_plus(
            query_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        query_ids = query_inputs['input_ids']
        query_mask = query_inputs['attention_mask']
        query_token_type_ids = query_inputs["token_type_ids"]

        query={
            'ids': torch.tensor(query_ids, dtype=torch.long),
            'mask': torch.tensor(query_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(query_token_type_ids, dtype=torch.long),
            'text': query_text
        }
        data={
            'key':key,
            'query':query
        }

        return data,torch.tensor(1, dtype=torch.int64)


class TestDataset(Dataset):

    def __init__(self, cfg,mode):
        print(mode)
        # get data
        assert mode in ['eval_key','eval_test','eval_dev','eval_train']
        
        if mode=='eval_key':
            self.id2txt=loadpickle(cfg.dataset.test.full_data_path)
            self.id=list(self.id2txt.keys())
        elif mode in ['eval_test','eval_dev','eval_train']:
            mode=mode.split('_')[1]
            self.id2txt=loadpickle(osp.join(cfg.dataset.test.datafolder,f"{mode}_txt.pkl"))
            self.id=loadpickle(osp.join(cfg.dataset.test.datafolder,f"{mode}.pkl"))
        
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        self.tokenizer = tokenizer
        
        self.max_length =cfg.tokenizer.max_length
        # self.empt=0

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        # print(self.cite_pair[index])
        paper_id=self.id[index]
        if paper_id in self.id2txt.keys() and self.id2txt[paper_id] is not None:
            text = self.id2txt[paper_id].strip().lower()
        else:
            text=""
            # self.empt+=1
            # print("empty abstract!",f"{self.empt}")
        # query_text=key_text
        _inputs = self.tokenizer.encode_plus(
            text,
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

        return key,paper_id
    

class SSDataset(Dataset):
    # self supervised dataset for sbert
    def __init__(self, cfg,mode):

        # get data
        assert mode in ['train','dev']
        self.top200=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_top200_std.pkl"))
        #top200=[{'query':paperid,'key':[paperid1,paperid2,...,paperidn]},'recall':[r1,r2,...,rn]}]
        self.id2txt=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_top200_id2txt.pkl"))
        if mode in ['dev']:
            random.shuffle(self.top200)
       
        
        # init tokenizer
        tokenizer =cfg.tokenizer.instance
        self.tokenizer = tokenizer
        
        self.max_length = cfg.tokenizer.max_length

    def __len__(self):
        return len(self.top200)

    def __getitem__(self, index):
        # print(self.cite_pair[index])
        query_id=self.top200[index]['query']
        key_id_list=self.top200[index]['key']
        recall_list=self.top200[index]['recall']

        key_text_list=[]
        for key_id in key_id_list:
            key_text_list.append(deal_with_none_abstract(self.id2txt[key_id]).strip().lower())

        key_ids=[]
        key_mask=[]
        key_token_type_ids=[]
        for key_text in key_text_list:
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
            key_token_type_ids.append(key_inputs["token_type_ids"])

            key={
                'ids': torch.tensor(key_ids, dtype=torch.long),
                'mask': torch.tensor(key_mask, dtype=torch.long),
                'token_type_ids': torch.tensor(key_token_type_ids, dtype=torch.long),
                'text': key_text_list
            }

        query_text=deal_with_none_abstract(self.id2txt[query_id]).strip().lower()
        query_inputs = self.tokenizer.encode_plus(
            query_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        query_ids = query_inputs['input_ids']
        query_mask = query_inputs['attention_mask']
        query_token_type_ids = query_inputs["token_type_ids"]

        query={
            'ids': torch.tensor(query_ids, dtype=torch.long),
            'mask': torch.tensor(query_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(query_token_type_ids, dtype=torch.long),
            'text': query_text
        }
        data={
            'key':key,
            'query':query
        }
        target=torch.tensor(recall_list)
        # print("target->",target)
        return data,target
