import torch
from torch.utils.data import Dataset
from utils.utils import *
import pickle
from transformers import AutoTokenizer
import random
    
class ArxivDataset_raw(Dataset):

    def __init__(self, cfg,mode):

        # get data
        assert mode in ['train','dev']
        self.cite_pair=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_top1000_cite_pair.pkl"))
        self.id2txt=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_top1000_cite2txt_abs_title_author.pkl"))
        if mode in ['dev']:
            random.shuffle(self.cite_pair)
       
        
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        self.tokenizer = tokenizer
        
        self.max_length = cfg.tokenizer.max_length

    def __len__(self):
        return len(self.cite_pair)

    def txt_generate(self,data):
        para=""
        para+=data['title']+'[SEP]'
        authortxt=' and '.join([item['name'] for item in data['authors']])
        para+=authortxt+'[SEP]'
        para+=data['abstract']
        return para

    def __getitem__(self, index):
        # print(self.cite_pair[index])
        query_id=self.cite_pair[index]['query']
        key_id=self.cite_pair[index]['key']
        target=self.cite_pair[index]['value']
        key_text = self.txt_generate(self.id2txt[key_id])
        query_text=self.txt_generate(self.id2txt[query_id])
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
        return data,torch.tensor(target, dtype=torch.float32)   

class ArxivDataset(Dataset):

    def __init__(self, cfg,mode):

        # get data
        assert mode in ['train','dev']
        self.cite_pair=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_top1000_cite_pair.pkl"))
        self.id2txt=loadpickle(osp.join(cfg.dataset.datafolder,f"{mode}_top1000_cite2txt_abs_title_author.pkl"))
        if mode in ['dev']:
            random.shuffle(self.cite_pair)
       
        
        # init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        self.tokenizer = tokenizer
        
        self.max_length = cfg.tokenizer.max_length

    def __len__(self):
        return len(self.cite_pair)

    def txt_generate(self,data):
        para=""
        para+=data['title']+'[SEP]'
        authortxt=' and '.join([item['name'] for item in data['authors']])
        para+=authortxt+'[SEP]'
        para+=data['abstract']
        return para

    def __getitem__(self, index):
        # print(self.cite_pair[index])
        query_id=self.cite_pair[index]['query']
        key_id=self.cite_pair[index]['key']
        target=self.cite_pair[index]['value']
        key_text = self.txt_generate(self.id2txt[key_id])
        query_text=self.txt_generate(self.id2txt[query_id])
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
        if target>1:
            target=1.
        elif target==1:
            target=0.6
        else:
            target=-1.
        return data,torch.tensor(target, dtype=torch.float32)


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
            self.id2txt=loadpickle(osp.join(cfg.dataset.test.datafolder,f"{mode}_filtered_abs_title_author_data.pkl"))
            self.id=loadpickle(osp.join(cfg.dataset.test.datafolder,f"{mode}_filtered.pkl"))
        
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
        data=self.id2txt[paper_id]
        sent1=' and '.join([item['name'] for item in data['authors']])
        sent2=data['title']+'.'+'\n'+data['abstract']
        
        _inputs = self.tokenizer.encode_plus(
            sent1,
            sent2,
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
    


