import os.path as osp
import numpy as np
import os
import random
import torch
import logging
import json
import torch.distributed as dist
import pickle



    
def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def savenp(dir,name,a):
    mkdir(dir)
    np.save(osp.join(dir,name),a)

def loadLogger(path:str,name='log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                    datefmt="%a %b %d %H:%M:%S %Y")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)


    fHandler = logging.FileHandler(path + f'/{name}.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)

    logger.addHandler(fHandler)


    return logger

def loadjson(path:str)->dict:
    with open(path,'r') as f:
        return json.load(f)

def loadpickle(path:str):
    with open(path,'rb') as f:
        return pickle.load(f)



def dict2device(_dict,device):
    for key in _dict.keys():
        if isinstance(_dict[key],dict):
             _dict[key]=dict2device( _dict[key],device)
        elif isinstance(_dict[key],torch.Tensor):
             _dict[key]=_dict[key].to(device)
    return _dict

def deal_with_none_abstract(txt):
    if txt==None:
        return " "
    else:
        return txt


def tokenize(sentences,tokenizer,max_length):
    # sentences: list of string
    # tokenizer: instance of tokenizer
    ids=[]
    mask=[]
    token_type_ids=[]
    for sent in sentences:
        _inputs = tokenizer.encode_plus(
                sent,
                None,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True,
            )
        ids.append(_inputs['input_ids'])
        mask.append(_inputs['attention_mask'])
        token_type_ids.append(_inputs["token_type_ids"])
    
    return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

def paperid2abstract(paperid,id2txt):
    if paperid in id2txt and id2txt[paperid]['abstract'] is not None:
        return id2txt[paperid]['abstract'].strip().lower()
    else:
        return ""
    

def get_token(query_text,key_text,tokenizer,max_length,query_length):
        # get query and key ids separately
        query=tokenizer(query_text,
                        max_length=max_length,
                        padding=False,
                        truncation=True,
                        )
        # delete last [SEP] and trauncated to query_length
        query_ids=query['input_ids'][:-1][:query_length]
        key=tokenizer(key_text,
                        max_length=max_length,
                        padding=False,
                        truncation=True,
                        )
        key_ids=key['input_ids']
        # replace begining [CLS] with [SEP]
        key_ids[0]=102


        input_ids=query_ids+key_ids
        #trauncated to max_length
        if len(input_ids)>max_length:
            input_ids=input_ids[:max_length]
            input_ids[-1]=102

        attention_mask=[1 for i in range(len(input_ids))]

        token_type_ids=[0 for i in range(len(query_ids)+1)]+[1 for i in range(len(key_ids)-1)]
        if len(token_type_ids)>max_length:
            token_type_ids=token_type_ids[:max_length]

        #padding
        while len(input_ids)<max_length:
            input_ids.append(0)
        while len(attention_mask)<max_length:
            attention_mask.append(0)
        while len(token_type_ids)<max_length:
            token_type_ids.append(0) 
        
        if len(input_ids)>512:
            print(len(input_ids))
            print(len(token_type_ids))
            print(len(attention_mask))
            print('maxlen->',max_length)
        return {'input_ids':input_ids,
                'token_type_ids':token_type_ids,
                'attention_mask':attention_mask,
        }