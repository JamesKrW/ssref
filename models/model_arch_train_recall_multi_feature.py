from transformers import BertModel
from torch import nn
import torch
from utils.utils import*
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer, AutoConfig
import torch.nn.functional as F




        
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
    
    def forward(self, input_ids,attention_mask,token_type_ids=None):
        model_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
class pair_sbert_freeze_key(nn.Module):
    #self supervised learning sbert
    def __init__(self, cfg):
        super().__init__()
        self.key_encoder = SentenceEncoder(cfg.model_arch.name,checkpoint_enable=False)
        self.query_encoder = SentenceEncoder(cfg.model_arch.name,checkpoint_enable=False)
        for param in self.key_encoder.parameters():
                param.requires_grad = False
       

    def forward(self, input,mode=None):

        
        if mode!=None:
            s=input['ids']
            ms=input['mask']
            tks=input['token_type_ids']
            if mode=='query':
                 return self.query_encoder(s,ms,tks)
            else:
                 return self.key_encoder(s,ms,tks)
        else:
            ss=input['query']['ids']
            sms=input['query']['mask']
            stks=input['query']['token_type_ids']
            
            

            ts=input['key']['ids']
            tms=input['key']['mask']
            ttks=input['key']['token_type_ids']

            query=self.query_encoder(ss,sms,stks)
            key=self.key_encoder(ts,tms,ttks)
            return  {'key':key,'query':query}
        
class single_sbert(nn.Module):
    #self supervised learning sbert
    def __init__(self, cfg):
        super().__init__()
        self.query_encoder = SentenceEncoder(cfg.model_arch.name,checkpoint_enable=False)
       

    def forward(self, input,mode=None):

        
        if mode!=None:
            s=input['ids']
            ms=input['mask']
            tks=input['token_type_ids']
            if mode=='query':
                 return self.query_encoder(s,ms,tks)
            else:
                 return self.query_encoder(s,ms,tks)
        else:
            ss=input['query']['ids']
            sms=input['query']['mask']
            stks=input['query']['token_type_ids']
            
            

            ts=input['key']['ids']
            tms=input['key']['mask']
            ttks=input['key']['token_type_ids']

            query=self.query_encoder(ss,sms,stks)
            key=self.query_encoder(ts,tms,ttks)

            return  {'key':key,'query':query}
        
class pair_sbert(nn.Module):
    #self supervised learning sbert
    def __init__(self, cfg):
        super().__init__()
        self.key_encoder = SentenceEncoder(cfg.model_arch.name,checkpoint_enable=False)
        self.query_encoder = SentenceEncoder(cfg.model_arch.name,checkpoint_enable=False)
       
       

    def forward(self, input,mode=None):

        
        if mode!=None:
            s=input['ids']
            ms=input['mask']
            tks=input['token_type_ids']
            if mode=='query':
                 return self.query_encoder(s,ms,tks)
            else:
                 return self.key_encoder(s,ms,tks)
        else:
            ss=input['query']['ids']
            sms=input['query']['mask']
            stks=input['query']['token_type_ids']
            
            

            ts=input['key']['ids']
            tms=input['key']['mask']
            ttks=input['key']['token_type_ids']

            query=self.query_encoder(ss,sms,stks)
            key=self.key_encoder(ts,tms,ttks)
        

            return  {'key':key,'query':query}
        
class pair_sbert_freeze_query(nn.Module):
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
                 return self.query_encoder(s,ms,tks)
            else:
                 return self.key_encoder(s,ms,tks)
        else:
            ss=input['query']['ids']
            sms=input['query']['mask']
            stks=input['query']['token_type_ids']
            
            

            ts=input['key']['ids']
            tms=input['key']['mask']
            ttks=input['key']['token_type_ids']

            query=self.query_encoder(ss,sms,stks)
            key=self.key_encoder(ts,tms,ttks)
        

            return  {'key':key,'query':query}