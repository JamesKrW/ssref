from transformers import BertModel
from torch import nn
import torch
from utils.utils import*
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer, AutoConfig
import torch.nn.functional as F



class BertSiameseClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        plm_config = AutoConfig.from_pretrained(cfg.model_arch.name)
       
        self.bert = AutoModel.from_pretrained(cfg.model_arch.name)

            
        self.bert.gradient_checkpointing_enable()
        self.attention_fc = nn.Linear(plm_config.hidden_size, 1, bias=False)
        self.loss = nn.CrossEntropyLoss()
        
    
    def forward(self, input,mode=None):

        LARGE_NEG = -1e9

        if mode!=None: # eval mode
            ss=input['ids']
            sms=input['mask']
            


            s_hiddens = self.bert(ss, attention_mask=sms)
            s_hiddens = s_hiddens[0] 
            s_hiddens=torch.tanh(s_hiddens)

            s_att_logits = self.attention_fc(s_hiddens).squeeze(-1) # (B, L) 
            s_att_logits = s_att_logits + (1. - sms)*LARGE_NEG # (B, L)
            s_att = F.softmax(s_att_logits, dim=-1) # (B, L)
            s_hiddens = torch.sum(s_hiddens * s_att.unsqueeze(-1), dim=1) # (B, H)

            return s_hiddens
        
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
    
    def forward(self, input_ids,attention_mask):
        model_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
class pair_sbert(nn.Module):
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
            if mode=='query':
                 return self.query_encoder(s,ms)
            else:
                 return self.key_encoder(s,ms)
        else:
            ss=input['query']['ids']
            sms=input['query']['mask']
            
            

            ts=input['key']['ids']
            tms=input['key']['mask']
            ts=ts.view(-1,ts.shape[2])
            tms=tms.view(-1,tms.shape[2])
            query=self.query_encoder(ss,sms)
            key=self.key_encoder(ts,tms)
            # n=ss.shape[0]
            # s=torch.cat((ss,ts),dim=0)
            # ms=torch.cat((sms,tms),dim=0)


            # embedding=self.query_encoder(s,ms)
    
            # key=embedding[n:]
            # query=embedding[:n]

            return  {'key':key,'query':query}