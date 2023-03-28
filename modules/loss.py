import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import sys
import math
class NTXent(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.device=cfg.device
        self.tau=torch.tensor([[cfg.loss.tau]],requires_grad=False).to(self.device)
        self.entropy=nn.CrossEntropyLoss()

    def forward(self, output,target):
        query=output['query']
        key=output['key']
        query=query/torch.sqrt(self.tau)
        key=key/torch.sqrt(self.tau)
        n=query.shape[0]
    
        pred_logits = torch.mm(query, key.transpose(0, 1))
        ys = torch.tensor(list(range(n))).to(self.device)
        loss2=self.entropy(pred_logits,ys)

        return loss2
    
class SSloss(nn.Module):
    # for sbert's self-supervised learning
    def __init__(self,cfg):
        super().__init__()
        

    def forward(self, output,target):
        target=target.view(-1)
        query=output['query']
        key=output['key']
        n=query.shape[0]
        loss=0.
        for i in range(n):
            pred_logit = torch.mm(query[i:i+1], key[i*200:(i+1)*200].transpose(0, 1))[0]
            # print("pred_logits->",pred_logit)
            log_probs = F.log_softmax(pred_logit, dim=0)
            # print("log_probs->",log_probs)
            # print("target->",target[i*200:(i+1)*200])
            x=-torch.mean(log_probs*target[i*200:(i+1)*200])
            # print("weighted_mean:",x)
            loss+=x
        loss/=n
        # print("loss:",loss)
        return loss
    
class SSloss_multifeature(nn.Module):
    # for sbert's self-supervised learning
    def __init__(self,cfg):
        super().__init__()
        
        self.MSE=torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    def forward(self, output,target):
        target=target.view(-1)
        query=output['query'] #b*v
        key=output['key'] #b*v
        rst=torch.einsum('ij,ij->i', query, key) #b*b
        # log_rst=torch.log(rst+1e-6) #b
        loss=self.MSE(rst,target)
        # loss_v=loss.item()
        # if (loss_v > 1e8 or math.isnan(loss_v)):
        #     print(target)
        #     print(rst)
        #     print(log_rst)
        return loss
        
