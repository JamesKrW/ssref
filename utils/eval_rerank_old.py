import torch
from tqdm import tqdm
from utils.utils import tokenize,paperid2abstract



def eval(model,dev_top,id2txt,tokenizer,device,batch_size=512,RANK_SET_SIZE=64):

    # dev_top: dict genertaed from xiaofeng's top100/200/400 pkl, remember to change key name
    # model:support get_key and get_query methods, which generate key/query embedding per batch
    # get_key/query take input of ids,mask and token_type_ids (B,max_length) and output embeddings (B,H)
    # id2txt: {'paperid_1': {'abstract': str}, 'paperid_2': {'abstract': str},...,'paperid_k': {'abstract': str}} could be full_data
    
    model=model.to(device)
    query=[]
    keys_per_query=[]
    gold_ref=[]
    key=[]
    skip_query=0
    for k,v in dev_top.items():
        per_query_key=[]
        
        
        ref_freq_dict={}
        try:
            for sim_idx, sim_ref_list in v['top100']:
                for paper_id in sim_ref_list:
                    if paper_id not in ref_freq_dict:
                        ref_freq_dict[paper_id] = 1
                    else:
                        ref_freq_dict[paper_id] += 1
            sorted_ref = sorted(ref_freq_dict.items(), key=lambda x:x[1], reverse=True)
            sorted_ref_set = [k for k,v in sorted_ref[:RANK_SET_SIZE]]
            
            keys_per_query.append(list(sorted_ref_set))
            for paperid in sorted_ref_set:
                key.append(paperid)
            query.append(k)
            gold_ref.append(set(v['gt_ref']))
            
        except Exception as e:
            skip_query+=1
            
            
            
    key=list(set(key))
    print(f"total key:{len(key)}")
    print(f'skip query: {skip_query}')

    # query:[query1,query2,query3...queryk], list of paperids for query
    # gold_ref: gold ref per query, list of sets
    # keys_per_query: candidates per query, list of lists
    # key: list of paperids for key

    query_id2embedding={} # embedding for query
    i=0
    pbar = tqdm(range(0,len(query),batch_size), postfix=f"generaten query embedding")
    model.eval()
    with torch.no_grad():
        for i in pbar:
            right=i
            left=min(i+batch_size,len(query))
            sentences=[paperid2abstract(paperid,id2txt) for paperid in query[right:left]]
            _input=tokenize(sentences,tokenizer,max_length=512)
            ids=_input['ids'].to(device)
            mask=_input['mask'].to(device)
            token_type_ids=_input['token_type_ids'].to(device)
            
            output=model.get_query(ids,mask,token_type_ids) #B*L
            for idx,paperid in enumerate(query[right:left]):
                query_id2embedding[paperid]=output[idx]

    
    key_id2embedding={} # embedding for key
    i=0
    pbar = tqdm(range(0,len(key),batch_size), postfix=f"generaten key embedding")
    model.eval()
    with torch.no_grad():
        for i in pbar:
            right=i
            left=min(i+batch_size,len(key))
            sentences=[paperid2abstract(paperid,id2txt) for paperid in key[right:left]]
            _input=tokenize(sentences,tokenizer,max_length=512)
            ids=_input['ids'].to(device)
            mask=_input['mask'].to(device)
            token_type_ids=_input['token_type_ids'].to(device)
            output=model.get_query(ids,mask,token_type_ids) #B*L
            for idx,paperid in enumerate(key[right:left]):
                key_id2embedding[paperid]=output[idx]

    pbar = tqdm(range(0,len(query),1), postfix=f"calculating f1")
    
    num_overlap = 0
    num_predict = 0
    num_gt = 0
    
    for i in pbar:
        query_embedding=query_id2embedding[query[i]].unsqueeze(0)
        key_embedding=[key_id2embedding[paper] for paper in keys_per_query[i]]
        key_embedding=torch.stack(key_embedding)

        query_embedding=query_embedding.to(device)
        key_embedding=key_embedding.to(device)

        pred_logits=torch.mm(query_embedding,key_embedding.T)
        
        min_k = min(len(gold_ref[i]), len(keys_per_query[i]))
        top_k=torch.topk(pred_logits[0],k=min_k)[1]
#         print(top_k)
        pred_set={keys_per_query[i][idx.cpu().item()] for idx in top_k}

        overlap_set=pred_set.intersection(gold_ref[i])

        num_gt+=len(gold_ref[i])
        num_predict+=len(pred_set)
        num_overlap+=len(overlap_set)

        prec, rec = num_overlap / num_predict, num_overlap / num_gt
        f1 = 2 * prec * rec/(prec+rec)
        
        pbar.postfix = "Testing: prec-{:.4f} rec-{:.4f} f1-{:.4f}".format(prec, rec, f1)
        
#         break

    print('average number of gt ref:', num_gt / len(query))
    print('average number of predicted ref:', num_predict / len(query))

    prec, rec = num_overlap / num_predict, num_overlap / num_gt
    print('precision: {:.4f} recall: {:.4f} f1: {:.4f}'.format(prec, rec, 2 * prec * rec/(prec+rec)))
    