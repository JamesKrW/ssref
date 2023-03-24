import pickle
import os.path as osp




def get_cited_pairs_and_txt_id(folder_path,full_data_path):
    
    with open(full_data_path,'rb') as f:
        fulldata=pickle.load(f)

    def execute(name):
        pkl_path=osp.join(folder_path,f"{name}.pkl")
        with open(pkl_path,'rb') as f:
            paperids=pickle.load(f)
        
        id2txt={}
        cite_pair=[]
        for paperid in paperids:
            if fulldata[paperid]['abstract'] is None:
                continue
            id2txt[paperid]=fulldata[paperid]['abstract']
            for citepaper in fulldata[paperid]['references']:
                citeid=citepaper["paperId"]
                if (citeid not in fulldata.keys()) or (fulldata[citeid]['abstract'] is None):
                    continue
                cite_pair.append((paperid,citeid))
                if citeid not in id2txt.keys():
                    id2txt[citeid]=fulldata[citeid]['abstract']
        
        dump_cite_path=osp.join(folder_path,f"{name}_cite.pkl")
        dump_id2txt_path=osp.join(folder_path,f"{name}_txt.pkl")
        with open(dump_cite_path,'wb') as f:
            pickle.dump(cite_pair,f)
        with open(dump_id2txt_path,'wb') as f:
            pickle.dump(id2txt,f)

    execute('dev')
    execute('test')
    execute('train')

def get_pool(folder_path,full_data_path):
    with open(full_data_path,'rb') as f:
        fulldata=pickle.load(f)
        id2txt={}
        for k,v in fulldata.items():
            id2txt[k]=v['abstract']
        full_data_path=osp.join(folder_path,f"full_data_txt.pkl")
        with open(full_data_path,'wb') as f:
            pickle.dump(id2txt,f)
    
def prun_no_embed(folder_path,full_data_path):
    with open(full_data_path,'rb') as f:
        fulldata=pickle.load(f)
        id2txt={}
        for k,v in fulldata.items():
            del v["embedding"]
            id2txt[k]=v
        full_data_path=osp.join(folder_path,f"full_data_no_embed.pkl")
        with open(full_data_path,'wb') as f:
            pickle.dump(id2txt,f)
    
    
# get_cited_pairs_and_txt_id('/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml-debug',
#                            "/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml-debug/full_data.pkl")
# get_pool('/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml',
#          "/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml/full_data_minus.pkl")
prun_no_embed('/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml',
              "/share/data/mei-work/kangrui/github/ref-sum/refsum/data/refsum-data/arxiv-aiml/full_data.pkl")

    

    
    
        
