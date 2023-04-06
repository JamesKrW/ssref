import argparse
import torch
from utils.utils import *
import time
# from torch.utils.tensorboard import SummaryWriter
class Config:
    def __init__(self):
        pass
    
    def get_info(self):
        _dict={}
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],Config):
                _dict[key]=self.__dict__[key].get_info()
            else:
                _dict[key]=self.__dict__[key]
        return _dict
    
    def init(self):
        parser = argparse.ArgumentParser(description='training template')
        parser.add_argument('--train_batch_size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 128)')
        parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                            help='input batch size for testing (default: 128)')
        parser.add_argument('--num_epoch', type=int, default=30, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=1e-5, metavar='N',
                            help='learning rate')
        parser.add_argument('--seed', type=int, default=2022, metavar='N',
                            help='random seed')
        parser.add_argument('--work_dir', type=str, default='./result', metavar='N',
                            help='working directory')
        parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', metavar='N',
                            help='model name for training')
        parser.add_argument('--multi_gpu', type=int, default=1, metavar='N',
                            help='use multi gpu to train')
        parser.add_argument('--log_interval_test', type=int, default=1000, metavar='N',
                            help='test interval vs step')
        parser.add_argument('--log_interval_train', type=int, default=500, metavar='N',
                            help='train interval vs step')
        parser.add_argument('--pt_path', type=str, 
                            help='model pt path for eval')
        parser.add_argument('--state_path', type=str, 
                            help='training state path for resume ')
        args = parser.parse_args()

        self.num_epoch=args.num_epoch
        self.seed=args.seed

        self.multi_gpu=args.multi_gpu
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name=args.model_name.replace("/", "_")
        self.work_dir = osp.join(osp.join(args.work_dir,args.model_name.replace("/", "_")),time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime()))
        self.chkpt_dir=osp.join(self.work_dir,"checkpoints")
        self.chkpt_interval=3 #epoch
        # training step
        self.log_interval_train=args.log_interval_train
        self.log_interval_test=args.log_interval_test
        self.log_interval_recall=500


        self.dataset=Config()
        self.dataset.datafolder='/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml-small'


        self.dataloader=Config()
        self.dataloader.train_batch_size=args.train_batch_size
        self.dataloader.test_batch_size=args.test_batch_size
        self.dataloader.train_num_workers=2
        self.dataloader.test_num_workers=2
        self.dataloader.divide_dataset_per_gpu=True
        self.dataloader.use_background_generator = True


        self.optimizer=Config()
        self.optimizer.lr=args.lr
        self.optimizer.weight_decay=0.01
        self.optimizer.betas=(0.9, 0.99)
        self.optimizer.optimizer_cfg={'lr':self.optimizer.lr,'weight_decay':self.optimizer.weight_decay,'betas':self.optimizer.betas}

        self.scheduler=Config()
        self.scheduler.name="linear"
        self.scheduler.num_warmup_steps=100
        self.scheduler.num_training_steps=None #caculated in training

        self.tokenizer=Config()
        self.tokenizer.name=args.model_name
        self.tokenizer.max_length=512

        self.model=Config()
        self.model.resume_state_path=args.state_path
        self.model.network_pth_path=args.pt_path
        self.model.strict_load=True


        self.model_arch=Config()
        self.model_arch.name=args.model_name
        self.model_arch.maxlen=512
        self.model_arch.dropout=0.3
        self.model_arch.outputdim=768
        self.model_arch.rawoutput=True


        self.loss=Config()
        self.loss.tau=1.

        self.dist=Config()
        self.dist.gpus=0
        self.dist.master_addr='127.0.0.1'
        self.dist.master_port="1234"
        self.dist.backend='nccl'
        self.dist.init_method='env://'
        self.dist.world_size=None
        self.dist.timeout=3600


        self.sstraining=Config()
        self.sstraining.id2paper="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml/full_data_no_embed.pkl"
        self.sstraining.key_id2embed="/share/data/mei-work/kangrui/github/ssref/result/pretrained_sbert/test_result/eval_key_id2embed.pkl"
        self.sstraining.query="/share/data/mei-work/kangrui/github/ssref/data/refsum-data/arxiv-aiml-small/dev_filtered.pkl"
    
    

