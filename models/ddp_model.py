import os.path as osp
from collections import OrderedDict
import torch
import torch.nn
from utils.utils import *
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP


class Model:
    def __init__(self,cfg,net_arch,loss_f=None,optimizer=None,scheduler=None,rank=0):
        self.cfg = cfg
        self.device = self.cfg.device
        self.net = net_arch.to(self.device)
        
        self.rank = rank
        if self.device != "cpu" and self.cfg.dist.gpus != 0:
            self.net = DDP(self.net, device_ids=[self.rank],find_unused_parameters=True)
        self.step = 0
        self.epoch = -1
        if is_logging_process():
            self._logger = cfg.logger
            self._logger.info(cfg.get_info())

        if optimizer is not None:
            self.optimizer=optimizer(self.net.parameters(),**self.cfg.optimizer.optimizer_cfg)
        else:
            self.optimizer=None
        if optimizer is not None:
            self.scheduler=scheduler(name=self.cfg.scheduler.name, optimizer=self.optimizer, 
                                    num_warmup_steps=self.cfg.scheduler.num_warmup_steps, 
                                    num_training_steps=self.cfg.scheduler.num_training_steps)
        else:
            self.scheduler=None

        # init loss
        self.loss_f = loss_f
        self.loss_v = 0

        self.best_test_rst=None

    def optimize_parameters(self, model_input, model_target):
        assert self.optimizer is not None
        assert self.loss_f is not None
        self.net.train()
        self.optimizer.zero_grad()
        output = self.run_network(model_input)
        loss_v = self.loss_f(output, model_target.to(self.device))
        loss_v.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.loss_v = loss_v.item()

    def inference(self, model_input):
        self.net.eval()
        output = self.run_network(model_input)
        return output

    def run_network(self, model_input):
        # may need refactoring for different input type
        model_input = dict2device(model_input,self.device)
        output = self.net(model_input)
        return output
    
    def get_key(self, input):
        # may need refactoring for different input type
        with torch.no_grad():
            self.net.eval()
            input = dict2device(input,self.device)
            output = self.net(input,'key')
        return output
    
    def get_query(self, input):
        # may need refactoring for different input type
        with torch.no_grad():
            self.net.eval()
            input = dict2device(input,self.device)
            output = self.net(input,'query')
        return output

    def save_network(self, save_file=True,name=None):
        if is_logging_process():
            net = self.net.module if isinstance(self.net, DDP) else self.net
            state_dict = net.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.to("cpu")
            if save_file:
                save_filename = "%s_%d.pt" % (self.cfg.model_name, self.step)
                if name is not None:
                    save_filename=f"{name}.pt"
                save_path = osp.join(self.cfg.chkpt_dir, save_filename)
                torch.save(state_dict, save_path)
                self._logger.info("Saved network checkpoint to: %s" % save_path)
            return state_dict

    def load_network(self, loaded_net=None,name=None):
        add_log = False
        if loaded_net is None:
            add_log = True
            loaded_net = torch.load(
                self.cfg.model.network_pth_path,
                map_location=torch.device(self.device),
            )
        loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
        for k, v in loaded_net.items():
            if k.startswith("module."):
                loaded_clean_net[k[7:]] = v
            else:
                loaded_clean_net[k] = v

        self.net.load_state_dict(loaded_clean_net, strict=self.cfg.model.strict_load)
        if is_logging_process() and add_log:
            self._logger.info(
                "Checkpoint %s is loaded" % self.cfg.model.network_pth_path
            )

    def save_training_state(self,name=None):
        if is_logging_process():
            save_filename = "%s_%d.state" % (self.cfg.model_name, self.step)
            if name is not None:
                save_filename=f"{name}.state"
            save_path = osp.join(self.cfg.chkpt_dir, save_filename)
            net_state_dict = self.save_network(False)
            state = {
                "model": net_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
            }
            torch.save(state, save_path)
            self._logger.info("Saved training state to: %s" % save_path)

    def load_training_state(self):
        resume_state = torch.load(
            self.cfg.model.resume_state_path,
            map_location=torch.device(self.device),
        )

        self.load_network(loaded_net=resume_state["model"])
        self.optimizer.load_state_dict(resume_state["optimizer"])
        self.step = resume_state["step"]
        self.epoch = resume_state["epoch"]
        if is_logging_process():
            self._logger.info(
                "Resuming from training state: %s" % self.cfg.model.resume_state_path
            )