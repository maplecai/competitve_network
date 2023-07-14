import datetime
import gzip
import itertools
import os
import random
from pathlib import Path
from struct import unpack
import logging
import logging.config
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from copy import deepcopy

np.set_printoptions(precision=8, suppress=True)
torch.set_printoptions(precision=8, sci_mode=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_obj(module, obj_dict:dict, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.

    `object = init_obj(obj_dict, module, a, b=1)`
    is equivalent to
    `object = module.obj_dict['type'](a, b=1)`
    """
    assert isinstance(obj_dict, dict), "invalid init object dict"
    
    module_name = obj_dict['type']
    module_args = dict(obj_dict.get('args', {}))
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    
    return getattr(module, module_name)(*args, **module_args)



def parse_config(config_file_path: str) -> dict:
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    # change something manually
    now_time = datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')
    config['log_dir'] = os.path.join(config['log_dir'], config['task_name'], now_time)
    config['results_dir'] = os.path.join(config['results_dir'], config['task_name'], now_time)
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])
    if not os.path.exists(config['results_dir']):
        os.makedirs(config['results_dir'])
    return config


# def process_config(config: dict) -> dict:

#     now_time = datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')
#     task_name = config['task_name']
#     config['log_dir'] = os.path.join(config['log_dir'], now_time)
#     config['results_dir'] = os.path.join(config['results_dir'], now_time)

#     log_dir = os.path.join(task_name, now_time, log_dir)
#     results_dir = os.path.join(task_name, now_time, results_dir)
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)

#     now_time = datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S_%f')
#     model_args = '_'.join([str(v) for v in config['model']['args'].values()])
#     file_name = now_time+'_'+task_name+'_'+model_args
    
#     config['log_dir'] = log_dir
#     config['results_dir'] = results_dir
#     config['file_name'] = file_name
#     config['logger']['handlers']['file_handler']['filename'] = os.path.join(log_dir, file_name)+'.log'

#     return config



# def load_data(data_path: str) -> dict:
#     train_x = np.load(f'{data_path}/train_x.npy')
#     train_y = np.load(f'{data_path}/train_y.npy')



# def config_dir(config_file_path:str) -> dict:
#     with open(config_file_path, 'rt') as f:
#         config = yaml.safe_load(f)

#     if config.get('save', False):
#         save_root_dir = Path(config['save_dir'])
#         exper_name = config['name']
#         run_id = config.get('run_id', datetime.now().strftime(r'%m%d_%H%M%S')) 

#         checkpoint_dir = save_root_dir / exper_name / run_id / 'checkpoints'
#         log_dir = save_root_dir / exper_name / run_id / 'log'
        
#         # make directory for saving checkpoints and log.
#         checkpoint_dir.mkdir(parents=True, exist_ok=False)
#         log_dir.mkdir(parents=True, exist_ok=False)
#         # copy config file to checkpoint dir
#         with (log_dir / 'config.yaml').open('wb') as f:
#             f.write(Path(config_file_path).read_bytes())
        
#         # update logging
#         loggingConfigDict = config['logger']
#         for _, handler in loggingConfigDict['handlers'].items():
#             if 'filename' in handler.keys():
#                 handler['filename'] = str(log_dir / handler['filename'])
        
        
#     else:
#         checkpoint_dir = ""
#         log_dir = ""
#         loggingConfigDict = config['logger']
#         for key, handler in list(loggingConfigDict['handlers'].items()):
#             if 'filename' in handler.keys():
#                 loggingConfigDict['handlers'].pop(key)
#                 loggingConfigDict['root']['handlers'].remove(key)
#     # update config_dict after write it
#     config['checkpoint_dir'] = checkpoint_dir
#     config['log_dir'] = log_dir
#     config['save'] = config.get('save', False)
#     logging.config.dictConfig(loggingConfigDict)

    
#     return config




def read_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28)
    return img.copy()

def read_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.frombuffer(f.read(), dtype=np.uint8)
    return lab.copy()



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.flag = False
        self.delta = delta
        self.trace_func = trace_func
        self.best_valid_loss = np.Inf
        self.best_model_state_dict = None

    def check(self, valid_loss, model_state_dict):

        if (valid_loss < self.best_valid_loss - self.delta):
            if self.verbose:
                self.trace_func(f'valid loss decreased {self.best_valid_loss:.6f} --> {valid_loss:.6f}. saving model')
            self.counter = 0
            self.best_valid_loss = valid_loss
            self.best_model_state_dict = deepcopy(model_state_dict)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.flag = True
            if self.verbose:
                self.trace_func(f'valid loss not decreased, counter: {self.counter}')



# class TopKAcc(nn.Module):
#     def __init__(self, k=3):
#         super().__init__()
#         self.k = k

#     def forward(self, output:torch.Tensor, label:torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         pred = torch.topk(output, self.k, dim=1)[1]
#         assert pred.shape[0] == len(label)
#         correct = 0
#         for i in range(self.k):
#             correct += torch.sum(pred[:, i] == label)
#         return correct / len(label)



# class Accuracy(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, output:torch.Tensor, label:torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(label)
#         correct = 0
#         correct += torch.sum(pred == label)
#         return correct / len(label)



if __name__ == '__main__':
    pass
