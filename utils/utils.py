import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml



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


def init_obj(obj_dict:dict, module, *args, **kwargs):
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
    return config


'''
def config_dir(config_file_path:str) -> dict:
    with open(config_file_path, 'rt') as f:
        config = yaml.safe_load(f)

    if config.get('save', False):
        save_root_dir = Path(config['save_dir'])
        exper_name = config['name']
        run_id = config.get('run_id', datetime.now().strftime(r'%m%d_%H%M%S')) 

        checkpoint_dir = save_root_dir / exper_name / run_id / 'checkpoints'
        log_dir = save_root_dir / exper_name / run_id / 'log'
        
        # make directory for saving checkpoints and log.
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir(parents=True, exist_ok=False)
        # copy config file to checkpoint dir
        with (log_dir / 'config.yaml').open('wb') as f:
            f.write(Path(config_file_path).read_bytes())
        
        # update logging
        loggingConfigDict = config['logger']
        for _, handler in loggingConfigDict['handlers'].items():
            if 'filename' in handler.keys():
                handler['filename'] = str(log_dir / handler['filename'])
        
        
    else:
        checkpoint_dir = ""
        log_dir = ""
        loggingConfigDict = config['logger']
        for key, handler in list(loggingConfigDict['handlers'].items()):
            if 'filename' in handler.keys():
                loggingConfigDict['handlers'].pop(key)
                loggingConfigDict['root']['handlers'].remove(key)
    # update config_dict after write it
    config['checkpoint_dir'] = checkpoint_dir
    config['log_dir'] = log_dir
    config['save'] = config.get('save', False)
    logging.config.dictConfig(loggingConfigDict)

    
    return config
'''


def my_mse(y_true, y_pred):
    mse_value = np.average((y_true - y_pred) ** 2, axis=0)
    return mse_value


def my_ssim(im1, im2):
    K1 = 0.01
    K2 = 0.03
    n = len(im1)    

    im1_mean = im1.mean()
    im2_mean = im2.mean()
    cov = ((im1 - im1_mean) * (im2 - im2_mean)).sum() / (n - 1)
    im1_var = ((im1 - im1_mean) * (im1 - im1_mean)).sum() / (n - 1)
    im2_var = ((im2 - im2_mean) * (im2 - im2_mean)).sum() / (n - 1)

    c1 = K1 ** 2
    c2 = K2 ** 2

    a1 = 2 * im1_mean * im2_mean + c1
    a2 = 2 * cov + c2
    b1 = im1_mean ** 2 + im2_mean ** 2 + c1
    b2 = im1_var + im2_var + c2

    ssim_value = (a1 * a2) / (b1 * b2)

    return ssim_value



def generate_Xs(nA=2, nB=2, AT_optionals=np.array([0.1, 1, 10])):
    dim = len(AT_optionals)
    ATs = np.array(list(itertools.product(AT_optionals, AT_optionals)))
    BTs = np.ones((dim*dim, nB))
    Xs = np.concatenate([ATs, BTs], axis=1)
    return Xs


def generate_Ys_list(dim=3, no_repeat=True):
    Ys_list = np.array(list(itertools.product([0, 1], repeat=dim*dim)))
    no_repeat_Ys_list = []
    if (no_repeat == False):
        return Ys_list
    else:
        for Ys in Ys_list:
            Ys_T = Ys.reshape(dim, dim).T.reshape(-1)
            flag = False
            for no_repeat_Ys in no_repeat_Ys_list:
                if ((no_repeat_Ys == Ys_T).all()):
                    flag = True
                    break
            if (flag == False):
                no_repeat_Ys_list.append(Ys)
    return no_repeat_Ys_list


if __name__ == '__main__':
    Xs = generate_Xs()
    Ys_list = generate_Ys_list()
    print(Xs)
    print(len(Ys_list))