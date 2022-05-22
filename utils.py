import numpy as np
import random 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def my_ssim(im1, im2, **kwargs):
    im1=im1.reshape(-1)
    im2=im2.reshape(-1)
    if im1.shape != im2.shape:
        raise ValueError('im1 and im2 must have same dimension!    \
                         im1.shape=%s but im2.shape=%s'%(im1.shape, im2.shape))

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")

    n = im1.shape[0]
    
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

    ssim_value = a1 * a2 / (b1 * b2)
    return ssim_value


if __name__ == '__main__':
    print('test')