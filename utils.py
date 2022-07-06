import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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



def generate_patterns(nA=2, nB=2, AT_optionals=np.array([0.1, 1, 10])):
    dim = len(AT_optionals)
    ATs = np.array(list(itertools.product(AT_optionals, AT_optionals)))
    BTs = np.ones((dim*dim, nB))
    Xs = np.concatenate([ATs, BTs], axis=1)

    Ys_list = np.array(list(itertools.product([0, 1], repeat=dim*dim)))
    no_repeat_Ys_list = []

    for Ys in Ys_list:
        Ys_T = Ys.reshape(dim, dim).T.reshape(-1)
        repeat = False
        for no_repeat_Ys in no_repeat_Ys_list:
            if ((no_repeat_Ys == Ys_T).all()):
                repeat = True
                break
        if (repeat == False):
            no_repeat_Ys_list.append(Ys)

    print(len(no_repeat_Ys_list))
    return Xs, no_repeat_Ys_list

if __name__ == '__main__':
    generate_patterns()
    