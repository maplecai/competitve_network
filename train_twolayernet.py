# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:53:16 2021

@author: zxr
"""
import numpy as np
import os
import random 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['Microsoft YaHei']



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
    
    
    #正常
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



class _mask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat_input):
        return F.relu(mat_input) 
    @staticmethod   
    def backward(ctx, grad_output):
        return grad_output
mask=_mask.apply


class TwoLayerNet(torch.nn.Module):
    def __init__(self, L_dim, A_dim, y_dim, max_step):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.max_step = max_step
        self.K = nn.Parameter(torch.Tensor(L_dim, A_dim))
        self.AT = nn.Parameter(torch.Tensor(A_dim))
        self.u = nn.Parameter(torch.Tensor(L_dim * A_dim, y_dim))
        self.e = nn.Parameter(torch.Tensor(A_dim, y_dim))
        self.b = nn.Parameter(torch.Tensor(y_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        self.K = nn.Parameter( (torch.rand(self.K.shape) + 0/3) *1.0 )
        self.AT = nn.Parameter( (torch.rand(self.AT.shape) + 0/3) *1.0 )
        self.u = nn.Parameter( (torch.rand(self.u.shape) + 0/3) *1.0 )
        self.e = nn.Parameter( (torch.rand(self.e.shape) + 0/3) *1.0 )
        self.b = nn.Parameter( (torch.rand(self.b.shape) + 0/3) *1.0 )

            
    def forward(self, LT):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        K = mask(self.K) 
        AT = mask(self.AT) 
        u = (self.u)
        e = (self.e)
        b = (self.b)

        AF = torch.zeros([LT.shape[0], AT.shape[0], 1]).to(device)
        LF = torch.zeros([LT.shape[0], LT.shape[1], 1]).to(device)
        # print(AF.shape, LF.shape)
        
        with torch.no_grad():
            err = 0
            for step in range(self.max_step):

                AF_last = AF
                LF_last = LF

                Ldi = torch.unsqueeze(LT, dim=2) / (K.matmul(AF)+1)
                Adi = torch.unsqueeze(AT, dim=1) / (K.T.matmul(LF)+1)
                AF = torch.unsqueeze(AT, dim=1) / (K.T.matmul(Ldi)+1)
                LF = torch.unsqueeze(LT, dim=2) / (K.matmul(Adi)+1)
                
                err = ((AF-AF_last).abs()/(AF_last+1e-5)).max() + ((LF-LF_last).abs()/(LF_last+1e-5)).max()
                if (err < tol):
                    break

            if (err < tol):
                pass
                # print('converge in', step, 'iterations')
            else:
                # print('not converge in', step, 'iterations')
                self.max_step *= 2


        Ldi = torch.unsqueeze(LT, dim=2) / (K.matmul(AF)+1)
        Adi = torch.unsqueeze(AT, dim=1) / (K.T.matmul(LF)+1)
        AF = torch.unsqueeze(AT, dim=1) / (K.T.matmul(Ldi)+1)
        LF = torch.unsqueeze(LT, dim=2) / (K.matmul(Adi)+1)

        # print(Ldi.shape, AF.shape, Adi.shape, LF.shape)
        # print(D.shape, K.shape, u.shape, e.shape, b.shape)

        AF = AF.transpose(1, 2)
        D = (K * (LF.matmul(AF))).view([LT.shape[0],-1])
        AF = torch.squeeze(AF, dim=1)
        y = D.matmul(u) + AF.matmul(e) + torch.unsqueeze(b, dim=0)
        return y 



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

MAX_EPOCH = 10000
MAX_STEP = 10000
BATCH_SIZE = 4
tol = 1e-3
#lr = 1e-4
loss_func = F.cross_entropy

if __name__ == '__main__':
    print('main')

    loss_list = []

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    x_train = torch.tensor(x, dtype=torch.float32).to(device)
    y_train = torch.tensor(y, dtype=torch.float32).to(device)

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TwoLayerNet(L_dim=2, A_dim=2, y_dim=1, max_step=MAX_STEP).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

    for epoch in range(MAX_EPOCH):
        for x, y in train_dl:
            # print(x.shape)
            out = model(x)
            loss = 1 - my_ssim(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        if (epoch % 1000 == 0):
            print('epoch', epoch, 'loss', loss.item())
            # print(model.K.cpu().detach().numpy())
            print(out)
            print(y)
            print(my_ssim(out, y))

        if (loss.item() < 1e-5):
            break

    plt.figure(figsize=(8,6), dpi=100)
    plt.plot(loss_list)
    plt.show()


