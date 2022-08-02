# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:53:16 2021

@author: zxr
"""
ty='CP'
import pandas as pd
quarter = 3
grids,start,end = 2,0,4
NUM=0
import numpy as np
import os
import random 
from struct import unpack
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
import copy
from sklearn.model_selection import StratifiedShuffleSplit
import time
from torch.nn import init
import math
from sklearn import svm
import swats
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

beta = 0.0
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# os.chdir(r'D:\学习\研究生\教研室\WTA\竞争\普通竞争（BMP）\bio')
seed=1#int(np.random.rand(1)*100)
print(seed)
set_seed(seed)

    
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
    


def predict_acc(x,y):
    with torch.no_grad():
        valid_acc = my_ssim(model(x), y)
    return valid_acc.item()

class _mask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat_input):
        return F.relu(mat_input) 
    @staticmethod   
    def backward(ctx, grad_output):
        return grad_output
mask=_mask.apply

def make_pic(grids=3,start=0,end=4):
    x=np.zeros([grids*grids,2])
    for i,s in enumerate(np.logspace(start,end,grids)):
        #print(s)
        x[range(i,grids*grids,grids),0] = s
        x[range(i*grids,i*grids+grids),1] = s    
    return torch.tensor(x,dtype=torch.float32).to(dev)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out,MAX_STEP = 10000,bias = True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.max_step = MAX_STEP
        self.K = nn.Parameter(torch.Tensor(D_in, H))
        self.AT = nn.Parameter(torch.Tensor(H))
        self.u = nn.Parameter(torch.Tensor(D_in *H, D_out))
        self.e = nn.Parameter(torch.Tensor(H, D_out))
        self.b = nn.Parameter(torch.Tensor(D_out))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
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
        AF=torch.zeros([LT.shape[0],self.AT.shape[0],1]).to(dev)
        LF=torch.zeros([LT.shape[0],LT.shape[1],1]).to(dev)
        K = mask(self.K) 
        AT = mask(self.AT) 
        u = (self.u) 
        e = (self.e)
        b = (self.b) 
        
        for step in range(self.max_step):
            LF_last = AF.detach() +0.0
            Ldi = torch.unsqueeze(LT, dim=2) / (K.detach().matmul(AF)+1)
            AF = torch.unsqueeze(AT.detach(), dim=1) / (K.detach().T.matmul(Ldi)+1)  
            
            AF_last = LF.detach() +0.0
            Adi = torch.unsqueeze(AT.detach(), dim=1) / (K.detach().T.matmul(LF)+1)
            LF = torch.unsqueeze(LT, dim=2) / (K.detach().matmul(Adi)+1)
            
            if ((AF.detach() - LF_last).abs()/(LF_last+1e-5) ).max() + ((LF.detach() - AF_last).abs()/(AF_last+1e-5) ).max()< 1e-3:
                break
        if step==self.max_step - 1:
            self.max_step = self.max_step + 2
        else:
            self.max_step = ( step + self.max_step +2 ) // 2  
        Ldi = torch.unsqueeze(LT, dim=2) / (K.matmul(AF)+1)
        AF = torch.unsqueeze(AT, dim=1)  / (K.T.matmul(Ldi)+1)  
        Adi = torch.unsqueeze(AT, dim=1) / (K.T.matmul(LF)+1)
        LF = torch.unsqueeze(LT, dim=2) / (K.matmul(Adi)+1)


        #print(Ldi.shape, AF.shape, Adi.shape, LF.shape)
        #print(K.shape, u.shape, e.shape, b.shape)

        AF = AF.transpose(1, 2)
        D = (K * (LF.matmul(AF))).view([LT.shape[0],-1])
        AF = torch.squeeze(AF, dim=1)

        y = D.matmul(u) + AF.matmul(e) + torch.unsqueeze(b, dim=0)
        return y 
    def pic(self, LT):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        AF=torch.zeros([LT.shape[0],self.AT.shape[0],1]).to(dev)
        LF=torch.zeros([LT.shape[0],LT.shape[1],1]).to(dev)
        K = mask(self.K) 
        AT = mask(self.AT) 
        u = mask(self.u) 
        e = mask(self.e)
        b = mask(self.b) 
        
        for step in range(self.max_step):
            AF_last = AF.detach() +0.0
            Ldi = torch.unsqueeze(LT, dim=2) / (K.detach().matmul(AF)+1)
            AF = torch.unsqueeze(AT.detach(), dim=1) / (K.detach().T.matmul(Ldi)+1)  
            
            LF_last = LF.detach() +0.0
            Adi = torch.unsqueeze(AT.detach(), dim=1) / (K.detach().T.matmul(LF)+1)
            LF = torch.unsqueeze(LT, dim=2) / (K.detach().matmul(Adi)+1)
            
            if ((AF.detach() - LF_last).abs()/(LF_last+1e-5) ).max() + ((LF.detach() - AF_last).abs()/(AF_last+1e-5) ).max()< 1e-3:
                break
        if step==self.max_step - 1:
            self.max_step = self.max_step + 2
        else:
            self.max_step = ( step + self.max_step +2 ) // 2  
        Ldi = torch.unsqueeze(LT, dim=2) / (K.matmul(AF)+1)
        AF = torch.unsqueeze(AT, dim=1)  / (K.T.matmul(Ldi)+1)  
        Adi = torch.unsqueeze(AT, dim=1) / (K.T.matmul(LF)+1)
        LF = torch.unsqueeze(LT, dim=2) / (K.matmul(Adi)+1)
        AF = AF.transpose(1, 2)
        D = (K * (LF.matmul(AF))).view([LT.shape[0],-1])
        AF = torch.squeeze(AF, dim=1)
        return torch.cat([D,AF],1)

# dev = torch.device("cuda") if torch.cuda.is_available() and 1 else torch.device("cpu")
dev = torch.device("cpu")
batch_size=256
max_epochs=20000

loss_func = F.cross_entropy
D_in, D_out=2,1


a = 0

b = 1
ab = a+b
x = np.array([[a, a],
       [ab, a],
       [a, ab],
       [ab, ab]])
y = np.array([0.0,1.0,1.0,0.0])

f1 = y.reshape([2,2])

plt.imshow(f1)
plt.title('目标模式')
# plt.show()

x_train = torch.tensor(x,dtype=torch.float32).to(dev)
y_train = torch.tensor(y,dtype=torch.float32).to(dev)

x_test = x_train+0.0
y_test = y_train + 0
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)


# while 1:    
model = TwoLayerNet(D_in, 2, D_out).to(dev)
opt = swats.SWATS(model.parameters())
opt = torch.optim.Adam(model.parameters())
    
max_acc=0
max_epoch=0
epoch_flag=10000
best_model=copy.deepcopy(model)

for epoch in range(max_epochs):
    #if epoch%20==0 or epoch<10: 
    #    print(epoch,model.AT[model.mask==1].detach().cpu().numpy())		
    for xb, yb in train_dl:
        pred = model(xb)
        loss = 1-my_ssim(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.AT.data = F.relu(model.AT) 
    test_acc =  predict_acc (x_test, y_test) 

    if (epoch % 1000 == 0):
        print('epoch', epoch, 'loss', loss.item(), 'test_acc', test_acc)

    if (epoch > 15000 and epoch % 100 == 0):
        print('epoch', epoch, 'loss', loss.item(), 'test_acc', test_acc)
        print(model.K, model.AT, model.u, model.e, model.b)

    if (test_acc > 0.99):
        print('epoch', epoch, 'loss', loss.item(), 'test_acc', test_acc)
        break
    '''
    if test_acc>max_acc*1.001:
        epoch_flag=1000
        max_epoch=epoch
        max_acc = test_acc
        best_model=copy.deepcopy(model)
        if test_acc>0.99:
            break
    elif epoch_flag>0:
        epoch_flag-=1
    else:
        break 
    '''


'''
#测试集
model=copy.deepcopy(best_model)
test_acc =  predict_acc ( x_test, y_test)
if test_acc>0.99:
    print('结构相似性：%.6f   拟合成功'%test_acc)
    break
print('结构相似性：%.6f    拟合失败'%test_acc)
'''
#1/0






f = model(x_test).reshape([grids,grids]).cpu().detach().numpy()
print(f)
plt.imshow(f)
plt.title('输出模式')
#plt.show()

