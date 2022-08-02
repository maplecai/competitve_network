import sys
import numpy as np
import torch
import torch.nn as nn
import time
from . import solvers


class CompetitiveLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, AT, BT, K):
        # 求解稳态
        # AF, BF, C = solver.torch_solve(AT, BT, K)
        # ctx.save_for_backward(AF, BF, K)

        # 不并行的情况下numpy计算更快  不知道为什么K不需要.detach()
        AF, BF, C = solvers.numpy_solve(AT.numpy(), BT.numpy(), K.detach().numpy())
        AF, BF, C = torch.from_numpy(AF), torch.from_numpy(BF), torch.from_numpy(C)
        ctx.save_for_backward(AF, BF, K)
        
        return C

    @staticmethod
    def backward(ctx, grad_output):
        # 求解梯度
        AF, BF, K = ctx.saved_tensors
        nA, nB = K.shape
        grad_AT, grad_BT, grad_K = None, None, None

        pA_pK, pB_pK, pC_pK = solvers.numpy_gradient(AF.numpy(), BF.numpy(), K.numpy())
        pA_pK, pB_pK, pC_pK = torch.from_numpy(pA_pK), torch.from_numpy(pB_pK), torch.from_numpy(pC_pK)
        # print(pC_pK.shape)
        
        grad_K = (pC_pK * grad_output.reshape(nA, nB, 1, 1)).sum(axis=[0,1])
        return grad_AT, grad_BT, grad_K

competitive_layer_function = CompetitiveLayerFunction.apply


class CompetitiveLayer(nn.Module):
    def __init__(self, nA, nB, reparam, mode, trainBT=False):
        super(CompetitiveLayer, self).__init__()
        # 可训练的参数只有K
        self.nA = nA
        self.nB = nB
        self.reparam = reparam
        self.mode = mode

        self.k = nn.Parameter(torch.empty(nA, nB))
        self.bt = nn.Parameter(torch.empty(nB))
        # 如果bt不需要梯度，等价于固定bt
        # self.bt.requires_grad_(False)
            
        self.reset_parameters()
        #self.reparameterize() 多进程会报错


    def reset_parameters(self):
        if (self.reparam == 'none'):
            nn.init.uniform_(self.k, 0, 1)
        elif (self.reparam == 'square'):
            nn.init.uniform_(self.k, 0, 1)
        elif (self.reparam == 'exp'):
            nn.init.uniform_(self.k, -1, 0)
        else:
            print('K not initialized')

        nn.init.constant_(self.bt, 1)


    def reparameterize(self):
        # 不同重参数化的方法
        if (self.reparam == 'none'):
            self.K = self.k
            self.BT = self.bt
        elif (self.reparam == 'square'):
            self.K = torch.square(self.k)
            self.BT = torch.square(self.bt)
        elif (self.reparam == 'exp'):
            self.K = torch.exp(self.k)
            self.BT = torch.exp(self.bt)
        else:
            print('K not reparameterized')


    def forward(self, AT, BT=None):
        self.reparameterize()
        
        K = self.K
        if (BT is None):
            BT = self.BT

        # 不固定A和B，用矩阵求逆求梯度
        if self.mode == 'notfix':
            C = competitive_layer_function(AT, BT, K)

        # 不固定A和B，用最后一次torch迭代近似，并没有变快
        elif self.mode == 'iterate':
            AF, BF, C = solvers.numpy_solve(AT.numpy(), BT.numpy(), K.detach().numpy())
            AF, BF, C = torch.from_numpy(AF), torch.from_numpy(BF), torch.from_numpy(C)
            AF, BF, C = solvers.torch_iterate_last(AT, BT, K, AF, BF, C)

        # 固定A的浓度，计算能力基本不变
        elif (self.mode == 'fixA'):
            C = (AT.reshape(-1, 1)*K*BT.reshape(1, -1)) / ((K*AT.reshape(-1, 1)).sum(axis=0, keepdims=True) + 1)

        # 固定B的浓度，导致退化成线性层
        elif (self.mode == 'fixB'):
            C = (AT.reshape(-1, 1)*K*BT.reshape(1, -1)) / ((K*BT.reshape(1, -1)).sum(axis=1, keepdims=True) + 1)

        return C



class CompetitiveNetwork(nn.Module):
    def __init__(self, nA, nB, nY, reparam, mode, trainBT=False):
        super(CompetitiveNetwork, self).__init__()
        self.comp_layer = CompetitiveLayer(nA, nB, reparam=reparam, mode=mode, trainBT=trainBT)
        self.linear = nn.Linear(nA*nB, nY)
        
    def forward(self, AT, BT=None):
        C = self.comp_layer(AT, BT)
        C = C.reshape(-1)
        Y = self.linear(C)
        return Y

    def reset_parameters(self):
        for module in self.children():
            module.reset_parameters()



if __name__ == '__main__':

    AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1., 1.])
    K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    Y =  torch.Tensor([1.])
    K.requires_grad = True
    nA = len(AT)
    nB = len(BT)
    nY = len(Y)

    print('numerical autocheck')
    autocheck = torch.autograd.gradcheck(competitive_layer_function, inputs=(AT, BT, K), eps=1e-3, atol=1e-3)
    print(autocheck)

    print('test my layer')
    layer = CompetitiveLayer(nA, nB, reparam='none', mode='notfix')
    layer.k.data = K

    C = layer(AT, BT)
    print(C)



    t0 = time.perf_counter()
    for i in range(1000):
        C = layer(AT, BT)
    t1 = time.perf_counter()
    print('forward time', t1-t0)

    t0 = time.perf_counter()
    for i in range(1000):
        C = layer(AT, BT)
        C.sum().backward()
    t1 = time.perf_counter()
    print('forward and backward time', t1-t0)

    print('test my network')

    model = CompetitiveNetwork(nA, nB, nY, reparam='exp', mode='notfix')

    t0 = time.perf_counter()
    for i in range(1000):
        Y_pred = model(AT, BT)
    t1 = time.perf_counter()
    print('forward time', t1-t0)
    
    model = CompetitiveNetwork(nA, nB, nY, reparam='exp', mode='notfix')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    t0 = time.perf_counter()
    for i in range(1000):
        model.reset_parameters()
        Y_pred = model(AT, BT)
        Y_pred.backward()
        optimizer.step()
        optimizer.zero_grad()
    t1 = time.perf_counter()
    print('forward and backward time', t1-t0)



# trainBT
if __name__ == '__main__':

    AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1., 1.])
    K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    Y =  torch.Tensor([1.])
    K.requires_grad = True
    nA = len(AT)
    nB = len(BT)
    nY = len(Y)

    print('numerical autocheck')
    autocheck = torch.autograd.gradcheck(competitive_layer_function, inputs=(AT, BT, K), eps=1e-3, atol=1e-3)
    print(autocheck)

    print('test my layer')
    layer = CompetitiveLayer(nA, nB, reparam='none', mode='notfix', trainBT=True)
    layer.k.data = K

    C = layer(AT)
    print(C)



    t0 = time.perf_counter()
    for i in range(1000):
        C = layer(AT)
    t1 = time.perf_counter()
    print('forward time', t1-t0)

    t0 = time.perf_counter()
    for i in range(1000):
        C = layer(AT)
        C.sum().backward()
    t1 = time.perf_counter()
    print('forward and backward time', t1-t0)

    print('test my network')

    model = CompetitiveNetwork(nA, nB, nY, reparam='exp', mode='notfix', trainBT=True)

    t0 = time.perf_counter()
    for i in range(1000):
        Y_pred = model(AT)
    t1 = time.perf_counter()
    print('forward time', t1-t0)
    
    model = CompetitiveNetwork(nA, nB, nY, reparam='exp', mode='notfix', trainBT=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    t0 = time.perf_counter()
    for i in range(1000):
        model.reset_parameters()
        Y_pred = model(AT)
        Y_pred.backward()
        optimizer.step()
        optimizer.zero_grad()
    t1 = time.perf_counter()
    print('forward and backward time', t1-t0)

'''
test my layer
tensor([[0.1543, 0.2517, 0.3188],
        [0.2858, 0.2915, 0.2954]], grad_fn=<CompetitiveLayerFunctionBackward>)
forward time 0.24688729998888448
forward and backward time 0.46601119998376817
test my network
forward time 0.2172947000071872
forward and backward time 0.6603156000201125
'''
