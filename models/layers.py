import time
import numpy as np
import torch
import torch.nn as nn

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









# 模型不含BT
class CompetitiveLayer(nn.Module):
    def __init__(self, nA, nB, reparam, mode):
        super().__init__()
        self.nA = nA
        self.nB = nB
        self.reparam = reparam
        self.mode = mode

        # 可训练的参数只有K
        self.k = nn.Parameter(torch.empty(nA, nB))
        self.K = torch.empty(nA, nB)
        self.reset_parameters()
        #self.reparameterize() 多进程写这句会报错


    def reset_parameters(self):
        if (self.reparam == 'none'):
            nn.init.uniform_(self.k, 0, 1)
        elif (self.reparam == 'pow'):
            nn.init.uniform_(self.k, 0, 1)
        elif (self.reparam == 'exp'):
            nn.init.uniform_(self.k, -1, 0)
        else:
            print('K not initialized')


    def reparameterize(self):
        # 重参数化保证参数满足限制
        if (self.reparam == 'none'):
            self.K = self.k
        elif (self.reparam == 'pow'):
            self.K = torch.pow(self.k)
        elif (self.reparam == 'exp'):
            self.K = torch.exp(self.k)
        else:
            print('K not reparameterized')


    def forward(self, AT: torch.Tensor, BT: torch.Tensor):
        self.reparameterize()
        K = self.K

        # 不固定A和B，numpy, 矩阵求逆求梯度，batch=1
        if self.mode == 'comp_1':
            C = competitive_layer_function(AT, BT, K)

        # 不固定A和B，numpy, batch=1
        elif self.mode == 'comp_2':
            AF, BF, C = solvers.numpy_solve(AT.numpy(), BT.numpy(), K.detach().numpy())
            AF, BF, C = torch.from_numpy(AF), torch.from_numpy(BF), torch.from_numpy(C)
            AF, BF, C = solvers.torch_iterate_last(AT, BT, K, AF, BF, C)

        # 不固定A和B，torch
        elif self.mode == 'comp':
            AT = AT.unsqueeze(dim=2)
            BT = BT.unsqueeze(dim=1)
            AF, BF, C = solvers.torch_solve(AT, BT, K)
            # simultaneous iteration
            AF_ = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)
            BF_ = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
            # alternative iteration
            # AF = AT / ((K * BF).sum(axis=1, keepdims=True) + 1)
            # BF = BT / ((K * AF).sum(axis=0, keepdims=True) + 1)
            C = AF_ * BF_ * K

        # semicomp : 固定输入A
        elif (self.mode == 'semicomp'):
            AT = AT.unsqueeze(dim=2)
            BT = BT.unsqueeze(dim=1)
            AF = AT
            #print((K*AF).shape, (K*AF).sum(axis=1, keepdim=True).shape)
            BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
            #print(AF.shape, BF.shape, K.shape)
            C = AF * BF * K

        # semicomp : 固定输入B
        elif (self.mode == 'semicomp_B'):
            AT = AT.unsqueeze(dim=2)
            BT = BT.unsqueeze(dim=1)
            BF = BT
            #print((K*AF).shape, (K*AF).sum(axis=1, keepdim=True).shape)
            AF = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)
            #print(AF.shape, BF.shape, K.shape)
            C = AF * BF * K

        # noncomp : 固定输入输出
        elif (self.mode == 'noncomp'):
            AT = AT.unsqueeze(dim=2)
            BT = BT.unsqueeze(dim=1)
            AF = AT
            BF = BT
            C = AF * BF * K

        return C






class CompetitiveLayer(nn.Module):
    def __init__(self, nA, nB, reparam, mode, trainBT):
        super().__init__()
        self.nA = nA
        self.nB = nB
        self.reparam = reparam
        self.mode = mode

        self.k = nn.Parameter(torch.empty(nA, nB))
        self.bt = nn.Parameter(torch.empty(1, nB)) # 如果bt不需要梯度，则固定bt，如果bt需要梯度，则bt是可训练参数
        self.bt.requires_grad_(trainBT)
        self.reset_parameters()
        #self.reparameterize() 多进程写这句会报错


    @property
    def K(self):
        K = self.reparameterize(self.k)
        return K

    @property
    def BT(self):
        BT = self.reparameterize(self.bt)
        return BT


    def reset_parameters(self):
        if (self.reparam == 'none'):
            nn.init.uniform_(self.k, 0, 1)
        elif (self.reparam == 'pow'):
            nn.init.uniform_(self.k, 0, 1)
        elif (self.reparam == 'exp'):
            nn.init.uniform_(self.k, -1, 1)
        else:
            print('k not initialized')

        if (self.reparam == 'none'):
            nn.init.constant_(self.bt, 1)
        elif (self.reparam == 'pow'):
            nn.init.constant_(self.bt, 1)
        elif (self.reparam == 'exp'):
            nn.init.constant_(self.bt, 0)
        else:
            print('bt not initialized')


    def reparameterize(self, p):
        # 重参数化保证参数满足限制
        if (self.reparam == 'none'):
            P = p
        elif (self.reparam == 'pow'):
            P = torch.pow(p, 2)
        elif (self.reparam == 'exp'):
            P = torch.exp(p)
        else:
            print('p not reparameterized')

        return P


    def forward(self, AT: torch.Tensor):
        K = self.K
        BT = self.BT.repeat(len(AT), 1)
        # BT = self.BT.expand(len(AT), -1) # repeat和expand一样

        # 不固定A和B，numpy, 矩阵求逆求梯度，batch=1
        if self.mode == 'comp_1':
            C = competitive_layer_function(AT, BT, K)

        # 不固定A和B，numpy, batch=1
        elif self.mode == 'comp_2':
            AF, BF, C = solvers.numpy_solve(AT.numpy(), BT.numpy(), K.detach().numpy())
            AF, BF, C = torch.from_numpy(AF), torch.from_numpy(BF), torch.from_numpy(C)
            AF, BF, C = solvers.torch_iterate_last(AT, BT, K, AF, BF, C)

        # 不固定A和B，torch
        elif self.mode == 'comp':
            AT = AT.unsqueeze(dim=2)
            BT = BT.unsqueeze(dim=1)
            AF, BF, C = solvers.torch_solve(AT, BT, K)
            # simultaneous iteration
            AF_ = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)
            BF_ = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
            # alternative iteration
            # AF = AT / ((K * BF).sum(axis=1, keepdims=True) + 1)
            # BF = BT / ((K * AF).sum(axis=0, keepdims=True) + 1)
            C = AF_ * BF_ * K

        # semicomp : 固定输入A
        elif (self.mode == 'semicomp'):
            AT = AT.unsqueeze(dim=2)
            BT = BT.unsqueeze(dim=1)
            AF = AT
            #print((K*AF).shape, (K*AF).sum(axis=1, keepdim=True).shape)
            BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
            #print(AF.shape, BF.shape, K.shape)
            C = AF * BF * K

        # semicomp : 固定输入B
        elif (self.mode == 'semicomp_B'):
            AT = AT.unsqueeze(dim=2)
            BT = BT.unsqueeze(dim=1)
            BF = BT
            #print((K*AF).shape, (K*AF).sum(axis=1, keepdim=True).shape)
            AF = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)
            #print(AF.shape, BF.shape, K.shape)
            C = AF * BF * K

        # noncomp : 固定输入输出
        elif (self.mode == 'noncomp'):
            AT = AT.unsqueeze(dim=2)
            BT = BT.unsqueeze(dim=1)
            AF = AT
            BF = BT
            C = AF * BF * K

        return C







'''
# 继承
class CompetitiveLayer(CompetitiveLayer):
    def __init__(self, nA, nB, reparam, mode, trainBT=False):
        super().__init__(nA, nB, reparam, mode)

        self.bt = nn.Parameter(torch.empty(1, nB))
        # 如果bt不需要梯度，则固定bt，如果bt需要梯度，则bt是可训练参数
        self.bt.requires_grad_(trainBT)

        self.reset_parameters()
        #self.reparameterize() 多进程写这句会报错


    def reset_parameters(self):
        super().reset_parameters()
        if (self.reparam == 'none'):
            nn.init.constant_(self.bt, 1)
        elif (self.reparam == 'pow'):
            nn.init.constant_(self.bt, 1)
        elif (self.reparam == 'exp'):
            nn.init.constant_(self.bt, 0)
        else:
            print('K not initialized')


    def reparameterize(self):
        super().reparameterize()
        if (self.reparam == 'none'):
            self.BT = self.bt
        elif (self.reparam == 'pow'):
            self.BT = torch.pow(self.bt)
        elif (self.reparam == 'exp'):
            self.BT = torch.exp(self.bt)
        else:
            print('K not reparameterized')


    def forward(self, AT):
        super().reset_parameters()
        self.reparameterize()

        BT = self.BT.repeat(len(AT), 1)
        C = super().forward(AT, BT)
        return C





class CompetitiveLayer(nn.Module):
    def __init__(self, nA, nB, reparam, mode, trainBT=False):
        super().__init__()
        self.nA = nA
        self.nB = nB
        self.reparam = reparam
        self.mode = mode

        self.layer = CompetitiveLayer(nA, nB, reparam, mode)
        self.bt = nn.Parameter(torch.empty(1, nB))
        self.BT = torch.empty(1, nB)
        # 如果bt不需要梯度，则固定bt，如果bt需要梯度，则bt是可训练参数
        self.bt.requires_grad_(trainBT)
        
        self.reset_parameters()
        #self.reparameterize() 多进程写这句会报错


    def reset_parameters(self):
        if (self.reparam == 'none'):
            nn.init.constant_(self.bt, 1)
        elif (self.reparam == 'pow'):
            nn.init.constant_(self.bt, 1)
        elif (self.reparam == 'exp'):
            nn.init.constant_(self.bt, 0)
        else:
            print('K not initialized')


    def reparameterize(self):
        if (self.reparam == 'none'):
            self.BT = self.bt
        elif (self.reparam == 'pow'):
            self.BT = torch.pow(self.bt)
        elif (self.reparam == 'exp'):
            self.BT = torch.exp(self.bt)
        else:
            print('K not reparameterized')


    def forward(self, AT):
        self.reparameterize()
        BT = self.BT.repeat(len(AT), 1)
        C = self.layer(AT, BT)
        return C'''


if __name__ == '__main__':

    nA = 2
    nB = 3
    nY = 1
    AT = torch.Tensor([1, 1]).reshape(1,2)
    BT = torch.Tensor([1, 1, 1]).reshape(1,3)
    K  = torch.Tensor([1, 2, 3, 4, 5, 6]).reshape(2,3)
    K.requires_grad = True
    Y =  torch.Tensor([1.])

    '''print('numerical autocheck')
    autocheck = torch.autograd.gradcheck(competitive_layer_function, inputs=(AT, BT, K), eps=1e-3, atol=1e-3)
    print(autocheck)'''


    for mode in ['comp', 'semicomp_B', 'semicomp', 'noncomp']:

        print('test layer', mode)
        layer = CompetitiveLayer(nA, nB, reparam='none', mode=mode)
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



