import numpy as np
import torch
import torch.nn as nn
import time
from .competitive_solver import *


solver = CompetitiveSolver(device="cpu", max_iter=20, tol=1e-3)


class CompetitiveLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, AT, BT, K):
        # 求解稳态
        AF, BF, C = solver.torch_solve(AT, BT, K)
        ctx.save_for_backward(AF, BF, K)
        return C

    @staticmethod
    def backward(ctx, grad_output):
        # 求解梯度
        AF, BF, K = ctx.saved_tensors
        nA, nB = K.shape
        grad_AT, grad_BT, grad_K = None, None, None
        pC_pK = solver.np_gradient_all(AF.numpy(), BF.numpy(), K.numpy())
        pC_pK = torch.from_numpy(pC_pK)
        #print(pC_pK.shape)
        grad_K = (pC_pK * grad_output.reshape(nA, nB, 1, 1)).sum(axis=[0,1])
        #print(grad_K.shape)
        return grad_AT, grad_BT, grad_K


competitive_layer_function = CompetitiveLayerFunction.apply


# layer1
class CompetitiveLayer(nn.Module):
    def __init__(self, nA, nB, reparameterize, gradient):
        super(CompetitiveLayer, self).__init__()
        # 可训练的参数只有K
        self.nA = nA
        self.nB = nB
        self.reparameterize = reparameterize
        self.gradient = gradient
        self.param = nn.Parameter(torch.empty(nA, nB))
        self.reset_parameters()

    def reset_parameters(self):
        if (self.reparameterize == 'none'):
            nn.init.uniform_(self.param, 0, 1)
        elif (self.reparameterize == 'square'):
            nn.init.uniform_(self.param, 0, 1)
        elif (self.reparameterize == 'exp'):
            nn.init.uniform_(self.param, -1, 0)

    def forward(self, AT, BT):
        # 不同重参数化的方法
        if (self.reparameterize == 'none'):
            K = self.param
        elif (self.reparameterize == 'square'):
            K = torch.square(self.param)
        elif (self.reparameterize == 'exp'):
            K = torch.exp(self.param)
        # 不同求梯度的方法
        if (self.gradient == 'linear_algebra'):
            C = competitive_layer_function(AT, BT, K)
        elif (self.gradient == 'last_iterate'):
            with torch.no_grad():
                AF, BF, C = solver.torch_solve(AT, BT, K)
            AF, BF, C = solver.torch_iterate_once(AT, BT, K, AF, BF, C)
        return C


class CompetitiveNetwork(nn.Module):
    def __init__(self, nA, nB, nY, reparameterize='none', gradient='linear_algebra', clip=False):
        super(CompetitiveNetwork, self).__init__()
        self.clip = clip
        self.comp_layer = CompetitiveLayer(nA, nB, reparameterize=reparameterize, gradient=gradient)
        self.linear = nn.Linear(nA*nB, nY)
        
    def forward(self, AT, BT):
        C = self.comp_layer(AT, BT)
        C = C.reshape(-1)
        Y = self.linear(C)
        return Y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, AT, BT):
        C = self.linear1(AT)
        C = self.relu(C)
        Y = self.linear2(C)
        return Y



if __name__ == '__main__':
    '''
    print('test my layer')

    AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1., 1.])
    K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    Y =  torch.Tensor([1.])
    nA = len(AT)
    nB = len(BT)
    nY = len(Y)

    layer = CompetitiveLayer(nA, nB)
    layer.K.data = K

    C = layer(AT, BT)
    print(C)
    
    for i in range(nA):
        for j in range(nB):
                C[i, j].backward(retain_graph=True)
                print(f'dC{i}{j}_dK')
                print(layer.K.grad)
                #K.grad.detach_()
                layer.K.grad.zero_()

    # autograd_check
    input = (AT, BT, K)
    autocheck = torch.autograd.gradcheck(competitive_layer_function, input, eps=1e-3, atol=1e-3)
    print(autocheck)
    '''

    print('test my network')

    AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1., 1.])
    K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    Y =  torch.Tensor([1.])
    nA = len(AT)
    nB = len(BT)
    nY = len(Y)

    model = CompetitiveNetwork(nA, nB, nY)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    t0 = time.perf_counter()
    for i in range(100):
        model = CompetitiveNetwork(nA, nB, nY)
        Y_pred = model(AT, BT)
    t1 = time.perf_counter()
    print(t1-t0)

    t0 = time.perf_counter()
    for i in range(100):
        model = CompetitiveNetwork(nA, nB, nY)
        Y_pred = model(AT, BT)
        #Y_pred.backward()
        #optimizer.step()
        #optimizer.zero_grad()
    t1 = time.perf_counter()
    
    print(t1-t0)
