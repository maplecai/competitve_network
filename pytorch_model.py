import numpy as np
import torch
import math
from gradient_simulation import *

class CompetitiveLayerFunction(torch.autograd.Function):
    solver = Solver()
    
    @staticmethod
    def forward(ctx, AT, BT, K):
        # 求解稳态
        AF, BF, C = CompetitiveLayerFunction.solver.torch_solve(AT, BT, K)
        ctx.save_for_backward(AF, BF, K)
        return C

    @staticmethod
    def backward(ctx, grad_output):
        AF, BF, K = ctx.saved_tensors
        grad_AT, grad_BT, grad_K = None, None, None

        pK = torch.zeros(K.shape)
        nA = len(AF)
        nB = len(BF)
        # 求解pC/pKpq
        for p in range(nA):
            for q in range(nB):
                with torch.no_grad():
                    pC = CompetitiveLayerFunction.solver.np_gradient(AF, BF, K, p, q)
                    pC = torch.Tensor(pC)
                pK[p, q] = (pC * grad_output).sum()
        grad_K = pK

        return grad_AT, grad_BT, grad_K


competitive_layer = CompetitiveLayerFunction.apply


class CompetitiveLayer(torch.nn.Module):
    def __init__(self, nA, nB):
        super(CompetitiveLayer, self).__init__()
        self.nA = nA
        self.nB = nB
        self.K = nn.Parameter(torch.empty(nA, nB))
        nn.init.uniform_(self.K, 0, 1)

    def forward(self, AT, BT):
        return competitive_layer(AT, BT, self.K)


# my network
class CompetitiveNetwork(torch.nn.Module):
    def __init__(self, nA, nB, nY, constrain_mode=None):
        super(CompetitiveNetwork, self).__init__()
        self.nA = nA
        self.nB = nB
        self.nY = nY
        self.comp_layer = CompetitiveLayer(nA, nB)
        #self.comp_layer.K.data = torch.Tensor([1,2,3,4,5,6]).reshape(2, 3)
        self.linear = nn.Linear(nA*nB, nY)
        self.constrain_mode = constrain_mode
        
    def forward(self, AT, BT):
        #batch_size = len(AT)
        C = self.comp_layer(AT, BT)
        #C = C.reshape(batch_size, -1)
        C = C.reshape(-1)
        Y = self.linear(C)
        return Y





if __name__ == '__main__':

    # test my layer

    '''AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1., 1.])
    K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    K.requires_grad = True

    cl = CompetitiveLayer(2, 3)
    cl.K.data = K

    C = cl(AT, BT)
    w = torch.Tensor([1,0,0,0,0,0]).reshape(2, 3)
    
    y = (C*w).sum()
    y.backward(retain_graph=True)
    print('dC00/dK')
    print(cl.K.grad)

    input = (AT, BT, K)
    test = torch.autograd.gradcheck(competitive_layer, input, eps=1e-3, atol=1e-3)
    print(test)'''





    # test my network

    model = CompetitiveNetwork(nA=2, nB=2, nY=1)

    AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1.])
    Y =  torch.Tensor([1.])

    #K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)

    Y_pred = model(AT, BT)
    print(Y)
    print(Y_pred)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    loss = criterion(Y, Y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

