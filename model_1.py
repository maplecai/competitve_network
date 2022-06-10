import numpy as np
import torch
from competitive_solver import *


class CompetitiveLayerFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, AT, BT, K):
        # 求解稳态
        AF, BF, C = solver.torch_solve(AT, BT, K)
        ctx.save_for_backward(AF, BF, K)
        return C

    @staticmethod
    def backward(ctx, grad_output):
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


# my layer and model
class CompetitiveLayer(torch.nn.Module):
    def __init__(self, nA, nB):
        super(CompetitiveLayer, self).__init__()
        # 可训练的参数只有K
        self.K = nn.Parameter(torch.empty(nA, nB))
        nn.init.uniform_(self.K, 0, 1)

    def forward(self, AT, BT):
        C = competitive_layer_function(AT, BT, self.K)
        return C


class CompetitiveNetwork(torch.nn.Module):
    def __init__(self, nA, nB, nY, constrain_mode=None):
        super(CompetitiveNetwork, self).__init__()
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


# other layer and model
class CompetitiveLayer_2(torch.nn.Module):
    def __init__(self, nA, nB):
        super(CompetitiveLayer_2, self).__init__()
        self.K = nn.Parameter(torch.empty(nA, nB))
        nn.init.uniform_(self.K, 0, 1)

    def forward(self, AT, BT):
        with torch.no_grad():
            AF, BF, C = solver.torch_solve(AT, BT, self.K)
        AF, BF, C = solver.torch_iterate_once(AT, BT, self.K, AF, BF, C)
        return C


class CompetitiveNetwork_2(torch.nn.Module):
    def __init__(self, nA, nB, nY, constrain_mode=None):
        super(CompetitiveNetwork_2, self).__init__()
        self.comp_layer = CompetitiveLayer_2(nA, nB)
        self.linear = nn.Linear(nA*nB, nY)
        self.constrain_mode = constrain_mode
        
    def forward(self, AT, BT):
        C = self.comp_layer(AT, BT)
        C = C.reshape(-1)
        Y = self.linear(C)
        return Y




if __name__ == '__main__':

    '''
    # autograd_check
    input = (AT, BT, K)
    test = torch.autograd.gradcheck(competitive_layer_function, input, eps=1e-3, atol=1e-3)
    print(test)
    '''

    # test my layer

    AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1., 1.])
    K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    nA = len(AT)
    nB = len(BT)

    layer = CompetitiveLayer(nA, nB)
    layer.K.data = K

    C = layer(AT, BT)
    print(C)
    
    for i in range(nA):
        for j in range(nB):
                C[i, j].backward(retain_graph=True)
                print(f'dC{i}{j}_dK'.format(i, j))
                print(layer.K.grad)
                #K.grad.detach_()
                layer.K.grad.zero_()



    # test my network
    Y =  torch.Tensor([1.])
    nY = len(Y)
    model = CompetitiveNetwork(nA, nB, nY)

    model.comp_layer.K.data  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    model.linear.weight.data = torch.Tensor([1., 0., 0., 0., 0., 0.]).reshape(1, 6)
    model.linear.bias.data = torch.Tensor([0.])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    Y_pred = model(AT, BT)
    Y_pred.backward()
    print('model_1')
    print(model.comp_layer.K.grad)
    optimizer.step()
    optimizer.zero_grad()



    # test my layer 2

    AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1., 1.])
    K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    nA = len(AT)
    nB = len(BT)

    layer = CompetitiveLayer_2(nA, nB)
    layer.K.data = K

    C = layer(AT, BT)
    print(C)
    
    for i in range(nA):
        for j in range(nB):
                C[i, j].backward(retain_graph=True)
                print(f'dC{i}{j}_dK'.format(i, j))
                print(layer.K.grad)
                #K.grad.detach_()
                layer.K.grad.zero_()

    # test my network 2
    Y =  torch.Tensor([1.])
    nY = len(Y)
    model = CompetitiveNetwork_2(nA, nB, nY)

    model.comp_layer.K.data  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    model.linear.weight.data = torch.Tensor([1., 0., 0., 0., 0., 0.]).reshape(1, 6)
    model.linear.bias.data = torch.Tensor([0.])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    Y_pred = model(AT, BT)
    Y_pred.backward()
    print('model_2')
    print(model.comp_layer.K.grad)
    optimizer.step()
    optimizer.zero_grad()




'''
    loss = criterion(Y, Y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
'''