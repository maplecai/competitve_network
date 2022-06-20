import numpy as np
import torch
from competitive_solver import *
import time


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


# layer1
class CompetitiveLayer_1(torch.nn.Module):
    def __init__(self, nA, nB):
        super(CompetitiveLayer_1, self).__init__()
        # 可训练的参数只有K
        # self.K = nn.Parameter(torch.empty(nA, nB))
        self.sqrt_K = nn.Parameter(torch.empty(nA, nB))
        nn.init.uniform_(self.sqrt_K, 0, 1)

    def forward(self, AT, BT):
        K = self.sqrt_K ** 2
        C = competitive_layer_function(AT, BT, K)
        return C


# layer2
class CompetitiveLayer_2(torch.nn.Module):
    def __init__(self, nA, nB):
        super(CompetitiveLayer_2, self).__init__()
        # 可训练的参数只有K
        # self.K = nn.Parameter(torch.empty(nA, nB))
        self.sqrt_K = nn.Parameter(torch.empty(nA, nB))
        nn.init.uniform_(self.sqrt_K, 0, 1)


    def forward(self, AT, BT):
        K = self.sqrt_K ** 2
        with torch.no_grad():
            AF, BF, C = solver.torch_solve(AT, BT, K)
        AF, BF, C = solver.torch_iterate_once(AT, BT, K, AF, BF, C)
        return C


class CompetitiveNetwork_1(torch.nn.Module):
    def __init__(self, nA, nB, nY, constrain_mode=None):
        super(CompetitiveNetwork_1, self).__init__()
        self.comp_layer = CompetitiveLayer_1(nA, nB)
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




class MLP(torch.nn.Module):
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
    # autograd_check
    input = (AT, BT, K)
    test = torch.autograd.gradcheck(competitive_layer_function, input, eps=1e-3, atol=1e-3)
    print(test)
    '''


    print('test my layer_1')

    AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1., 1.])
    K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    Y =  torch.Tensor([1.])
    nA = len(AT)
    nB = len(BT)
    nY = len(Y)

    layer = CompetitiveLayer_1(nA, nB)
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



    print('test my network_1')

    model = CompetitiveNetwork_1(nA, nB, nY)
    model.comp_layer.K.data  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    model.linear.weight.data = torch.Tensor([1., 0., 0., 0., 0., 0.]).reshape(1, 6)
    model.linear.bias.data = torch.Tensor([0.])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    t0 = time.perf_counter()
    
    Y_pred = model(AT, BT)

    t1 = time.perf_counter()

    Y_pred.backward()
    optimizer.step()
    optimizer.zero_grad()

    t2 = time.perf_counter()
    
    print(t1-t0)
    print(t2-t0)


