import numpy as np
import torch
import torch.nn.functional as F
from utils import *

class TwoLayerNet(torch.nn.Module):
    def __init__(self, A_dim, B_dim, y_dim, max_iter=16, tol=1e-3, constrain='reparam'):
        super(TwoLayerNet, self).__init__()

        self.K = nn.Parameter(torch.rand(A_dim, B_dim))
        self.BT = nn.Parameter(torch.rand(B_dim))
        self.u = nn.Parameter(torch.rand(A_dim*B_dim, y_dim))
        self.b = nn.Parameter(torch.rand(y_dim))

        # fix BT
        self.BT.data = torch.Tensor([1,1])
        self.BT.requires_grad = False

        self.max_iter = max_iter
        self.tol = tol
        self.constrain = constrain
        
    def forward(self, AT):
        batch_size = AT.shape[0]

        if (self.constrain == 'reparam'):
            # reparameterize x^2
            K = (self.K) ** 2
            BT = (self.BT)
            u = (self.u)
            b = (self.b)
        
        elif (self.constrain == 'clip'):
            # cilp to (0, inf)
            self.K.data = F.relu(self.K)
            K = (self.K)
            BT = (self.BT)
            u = (self.u)
            b = (self.b)

        AF = torch.zeros((batch_size, K.shape[0], 1)).to(self.K.device)
        BF = torch.zeros((batch_size, K.shape[1], 1)).to(self.K.device)
        
        with torch.no_grad():
            err = 0
            for iter in range(self.max_iter):
                AF_last = AF
                BF_last = BF

                AF = torch.unsqueeze(AT, dim=2) / (K.matmul(BF)+1)
                BF = torch.unsqueeze(BT, dim=1) / (K.T.matmul(AF)+1)
                
                err = (AF-AF_last).abs().max() + (BF-BF_last).abs().max()
                if (err < self.tol):
                    break

            if (err < self.tol):
                pass
                # print(f'converge in {iter} iterations')
            else:
                print(f'not converge in {iter} iterations')
                self.max_iter *= 2

        AF = torch.unsqueeze(AT, dim=2) / (K.matmul(BF)+1)
        BF = torch.unsqueeze(BT, dim=1) / (K.T.matmul(AF)+1)

        C = K * AF.matmul(BF.transpose(1, 2))
        C = C.reshape(batch_size, -1)
        y = C.matmul(u) + b

        return y



if __name__ == '__main__':

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    model = TwoLayerNet(A_dim=2, B_dim=2, y_dim=1, max_iter=10, tol=1e-3, constrain='reparam')
    y = model(x)
    print(y)
