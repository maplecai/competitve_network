import time
import math
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch import tensor, Tensor

from . import solvers




class MixConstrain(nn.Module):
    # (-inf, 1) to (-inf, 1) y = x
    # (1, inf) to (1, inf)   y = e^(x-1)
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # if x <= 1:
        #     y = x
        # else:
        #     y = torch.exp((x-1))
        y = (x <= 1) * x + (x > 1) * torch.exp((x-1))
        return y
    
    def right_inverse(self, y: Tensor) -> Tensor:
        x = (y <= 1) * y + (y > 1) * (torch.log(torch.clamp(y, min=1e-9)) + 1)
        return x



class ExpConstrain(nn.Module):
    # (-inf, inf) to (0, inf)
    def __init__(
        self,
        base: float = math.e,
        ) -> None:
        super().__init__()
        self.base = base
        self.log_base = math.log(self.base)

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.log_base
        x = torch.exp(x)
        return x
    
    def right_inverse(self, x: Tensor) -> Tensor:
        x = torch.clamp(x, 1e-6)
        x = torch.log(x)
        x = x / self.log_base
        return x




class ClampConstrain(nn.Module):
    # (-inf, inf) to (min, max)
    def __init__(
        self,
        min: float = 0.0,
        max: float = 1.0,
        ) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        x = torch.clamp(x, self.min, self.max)
        return x
    
    def right_inverse(self, x: Tensor) -> Tensor:
        x = torch.clamp(x, self.min, self.max)
        return x




class NoConstrain(nn.Module):
    # (-inf, inf) to (min, max)
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x+0
    
    def right_inverse(self, x: Tensor) -> Tensor:
        return x+0



# class SigmoidConstrain(nn.Module):
#     # (-inf, inf) to (0,1) to (log_min, log_max) to (min, max)
#     def __init__(
#         self,
#         min: float = 1e-3,
#         max: float = 1e3,
#         ) -> None:
#         super().__init__()
#         self.min = min
#         self.max = max
#         self.log_min = math.log(self.min)
#         self.log_max = math.log(self.max)

#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.sigmoid(x)
#         x = x * (self.log_max - self.log_min) + self.log_min
#         x = torch.exp(x)
#         return x

#     def right_inverse(self, x: Tensor) -> Tensor:
#         x = torch.clamp(x, 1e-6)
#         x = torch.log(x)
#         x = (x - self.log_min) / (self.log_max - self.log_min)
#         x = torch.clamp(x, 1e-6, 1-1e-6)
#         x = torch.log(x / (1 - x))
#         return x



class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, linear_constrain, *args, **kwargs):
        super().__init__()
        self.linear_layer = ConstrainedLinear(input_size, output_size, bias=True, constrain=linear_constrain)
        if linear_constrain == 'exp' or linear_constrain == 'positive':
            self.last_layer = HillLayer()
        elif linear_constrain == 'none':
            self.last_layer = nn.Sigmoid()
        else:
            raise ValueError

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.last_layer(x)
        x = torch.squeeze(x, dim=1)
        return x





class CompetitiveLayer(nn.Module):
    '''
    竞争层
    '''
    def __init__(
        self,
        nA: int,
        nB: int,
        mode: str,
        constrain: str
        ) -> None:

        super().__init__()
        self.nA = nA
        self.nB = nB
        self.mode = mode
        self.constrain = constrain


        self.K = nn.Parameter(torch.empty(nA, nB))
        self.BT = nn.Parameter(torch.empty(nB))

        if self.constrain == 'exp':
            parametrize.register_parametrization(self, "K", ExpConstrain())
            parametrize.register_parametrization(self, "BT", ExpConstrain())
            parametrize.register_parametrization(self, "K", ClampConstrain(0, 1e3))
            parametrize.register_parametrization(self, "BT", ClampConstrain(0, 1e3))
        elif self.constrain == 'positive':
            parametrize.register_parametrization(self, "K", ClampConstrain(0, 1e3))
            parametrize.register_parametrization(self, "BT", ClampConstrain(0, 1e3))
        elif self.constrain == 'none':
            pass
        elif self.constrain == 'mix':
            parametrize.register_parametrization(self, "K", MixConstrain())
            parametrize.register_parametrization(self, "BT", MixConstrain())
            parametrize.register_parametrization(self, "K", ClampConstrain(0, 1e3))
            parametrize.register_parametrization(self, "BT", ClampConstrain(0, 1e3))
        else:
            raise ValueError
        
        # self.K = torch.randn_like(self.K) # N(0,1)
        self.K = torch.rand_like(self.K) # U(0,1)
        self.BT = torch.full_like(self.BT, 1)

        #self.reset_parameters()


    # @property
    # def K(self):
    #     return self.constrain_func(self._K)

    # @property
    # def BT(self):
    #     return self.constrain_func(self._BT)

    # def reset_parameters(self):
    #     self.K.data = torch.rand_like(self.K)
    #     self.BT.data = torch.full_like(self.BT, 1)


    def forward(
        self,
        AT: Tensor,
        BT: Tensor = None,
        K: Tensor = None,
        ) -> Tuple[Tensor, Tensor, Tensor]:
        
        # 可以输入BT和K,也可以不输入
        if BT is None:
            BT = self.BT.repeat(len(AT), 1)
        if K is None:
            K = self.K

        # print(AT.shape, BT.shape)
        AT = AT.unsqueeze(dim=2)
        BT = BT.unsqueeze(dim=1)
        # print(AT.shape, BT.shape)

        # 不固定A和B的浓度,用迭代自动求导的方法求梯度
        if self.mode == 'comp':
            # solve the fix point
            AF = solvers.torch_solve(AT, BT, K)
            # iter one more time
            BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
            AF = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)
            # calculate BF
            BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
            C = AF * BF * K


        # 不固定A和B的浓度,用平衡点隐函数的方法求梯度
        if self.mode == 'comp2':
            # solve the fix point
            with torch.no_grad():
                AF = solvers.torch_solve(AT, BT, K)
            # iter one more time
            BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
            AF = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)
            # add reverse grad hook
            jacobian = torch.autograd.functional.jacobian(func=lambda x: solvers.one_iter(AT, BT, K, x), inputs=AF)[:, :, 0, :, :, 0].diagonal(dim1=0, dim2=2).permute(2,0,1)
            AF.register_hook(lambda grad: torch.linalg.solve(torch.eye(jacobian.shape[-1]) - jacobian.transpose(1,2), grad))
            #AF.register_hook(lambda grad: torch.linalg.inv(torch.eye(dAF.shape[-1]) - dAF.transpose(1,2)).matmul(grad))
            BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
            C = AF * BF * K


        # semicomp : 固定输入A浓度
        if self.mode == 'semiA' or self.mode == 'compA':
            AF = AT
            BF = BT / ((K * AF).sum(axis=1, keepdim=True) + 1)
            C = AF * BF * K


        # semicomp : 固定输入B浓度，退化线性
        if self.mode == 'semiB' or self.mode == 'compB':
            BF = BT
            AF = AT / ((K * BF).sum(axis=2, keepdim=True) + 1)
            C = AF * BF * K


        # notcomp : 固定输入AB浓度，退化线性
        if self.mode == 'noncomp':
            BF = BT
            AF = AT
            C = AF * BF * K


        return AF, BF, C







class ConstrainedLinear(nn.Module):
    '''
    带有约束的线性层
    '''
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        constrain: str = '',
        ) -> None:

        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.constrain = constrain
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        if constrain == 'exp':
            parametrize.register_parametrization(self, "weight", ExpConstrain())
            parametrize.register_parametrization(self, "bias", ExpConstrain())
            parametrize.register_parametrization(self, "weight", ClampConstrain(0, 10))
            parametrize.register_parametrization(self, "bias", ClampConstrain(0, 10))
        elif constrain == 'positive':
            parametrize.register_parametrization(self, "weight", ClampConstrain(0, 10))
            parametrize.register_parametrization(self, "bias", ClampConstrain(0, 10))
        elif constrain == 'none':
            # parametrize.register_parametrization(self, "weight", NoConstrain())
            # parametrize.register_parametrization(self, "bias", NoConstrain())
            parametrize.register_parametrization(self, "weight", ClampConstrain(-10, 10))
            parametrize.register_parametrization(self, "bias", ClampConstrain(-10, 10))
        else:
            raise ValueError

        self.weight = torch.randn_like(self.weight) # N(0,1)
        # self.weight = torch.rand_like(self.weight) # U(0,1)
        self.bias = torch.full_like(self.bias, 0)
        


        # self.reset_parameters()

    # def reset_parameters(self) -> None:
    #     # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    #     # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    #     # https://github.com/pytorch/pytorch/issues/57109
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #         init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        out = nn.functional.linear(input, self.weight, self.bias)
        if self.constrain == 'positive':
            # log+sigmoid == hill
            out = torch.log(torch.clamp(out, min=1e-6))
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )





class HillLayer(nn.Module):
    '''
    Hill function transfer (0, inf) to (0, 1)
    '''
    def __init__(self, coef=1) -> None:
        super().__init__()
        self.coef = coef
    
    def forward(self, x):
        x = torch.pow(x, self.coef)
        x = x / (1 + x)
        return x



class CompetitiveNetwork(nn.Module):
    def __init__(
        self, 
        nA: int = 2,
        nB: int = 2,
        nY: int = 1,
        mode: str = 'comp',
        output: str = 'C',
        comp_constrain: str = 'exp',
        linear_constrain: str = 'none',
        last_layer: str = 'none',
        *args,
        **kwargs
        ) -> None:
        super().__init__()

        if args or kwargs:
            print('Extra parameters')
            print(kwargs)

        self.nA = nA
        self.nB = nB
        self.nY = nY
        self.mode = mode
        self.output = output
        # self.constrain = constrain
        # self.last_layer = last_layer

        self.comp_layer = CompetitiveLayer(nA, nB, mode=mode, constrain=comp_constrain)

        if (output == 'A'):
            in_features = nA
        elif (output == 'B'):
            in_features = nB
        elif (output == 'C'):
            in_features = nA*nB
        elif (output == 'AB'):
            in_features = nA + nB
        elif (output == 'AC'):
            in_features = nA + nA*nB
        elif (output == 'BC'):
            in_features = nB + nA*nB
        elif (output == 'ABC'):
            in_features = nA + nB + nA*nB
        elif (output == 'firstB'):
            in_features = 1
        elif (output == 'directB'):
            in_features = 1
        elif (output == 'C11'):
            in_features = 1
        else:
            raise ValueError(f'Unknown output: {output}')
        
        self.linear_layer = ConstrainedLinear(in_features, nY, bias=True, constrain=linear_constrain)

        if last_layer == 'none':
            self.last_layer = nn.Identity()
        elif last_layer == 'hill':
            self.last_layer = HillLayer(coef=1)
        elif last_layer == 'sigmoid':
            self.last_layer = nn.Sigmoid() 
        elif last_layer == 'softmax':
            self.last_layer = nn.Softmax(dim=1)
        else:
            raise ValueError(f'Unknown last_layer: {last_layer}')
        
        
        # 没有parametrize的初始化是，nn.init(layer.weight) 或者 layer.weight.data = torch.random
        # 有parametrize的初始化是，nn.init(layer.parametrizations.weight.original) 或者 layer.weight = torch.random

    def forward(
        self,
        AT: Tensor
        ) -> Tensor:

        assert AT.shape[1] == self.nA

        AF, BF, C = self.comp_layer(AT)
        AF, BF, C = AF.reshape(len(AF), -1), BF.reshape(len(BF), -1), C.reshape(len(C), -1)

        if (self.output == 'A'):
            H = AF
        elif (self.output == 'B'):
            H = BF
        elif (self.output == 'C'):
            H = C
        elif (self.output == 'AB'):
            H = torch.cat([AF, BF], dim=1)
        elif (self.output == 'AC'):
            H = torch.cat([AF, C], dim=1)
        elif (self.output == 'BC'):
            H = torch.cat([BF, C], dim=1)
        elif (self.output == 'ABC'):
            H = torch.cat([AF, BF, C], dim=1)
        elif (self.output == 'firstB'):
            H = BF[:, :1]
        elif (self.output == 'directB'):
            Y = - BF
            torch.squeeze(Y, dim=1)
            return Y
        elif (self.output == 'C11'):
            H = C[:, [0]]

        Y = self.linear_layer(H)
        Y = torch.squeeze(Y, dim=1)
        Y = self.last_layer(Y)
        
        return Y



    def get_hidden(
        self,
        AT: Tensor
        ) -> Tensor:

        AF, BF, C = self.comp_layer(AT)
        AF, BF, C = AF.reshape(len(AF), -1), BF.reshape(len(BF), -1), C.reshape(len(C), -1)
        H = torch.cat([AF, BF, C], dim=1)
        return H




    # def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
    #     self_state_dict = self.state_dict()
    #     nA, nB_1 = self_state_dict['comp_layer.parametrizations.K.original'].shape
    #     nA, nB_2 = state_dict['comp_layer.parametrizations.K.original'].shape

    #     if nB_1 != nB_2:
    #         print(f'loading {nB_2} parameters with into {nB_1} model')
    #         nB_min = min(nB_1, nB_2)
    #         self_state_dict['comp_layer.parametrizations.K.original'][:, :nB_min] = state_dict['comp_layer.parametrizations.K.original'][:, :nB_min]
    #         self_state_dict['comp_layer.parametrizations.BT.original'][:, :nB_min] = state_dict['comp_layer.parametrizations.BT.original'][:, :nB_min]
    #         if self.output == 'C':
    #             weight = self_state_dict['linear_layer.weight'].reshape(-1, nA, nB_1)
    #             weight[:, :, :nB_min] = state_dict['linear_layer.weight'].reshape(-1, nA, nB_2)[:, :, :nB_min]
    #             self_state_dict['linear_layer.weight'] = weight.reshape(-1, nA*nB_1)
    #         # if self.output == 'BF':
    #         #     self_state_dict['linear_layer.weight'][:, :nB_min] = state_dict['linear_layer.weight'][:, :nB_min]
    #         self_state_dict['linear_layer.bias'] = state_dict['linear_layer.bias']

    #         # 新的K设为很小
    #         self_state_dict['comp_layer.parametrizations.K.original'][:, nB_min:] = -3
    #         super().load_state_dict(self_state_dict)
    #     else:
    #         super().load_state_dict(state_dict)





# class CompetitiveNetwork_structure(nn.Module):
#     def __init__(
#         self, 
#         nA: int = 2,
#         nB: int = 2,
#         nY: int = 1,
#         mode: str = 'comp',
#         output: str = 'C',
#         constrain: str = 'all',
#         *args,
#         **kwargs
#         ) -> None:

#         super().__init__()
        
#         self.nA = nA
#         self.nB = nB
#         self.nY = nY
#         self.mode = mode
#         self.output = output
#         self.constrain = constrain

#         self.comp_layer_1 = CompetitiveLayer(nA, nB, mode)
#         self.comp_layer_2 = CompetitiveLayer(nA, nB, mode)
#         self.comp_layer_2.K = self.comp_layer_1.K

#         if (output == 'AF'):
#             self.linear_layer = nn.Linear(nA, nY)
#         elif (output == 'BF'):
#             self.linear_layer = nn.Linear(nB, nY)
#         elif (output == 'C'):
#             self.linear_layer = nn.Linear(nA*nB, nY)

#         parametrize.register_parametrization(self.comp_layer_1, "K", ExpConstain())
#         parametrize.register_parametrization(self.comp_layer_1, "BT", ExpConstain())
#         parametrize.register_parametrization(self.comp_layer_2, "K", ExpConstain())
#         parametrize.register_parametrization(self.comp_layer_2, "BT", ExpConstain())
#         parametrize.register_parametrization(self.linear_layer, "weight", ExpConstain())
#         parametrize.register_parametrization(self.linear_layer, "bias", ExpConstain())

#         self.comp_layer_1.K = nn.init.uniform_(self.comp_layer_1.K, 0, 1)
#         self.comp_layer_1.BT = nn.init.uniform_(self.comp_layer_1.BT, 0, 1)
#         self.comp_layer_2.K = nn.init.uniform_(self.comp_layer_2.K, 0, 1)
#         self.comp_layer_2.BT = nn.init.uniform_(self.comp_layer_2.BT, 0, 1)
#         self.linear_layer.weight = nn.init.uniform_(self.linear_layer.weight, 0, 1)
#         self.linear_layer.bias = nn.init.uniform_(self.linear_layer.bias, 0, 1)

        
#     def forward(
#         self,
#         AT: Tensor
#         ) -> Tensor:
#         AF, BF, C = self.comp_layer_1(AT)
#         if (self.output == 'AF'):
#             out = AF
#         elif (self.output == 'BF'):
#             out = BF
#         elif (self.output == 'C'):
#             out = C
#         out = out.reshape(len(out), -1)
#         Y1 = self.linear_layer(out)


#         AF, BF, C = self.comp_layer_2(AT)
#         if (self.output == 'AF'):
#             out = AF
#         elif (self.output == 'BF'):
#             out = BF
#         elif (self.output == 'C'):
#             out = C
#         out = out.reshape(len(out), -1)
#         Y2 = self.linear_layer(out)

#         Y = torch.cat([Y1, Y2], dim=1)

#         return Y






# class CompetitiveNetwork_2(nn.Module):
#     def __init__(self, nA=2, nB=2, nC=2, nY=1, mode='comp', constrain='KB', trainable='K', output='C', *args, **kwargs):
#         super().__init__()
#         if args or kwargs:
#             print('input extra parameters', args, kwargs)

#         self.comp_layer1 = CompetitiveLayer(nA, nB, mode, trainable, constrain)
#         self.comp_layer2 = CompetitiveLayer(nA*nB, nC, mode, trainable, constrain)

#         self.output = output
#         self.linear_layer = nn.Linear(nA*nB*nC, nY, bias=True)

        
#     def forward(self, AT):
#         batch_size = len(AT)
#         A, B, AB = self.comp_layer1(AT)
#         AB = AB.reshape(batch_size, -1)
#         AB, C, ABC = self.comp_layer2(AB)
#         ABC = ABC.reshape(batch_size, -1)
#         Y = self.linear_layer(ABC)
#         return Y







# class CompetitiveNetworkWithoutLinear(nn.Module):
#     def __init__(
#         self, 
#         nA: int = 2,
#         nB: int = 2,
#         nY: int = 1,
#         mode: str = 'comp',
#         output: str = 'C',
#         constrain: str = 'all',
#         *args,
#         **kwargs
#         ) -> None:
#         super().__init__()

#         if args or kwargs:
#             print('Extra parameters')

#         assert nY <= nB

#         self.nA = nA
#         self.nB = nB
#         self.nY = nY
#         self.mode = mode
#         self.output = output
#         self.constrain = constrain

#         self.comp_layer = CompetitiveLayer(nA, nB, mode)

#         parametrize.register_parametrization(self.comp_layer, "K", ExpConstain())
#         parametrize.register_parametrization(self.comp_layer, "BT", ExpConstain())

#         nn.init.uniform_(self.comp_layer.parametrizations.K.original, -2.3, 2.3)
#         nn.init.constant_(self.comp_layer.parametrizations.BT.original, 0)
        
#     def forward(
#         self,
#         AT: Tensor
#         ) -> Tensor:

#         AF, BF, C = self.comp_layer(AT)
#         # Y = torch.sum(C, dim=1)[:, :self.nY]
#         Y = BF[:, 0, :self.nY]
#         return Y





if __name__ == '__main__':


    torch.set_printoptions(precision=16)

    torch.random.manual_seed(0)
    linear_layer_1 = nn.Linear(2,2)
    linear_layer_2 = nn.Linear(2,2)

    parametrize.register_parametrization(linear_layer_2, "weight", MixConstrain())
    parametrize.register_parametrization(linear_layer_2, "weight", ClampConstrain(0, 1e3))
    
    a = torch.Tensor([[1, 10], [100, 1e6]])
    linear_layer_1.weight.data = a
    linear_layer_2.weight = a
    print(linear_layer_1.weight)
    print(linear_layer_2.weight)
    print(linear_layer_2.parametrizations.weight.original)

    linear_layer_2.parametrizations.weight.original.data = torch.rand_like(linear_layer_2.weight) *2 -1
    print(linear_layer_2.weight)

    linear_layer_2.weight = torch.rand_like(linear_layer_2.weight) *2
    print(linear_layer_2.weight)


    # torch.set_printoptions(precision=16)

    # torch.random.manual_seed(0)
    # linear_layer_1 = nn.Linear(2,2)
    # linear_layer_2 = nn.Linear(2,2)

    # parametrize.register_parametrization(linear_layer_2, "weight", ExpConstrain())
    # parametrize.register_parametrization(linear_layer_2, "weight", ClampConstrain(1e-3, 1e3))
    
    # a = torch.Tensor([[1, 10], [100, 1e6]])
    # linear_layer_1.weight.data = a
    # linear_layer_2.weight = a
    # print(linear_layer_1.weight)
    # print(linear_layer_2.weight)
    # print(linear_layer_2.parametrizations.weight.original)

    # linear_layer_2.parametrizations.weight.original.data = torch.rand_like(linear_layer_2.weight) *2 -1
    # print(linear_layer_2.weight)

    # linear_layer_2.weight = torch.rand_like(linear_layer_2.weight) *2
    # print(linear_layer_2.weight)







    # with torch.no_grad():
    #     linear_layer.weight[:] = torch.rand_like(linear_layer.weight)
    #     print(linear_layer.weight)

    # parametrize.register_parametrization(linear_layer, "weight", ExpConstain())

    # print(linear_layer.parametrizations.weight.original)
    # print(linear_layer.weight)
    
    # nn.init.uniform_(linear_layer.parametrizations.weight.original, -1, 0)
    # print(linear_layer.parametrizations.weight.original)
    # print(linear_layer.weight)

    # linear_layer.parametrizations.weight.original = nn.init.uniform_(linear_layer.parametrizations.weight.original, -1, 0)
    # print(linear_layer.parametrizations.weight.original)
    # print(linear_layer.weight)

    # linear_layer.weight = nn.init.uniform_(linear_layer.weight, 0, 1)
    # print(linear_layer.parametrizations.weight.original)
    # print(linear_layer.weight)

    # linear_layer.weight = torch.rand_like(linear_layer.weight)
    # print(linear_layer.parametrizations.weight.original)
    # print(linear_layer.weight)

    # with torch.no_grad():
    #     linear_layer.weight[:] = torch.rand_like(linear_layer.weight)
    #     print(linear_layer.parametrizations.weight.original)
    #     print(linear_layer.weight)


    # torch.manual_seed(0)
    # layer = CompetitiveLayer(2, 3, mode='comp', trainable='KB', constrain='KB')
    # print(layer.K)
    # print(layer.BT)

    # torch.manual_seed(0)
    # layer = CompetitiveLayer(2, 4, mode='comp', trainable='KB', constrain='KB')
    # print(layer.K)
    # print(layer.BT)

    # torch.manual_seed(0)
    # layer = CompetitiveLayer(2, 3, mode='comp2', trainable='KB', constrain='KB')
    # print(layer.K)
    # print(layer.BT)


    
    # AT = torch.Tensor([1, 1]).reshape(1,2).requires_grad_(True)
    # BT = torch.Tensor([1, 1, 1]).reshape(1,3).requires_grad_(True)
    # K  = torch.Tensor([1, 2, 3, 4, 5, 6]).reshape(2,3).requires_grad_(True)
    # torch.manual_seed(0)
    # layer = CompetitiveLayer(2, 3, mode='comp', trainable='KB', constrain='KB')
    # #layer.K.data = K

    # AF, BF, C = layer(AT, BT)
    # print(AF, BF, C)

    # j = torch.autograd.functional.jacobian(func=lambda x: layer(x, BT), inputs=AT)[2]
    # print(j)
    # print(j.shape)
    # print('\n')

    # j = torch.autograd.functional.jacobian(func=lambda x: layer(AT, x), inputs=BT)[2]
    # print(j)
    # print(j.shape)
    # print('\n')

    # j = torch.autograd.functional.jacobian(func=lambda x: layer(AT, BT, x), inputs=layer.K)[2]
    # print(j)
    # print(j.shape)
    # print('\n')



    # torch.manual_seed(0)
    # layer = CompetitiveLayer(2, 3, mode='comp2', trainable='KB', constrain='KB')
    # #layer.K.data = K

    # AF, BF, C = layer(AT, BT)
    # print(AF, BF, C)

    # j = torch.autograd.functional.jacobian(func=lambda x: layer(x, BT)[2], inputs=AT)
    # print(j)
    # print(j.shape)
    # print('\n')

    # j = torch.autograd.functional.jacobian(func=lambda x: layer(AT, x), inputs=BT)[2]
    # print(j)
    # print(j.shape)
    # print('\n')

    # j = torch.autograd.functional.jacobian(func=lambda x: layer(AT, BT, x), inputs=layer.K)[2]
    # print(j)
    # print(j.shape)
    # print('\n')






    # model = CompetitiveNetwork(nA, nB, nY, reparam='exp', mode='comp_1')
    # loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


    # t0 = time.perf_counter()
    # for i in range(1000):
    #     Y_pred = model(AT, BT)
    # t1 = time.perf_counter()
    # print('forward time', t1-t0)


    # t0 = time.perf_counter()
    # for i in range(1000):
    #     Y_pred = model(AT, BT)
    #     Y_pred.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    # t1 = time.perf_counter()
    # print('forward and backward time', t1-t0)




    # AT = torch.tensor([1., 1.]).reshape(1, 2)
    # BT = torch.tensor([1., 1., 1.]).reshape(1, 3)
    # K  = torch.tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)
    # Y =  torch.tensor([1.])
    # K.requires_grad = True


    # loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


    # t0 = time.perf_counter()
    # for i in range(1000):
    #     Y_pred = model(AT)
    # t1 = time.perf_counter()
    # print('forward time', t1-t0)
    

    # t0 = time.perf_counter()
    # for i in range(1000):
    #     Y_pred = model(AT)
    #     Y_pred.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    # t1 = time.perf_counter()
    # print('forward and backward time', t1-t0)

