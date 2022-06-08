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



class CompetitiveLayer(nn.Module):
    def __init__(self, nA, nB):
        super(CompetitiveLayer, self).__init__()
        self.nA = nA
        self.nB = nB
        self.K = nn.Parameter(torch.empty(nA, nB))
        self.K.data = torch.Tensor([1,2,3,4,5,6]).reshape(2, 3)
        # nn.init.uniform_(self.K, 0, 1)


    def forward(self, AT, BT):
        return CompetitiveLayerFunction.apply(AT, BT, self.K)


if __name__ == '__main__':
    cl = CompetitiveLayer(2, 3)

    AT = torch.Tensor([1., 1.])
    BT = torch.Tensor([1., 1., 1.])
    K  = torch.Tensor([1., 2., 3., 4., 5., 6.]).reshape(2, 3)

    C = cl(AT, BT)

    w = torch.Tensor([1,0,0,0,0,0]).reshape(2, 3)
    y = (C*w).sum()
    y.backward(retain_graph=True)
    print('dC00/dK')
    print(cl.K.grad)
    




'''
# an example of linear layer

# Inherit from Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
'''