import numpy as np
import torch
import torch.nn as nn
from . import models


class RegularizationLoss(nn.Module):
    def __init__(self, weight=1, p=2) -> None:
        super().__init__()
        self.weight = weight
        self.p = p

    def forward(self, model):
        reg_loss = 0
        for param in model.parameters():
            norm = torch.linalg.norm(param, self.p)
            reg_loss += torch.power(norm, self.p)
        reg_loss = self.weight * reg_loss
        return reg_loss



class RegularizationLossCompetition(nn.Module):
    def __init__(self, weight=1, p=2) -> None:
        super().__init__()
        self.weight = weight
        self.p = p

    def forward(self, model):
        reg_loss = 0
        if hasattr(model, 'comp_layer'):
            for w in [model.comp_layer.K, model.comp_layer.BT]:
                if self.p == 1:
                    reg_loss += w.abs().sum()
                elif self.p == 2:
                    reg_loss += w.pow(2).sum()
                    
        for w in [model.linear_layer.weight, model.linear_layer.bias]:
            if self.p == 1:
                reg_loss += w.abs().sum()
            elif self.p == 2:
                reg_loss += w.pow(2).sum()
        
        # for param in model.parameters():
        #     norm = torch.linalg.norm(param, self.p)
        #     reg_loss += torch.power(norm, self.p)
        # print(self.weight, reg_loss)
        reg_loss = self.weight * reg_loss
        # print(reg_loss)
        return reg_loss


if __name__ == '__main__':
    param = torch.tensor([1.0 ,2,3,4])
    a = torch.pow(param, 2).sum()
    b = torch.pow(torch.linalg.norm(param, 2), 2)
    c = param.pow(2).sum()
    print(a, b, c)

    model = models.CompetitiveNetwork()
    reg_loss_func = RegularizationLossCompetition(1, 1)
    a = reg_loss_func(model)
    print(a)

    reg_loss_func = RegularizationLossCompetition(1e-1, 1)
    a = reg_loss_func(model)
    print(a)