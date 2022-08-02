import torch
import torch.nn as nn
import os


a = nn.Parameter(torch.tensor([1.0,2]))
#print(a)


b = a*a
print(b)


print(__file__)