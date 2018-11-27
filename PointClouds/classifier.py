import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import h5py
import pdb
from tqdm import tqdm, trange

from sublayer import MultiHeadAttention, PositionwiseFeedForward

batch_size = 64

class SAB(nn.Module):
  def __init__(self, n_head, in_dim, out_dim):
    super(SAB, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Attn = MultiHeadAttention(n_head, out_dim, out_dim, out_dim)
    self.rFF = PositionwiseFeedForward(out_dim, out_dim)
    
  def forward(self, x, relu_2=False):
    output = self.Gamma(x)
    output, _ = self.Attn(output, output, output)
    output = self.rFF(output, relu_2)
    return output
    
class ISAB(nn.Module):
  def __init__(self, n_head, in_dim, I_dim, out_dim):
    super(ISAB, self).__init__()
    self.I = nn.Parameter(torch.zeros(batch_size, I_dim, out_dim), requires_grad=True)
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Attn_1 = MultiHeadAttention(n_head, out_dim, out_dim, out_dim)
    self.rFF_1 = PositionwiseFeedForward(out_dim, out_dim)
    self.Attn_2 = MultiHeadAttention(n_head, out_dim, out_dim, out_dim)
    self.rFF_2 = PositionwiseFeedForward(out_dim, out_dim)
    
  def forward(self, x, relu_2=False):
    x_output = self.Gamma(x)
    output_1, _ = self.Attn_1(self.I, x_output, x_output)
    output_1 = self.rFF_1(output_1, True)
    output_2, _ = self.Attn_2(x_output, output_1, output_1)
    output_2 = self.rFF_2(output_2, relu_2)
    return output_2

class PermEqui1_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x

class PermEqui1_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x

class PermEqui2_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    xm = self.Lambda(xm) 
    x = self.Gamma(x)
    x = x - xm
    return x

class PermEqui2_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    xm = self.Lambda(xm) 
    x = self.Gamma(x)
    x = x - xm
    return x


class SAB_Pooling(nn.Module):
  def __init__(self, n_head, d_dim, x_dim=3):
    super(SAB_Pooling, self).__init__()
    self.SAB_1 = SAB(n_head, x_dim, d_dim)
    self.SAB_2 = SAB(n_head, d_dim, d_dim)
    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(d_dim, d_dim),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5),
       nn.Linear(d_dim, 40),
    )
    print(self)
    
  def forward(self, x):
    encode = self.SAB_1(x, relu_2=True)
    encode = self.SAB_2(encode, relu_2=False)
    max_encode, _ = encode.max(1)
    ro_output = self.ro(max_encode)
    return ro_output
    
class ISAB_Pooling(nn.Module):
  def __init__(self, n_head, d_dim, x_dim=3, I_dim=16):
    super(ISAB_Pooling, self).__init__()
    self.ISAB_1 = ISAB(n_head, x_dim, I_dim, d_dim)
    self.ISAB_2 = ISAB(n_head, d_dim, I_dim, d_dim)
    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(d_dim, d_dim),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5),
       nn.Linear(d_dim, 40),
    )
    print(self)
    
  def forward(self, x):
    encode = self.ISAB_1(x, relu_2=True)
    encode = self.ISAB_2(encode, relu_2=False)
    max_encode, _ = encode.max(1)
    ro_output = self.ro(max_encode)
    return ro_output

class D(nn.Module):

  def __init__(self, d_dim, x_dim=3, pool = 'mean'):
    super(D, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim

    if pool == 'max':
        self.phi = nn.Sequential(
          PermEqui2_max(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )
    elif pool == 'max1':
        self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )
    elif pool == 'mean':
        self.phi = nn.Sequential(
          PermEqui2_mean(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )
    elif pool == 'mean1':
        self.phi = nn.Sequential(
          PermEqui1_mean(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )

    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.ELU(inplace=True),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, 40),
    )
    print(self)

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output = phi_output.mean(1)
    ro_output = self.ro(sum_output)
    return ro_output


class DTanh(nn.Module):

  def __init__(self, d_dim, x_dim=3, pool = 'mean'):
    super(DTanh, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim

    if pool == 'max':
        self.phi = nn.Sequential(
          PermEqui2_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'max1':
        self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'mean':
        self.phi = nn.Sequential(
          PermEqui2_mean(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'mean1':
        self.phi = nn.Sequential(
          PermEqui1_mean(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
        )

    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.Tanh(),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, 40),
    )
    print(self)

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output, _ = phi_output.max(1)
    ro_output = self.ro(sum_output)
    return ro_output


def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm
