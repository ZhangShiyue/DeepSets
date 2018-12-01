import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import h5py
import pdb
from tqdm import tqdm, trange
from SubLayers import MultiHeadAttention, DotAttention


class PermEqui_attn(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui_attn, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.attn = MultiHeadAttention(1, in_dim, out_dim, in_dim)

    def forward(self, x):
        xa, _ = self.attn(x, x, x)
        x = self.Gamma(x - xa)
        return x


class PermEqui_dot_attn(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui_dot_attn, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.attn = DotAttention(1, in_dim, in_dim, in_dim)

    def forward(self, x):
        xa, _ = self.attn(x, x, x)
        x = self.Gamma(x - xa)
        return x


class PermEqui_attn_concat(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui_attn_concat, self).__init__()
        self.Gamma = nn.Linear(in_dim * 2, out_dim)
        self.attn = MultiHeadAttention(1, in_dim, out_dim, in_dim)

    def forward(self, x):
        xa, _ = self.attn(x, x, x)
        x = self.Gamma(torch.cat((x, xa), -1))
        return x


class PermEqui_attn_dot(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui_attn_dot, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.attn = MultiHeadAttention(1, in_dim, out_dim, in_dim)

    def forward(self, x):
        xa, _ = self.attn(x, x, x)
        x = self.Gamma(x * xa)
        return x


class Pool_attn(nn.Module):
    def __init__(self, batch_size, out_dim):
        super(Pool_attn, self).__init__()
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.k = torch.nn.Parameter(torch.Tensor(1, 1, out_dim)).cuda()
        nn.init.zeros_(self.k)
        self.kk = self.k.repeat([batch_size, 1, 1]).cuda()
        self.attn = MultiHeadAttention(1, out_dim, out_dim, out_dim)

    def forward(self, x):
        ax, _ = self.attn(self.kk, x, x)
        ax = ax.view(self.batch_size, self.out_dim).cuda()
        return ax


class PermEqui_attn_norm(nn.Module):
    def __init__(self, batch_size, in_dim, out_dim):
        super(PermEqui_attn_norm, self).__init__()
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = torch.nn.Parameter(torch.Tensor(1, 1, in_dim)).cuda()
        nn.init.zeros_(self.k)
        self.kk = self.k.repeat([batch_size, 1, 1]).cuda()
        self.attn = MultiHeadAttention(1, in_dim, out_dim, in_dim)
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        ax, _ = self.attn(self.kk, x, x)
        ax = ax.view(self.batch_size, 1, self.in_dim).cuda()
        x = self.Gamma(x - ax)
        return x


class PermEqui1_norm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_norm, self).__init__()
        self.layer_norm = nn.LayerNorm(out_dim)
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.Gamma(x)
        x = self.layer_norm(x)
        return x


class PermEqui1_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        x = self.Gamma(x - xm)
        return x


class PermEqui1_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        x = self.Gamma(x - xm)
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


class D(nn.Module):
    def __init__(self, d_dim, x_dim=3, pool='mean'):
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
    def __init__(self, batch_size, d_dim, x_dim=3, pool='mean'):
        super(DTanh, self).__init__()
        self.d_dim = d_dim
        self.x_dim = x_dim
        self.batch_size = batch_size

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
        elif pool == 'attn':
            self.phi = nn.Sequential(
                    PermEqui_attn(self.x_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn(self.d_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn(self.d_dim, self.d_dim),
                    nn.Tanh(),
            )
        elif pool == 'attn_concat':
            self.phi = nn.Sequential(
                    PermEqui_attn_concat(self.x_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn_concat(self.d_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn_concat(self.d_dim, self.d_dim),
                    nn.Tanh(),
            )
        elif pool == 'attn_dot':
            self.phi = nn.Sequential(
                    PermEqui_attn_dot(self.x_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn_dot(self.d_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn_dot(self.d_dim, self.d_dim),
                    nn.Tanh(),
            )
        elif pool == 'dot_attn':
            self.phi = nn.Sequential(
                    PermEqui_dot_attn(self.x_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_dot_attn(self.d_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_dot_attn(self.d_dim, self.d_dim),
                    nn.Tanh(),
            )
        elif pool == 'norm':
            self.phi = nn.Sequential(
                    PermEqui1_norm(self.x_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui1_norm(self.d_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui1_norm(self.d_dim, self.d_dim),
                    nn.Tanh(),
            )
        elif pool == 'hybrid':
            self.phi = nn.Sequential(
                    PermEqui1_max(self.x_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn(self.d_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui1_max(self.d_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn(self.d_dim, self.d_dim),
                    nn.Tanh(),
            )
        elif pool == 'attn_norm':
            self.phi = nn.Sequential(
                    PermEqui_attn_norm(self.batch_size, self.x_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn_norm(self.batch_size, self.d_dim, self.d_dim),
                    nn.Tanh(),
                    PermEqui_attn_norm(self.batch_size, self.d_dim, self.d_dim),
                    nn.Tanh(),
            )
        # self.pma = nn.Sequential(
        #     Pool_attn(self.batch_size, self.d_dim),
        # )

        self.ro = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.d_dim, self.d_dim),
                nn.Tanh(),
                nn.Dropout(p=0.5),
                nn.Linear(self.d_dim, 40),
        )

    def forward(self, x):
        phi_output = self.phi(x)
        # sum_output = self.pma(phi_output)
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
