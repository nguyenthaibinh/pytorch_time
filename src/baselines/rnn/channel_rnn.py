import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch as th
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class MatrixGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_nodes, activation=nn.RReLU()):
        super(MatrixGRU, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.evolve_E = MatrixGRU_Cell(n_rows=in_dim, n_cols=num_nodes)

        self.activation = activation

    def init_hidden_state(self, batch_size):
        return th.zeros(batch_size, self.hidden_dim, self.num_nodes)

    def reset_param(self, t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, input, init_E=None):
        # input: [B, C, N, T]
        B, T = input.size(0), input.size(-1)

        if init_E is None:
            state = self.init_hidden_state(B)  # [d, N]
            state = state.to(input.device)
        else:
            state = init_E.unsqueeze(0).repeat(B, 1, 1)

        out_E = []
        for t in range(T):
            state = self.evolve_E(state, input[:, :, :, t])
            out_E.append(state.unsqueeze(-1))

        out_E = th.cat(out_E, dim=-1)      # [B, clips, d, N]
        return out_E

class MatrixGRU_Cell(nn.Module):
    def __init__(self, n_rows, n_cols):
        super(MatrixGRU_Cell, self).__init__()
        self.update = CellGate(n_rows, n_cols, th.nn.Sigmoid())

        self.reset = CellGate(n_rows, n_cols, th.nn.Sigmoid())

        self.h_tilda = CellGate(n_rows, n_cols, th.nn.Tanh())

    def forward(self, prev_Q, prev_Z):
        # B, C, N
        z_topk = prev_Z   # self.choose_topk(prev_Z, mask)

        z_t = self.update(z_topk, prev_Q)
        r_t = self.reset(z_topk, prev_Q)

        h_cap = r_t * prev_Q
        h_cap = self.h_tilda(z_topk, h_cap)

        new_Q = (1 - z_t) * prev_Q + z_t * h_cap

        return new_Q


class CellGate(th.nn.Module):
    def __init__(self, n_rows, n_cols, activation):
        super(CellGate, self).__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(th.Tensor(n_rows, n_rows), requires_grad=True)
        self.reset_param(self.W)

        self.U = Parameter(th.Tensor(n_rows, n_rows), requires_grad=True)
        self.reset_param(self.U)

        self.bias = Parameter(th.zeros(n_rows, n_cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        # x: [B, C, N]
        # hidden: [B, C, N]
        out_x = th.einsum('fc,bcn->bfn', self.W, x)
        out_h = th.einsum('fc,bcn->bfn', self.U, hidden)
        out = self.activation(out_x + out_h + self.bias)

        return out