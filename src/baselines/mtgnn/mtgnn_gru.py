from baselines.mtgnn.layer import graph_constructor, tensor_mixprop, LayerNorm
# from baselines.mtgnn.attention import MultiHeadConvAttention
from evolve_gnn.attention import AttentionConv
import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from baselines.base_model import BaseModel
from torch.nn.parameter import Parameter
import math

t_kernels = [
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3]
]
t_dilations = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2]
]
t_strides = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]
t_paddings = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

class AttentionalConvolution(nn.Module):
    def __init__(self, cin, cout, kernel_set, dilation_factor=2):
        super(AttentionalConvolution, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout/len(self.kernel_set))
        """
        for kern in self.kernel_set:
            self.tconv.append(MultiHeadConvAttention(cin, cout, n_heads=1, t_kernel_size=kern, t_stride=1,
                                                     t_dilation=dilation_factor, time_last=True))
            # self.tconv.append(nn.Conv2d(cin, cout, (1, kern),dilation=(1, dilation_factor)))
        """

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            y = self.tconv[i](input, input, input)
            x.append(y)
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = th.cat(x, dim=1)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, kernel_set=[2, 3, 6, 7], dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            # self.tconv.append(MultiHeadLocalSelfAttention(cin, cout, n_heads=1, t_kernel_size=kern, t_stride=1,
            #                                               t_dilation=dilation_factor, time_last=True))
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern),dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            y = self.tconv[i](input)
            x.append(y)
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = th.cat(x, dim=1)
        return x


class MTGNN_GRU(BaseModel):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, kernel_set, device, predefined_A=None,
                 static_feat=None, dropout=0.3, subgraph_size=20, node_dim=32, dilation_exponential=1,
                 conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12,
                 in_dim=2, out_dim=12, layers=3, prop_alpha=0.05, tanh_alpha=3, layer_norm_affine=True,
                 attention_side='both', adj_type='symmetric'):
        super(MTGNN_GRU, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.subgraph_size = subgraph_size
        self.device = device
        self.adj_type = adj_type
        self.filter_convs = nn.ModuleList()
        self.sa_filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 5),
                                    padding=(0, 2))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanh_alpha,
                                    static_feat=static_feat)
        self.E1 = nn.Conv2d(residual_channels, node_dim, kernel_size=(1, 1), padding=(0, 0), bias=True)
        self.E2 = nn.Conv2d(residual_channels, node_dim, kernel_size=(1, 1), padding=(0, 0), bias=True)
        self.E1_gru = MatrixGRU(node_dim, num_nodes)
        self.E2_gru = MatrixGRU(node_dim, num_nodes)

        self.seq_length = seq_length
        kernel_size = 3
        if dilation_exponential > 1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(AttentionConv(residual_channels, conv_channels, n_groups=4,
                                                       t_kernel_size=kernel_size, t_stride=1, t_padding=0,
                                                       t_dilation=new_dilation, time_last=True, side=attention_side
                                                       ))
                self.gate_convs.append(AttentionConv(residual_channels, conv_channels, n_groups=4,
                                                     t_kernel_size=kernel_size, t_stride=1, t_padding=0,
                                                     t_dilation=new_dilation, time_last=True, side=attention_side
                                                     ))

                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(tensor_mixprop(residual_channels, residual_channels, gcn_depth, dropout,
                                                      prop_alpha))
                    self.gconv2.append(tensor_mixprop(residual_channels, residual_channels, gcn_depth, dropout,
                                                      prop_alpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affine))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affine))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, 1), bias=True)

        self.idx = th.arange(self.num_nodes).to(device)

    def forward(self, input, idx=None):
        # input: B, T, N, C
        B = input.size(0)
        input = input.permute(0, 3, 2, 1)   # B, C, N, T
        seq_len = input.size(3)
        assert seq_len == self.seq_length, f'input sequence length ({seq_len}) not equal to preset sequence length ' \
                                           f'({self.seq_length}).'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field-self.seq_length, 0, 0, 0))

        x = self.start_conv(input)                  # [B, residual_channels, N, T]
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))   # [B, skip_channels, N, 1]
        for i in range(self.layers):
            residual = x

            filter = self.filter_convs[i](x)
            filter = th.tanh(filter)
            gate = self.gate_convs[i](x)            # [B, conv_channels, N, T]
            gate = th.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)               # [B, skip_channels, N, 1]
            skip = s + skip

            if self.gcn_true:
                if self.adj_type == 'symmetric':
                    x_E = self.E1(x)
                    E = self.E1_gru(x_E)
                    M = th.einsum('bcnt,bcwt->bnwt', E, E)
                elif self.adj_type == 'asymmetric':
                    x_E = self.E1(x)
                    E1 = self.E1_gru(x_E)
                    E2 = self.E1_gru(x_E)
                    M = th.einsum('bcnt,bcwt->bnwt', E1, E2)
                else:
                    raise Exception(f'Adjacency matrix type not found: {self.adj_type}!.')
                M = th.relu(th.tanh(M))
                x = self.gconv1[i](x, M) + self.gconv2[i](x, M.transpose(1, 2))  # [B, C, N, T]
            else:
                x = self.residual_convs[i](x)   # [B, C, N, T]

            x = x + residual[:, :, :, -x.size(3):]

            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)      # B, T, N, 1
        return x

def get_model(args, adj_matrix):
    # adj = th.tensor(adj, dtype=th.float, requires_grad=False)
    # adj = adj.to(args.device)
    A = np.tril(adj_matrix) + np.tril(adj_matrix, -1).T
    A = th.tensor(A, dtype=th.float, requires_grad=False)
    A = A.to(args.device)

    model = MTGNN_GRU(gcn_true=args.gcn_true, buildA_true=args.buildA_true, gcn_depth=args.gcn_depth,
                      num_nodes=args.num_nodes, device=args.device, kernel_set=args.kernel_set, predefined_A=A,
                      in_dim=args.in_dim, seq_length=args.lag, attention_side=args.attention_side,
                      adj_type=args.adj_type, layers=args.layers)

    if args.load_weights:
        print("Loading pretrained weights...")
        model.load_state_dict(th.load(f'{args.weigth_path}'))
    else:
        print("Initializing baselines weights...")
        model.reset_parameters()
        # model.apply(weights_init)

    return model

class MatrixGRU(th.nn.Module):
    def __init__(self, in_dim, num_nodes, activation=nn.RReLU()):
        super(MatrixGRU, self).__init__()
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        self.evolve_E = MatrixGRU_Cell(n_rows=in_dim, n_cols=num_nodes)

        self.activation = activation

    def init_hidden_state(self, batch_size):
        return th.zeros(batch_size, self.in_dim, self.num_nodes)

    def reset_param(self, t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # input: [B, C, N, T]
        B, T = input.size(0), input.size(-1)

        state = self.init_hidden_state(B)  # [d, N]
        state = state.to(input.device)
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