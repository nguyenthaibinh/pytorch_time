from baselines.mtgnn.layer import graph_constructor, dilated_inception, mixprop, LayerNorm
import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from baselines.base_model import BaseModel

class MTGNN_NO_INCEPTION(BaseModel):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, kernel_set, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(MTGNN_NO_INCEPTION, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = kernel_set[-1]
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

                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=conv_channels,
                                                   kernel_size=(1, 3), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels, out_channels=conv_channels,
                                                 kernel_size=(1, 3), dilation=new_dilation))
                
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))

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
        input = input.permute(0, 3, 2, 1)   # B, C, N, T
        seq_len = input.size(3)
        assert seq_len == self.seq_length, f'input sequence length ({seq_len}) not equal to preset sequence length ' \
                                           f'({self.seq_length}).'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field-self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)         # adp: [N, N]
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)                  # [B, residual_channels, N, T]
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))   # [B, skip_channels, N, 1]
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)        # [B, conv_channels, N, T]
            filter = th.tanh(filter)
            gate = self.gate_convs[i](x)            # [B, conv_channels, N, T]
            gate = th.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)               # [B, skip_channels, N, 1]
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1, 0))    # [B, C, N, T]
            else:
                x = self.residual_convs[i](x)   # [B, C, N, T]

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

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

    model = MTGNN_NO_INCEPTION(gcn_true=args.gcn_true, buildA_true=args.buildA_true, gcn_depth=args.gcn_depth,
                               num_nodes=args.num_nodes, kernel_set=args.kernel_set, device=args.device, predefined_A=A,
                               in_dim=args.in_dim, seq_length=args.lag)

    if args.load_weights:
        print("Loading pretrained weights...")
        model.load_state_dict(th.load(f'{args.weigth_path}'))
    else:
        print("Initializing baselines weights...")
        model.reset_parameters()
        # model.apply(weights_init)

    return model
