import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from baselines.base_model import BaseModel
from baselines.graph_wavenet.utils import normalize_adj


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = th.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = th.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in, c_out, dropout, support_len=3,order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = th.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNET(BaseModel):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, add_apt_adj=True, apt_init=None,
                 in_dim=2, horizon=12, residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=1, layers=3):
        super(GWNET, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.add_apt_adj = add_apt_adj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and add_apt_adj:
            if apt_init is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(th.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(th.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = th.svd(apt_init)
                initemb1 = th.mm(m[:, :10], th.diag(p[:10] ** 0.5))
                initemb2 = th.mm(th.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels, out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels, out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels, out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels, out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=horizon, kernel_size=(1, 1), bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.add_apt_adj and self.supports is not None:
            adp = F.softmax(F.relu(th.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = th.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = th.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.add_apt_adj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, None

def get_model(args, adj_matrix):
    # adj = th.tensor(adj, dtype=th.float, requires_grad=False)
    # adj = adj.to(args.device)
    if adj_matrix is not None:
        adj_mx = np.tril(adj_matrix) + np.tril(adj_matrix, -1).T
        adj_mx = normalize_adj(adj_mx, args.adj_type)
        supports = [th.tensor(i).to(args.device) for i in adj_mx]

        if args.random_adj:
            adj_init = None
        else:
            adj_init = supports[0]

    if args.apt_only:
        supports = None
        adj_init = None

    # device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, adapt_adj=True, apt_init=None,
    #                  in_dim=2, horizon=12, residual_channels=32, dilation_channels=32, skip_channels=256,
    #                  end_channels=512, kernel_size=2, blocks=4, layers=2
    model = GWNET(in_dim=args.in_dim, horizon=args.horizon, num_nodes=args.num_nodes, device=args.device,
                  supports=supports, gcn_bool=args.gcn_bool, add_apt_adj=args.add_apt_adj, apt_init=adj_init,
                  residual_channels=args.residual_channels, dilation_channels=args.dilation_channels,
                  skip_channels=args.skip_channels, end_channels=args.end_channels, kernel_size=args.t_kernel_size,
                  blocks=args.blocks, layers=args.layers, dropout=args.dropout)
    # model = GWNET(supports=supports, **vars(args))

    if args.load_weights:
        print("Loading pretrained weights...")
        model.load_state_dict(th.load(f'{args.weigth_path}'))
    else:
        print("Initializing baselines weights...")
        model.reset_parameters()
        # model.apply(weights_init)

    return model