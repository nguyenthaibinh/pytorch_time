from torch import nn
from baselines.tcn.tcn import TemporalConvNet
import torch

class TCN_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, out_len, num_layers, kernel_size, dropout):
        super(TCN_MLP, self).__init__()
        self.out_dim = out_dim
        self.out_len = out_len
        num_channels = [hidden_dim] * num_layers
        self.tcn = TemporalConvNet(in_dim, num_channels, kernel_size, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, out_dim * out_len)
        # self.sig = nn.Sigmoid()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        # B, C, N, T
        batch_size, C, N, T = x.size()
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = x.view(batch_size, C * N, T).contiguous()    # B, C * N, T
        out = self.tcn(x).transpose(1, 2)
        out = out[:, -1, :]
        out = self.fc(out)

        out = out.view(batch_size, self.out_len, self.out_dim)
        out = out.unsqueeze(-1)
        return out, None

def get_model(args):
    model = TCN_MLP(in_dim=args.in_dim * args.num_nodes, hidden_dim=args.hidden_dim, out_dim=args.num_nodes,
                    out_len=args.out_len, num_layers=args.layers, kernel_size=args.kernel_size, dropout=args.dropout)

    if args.load_weights:
        print("Loading pretrained weights...")
        model.load_state_dict(torch.load(f'{args.weigth_path}'))
    else:
        print("Initializing model weights...")
        model.reset_parameters()

    return model