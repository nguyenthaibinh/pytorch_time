import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNNBaseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_len, num_layers, cell, device):

        super(RNNBaseModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_len = output_len
        self.num_layers = num_layers
        self.device = device

        if cell == "RNN":
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if cell == "LSTM":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if cell == "GRU":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, output_dim * output_len)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)


class RNN_MLP(RNNBaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim, output_len, num_layers, device):

        super(RNN_MLP, self).__init__(input_dim, hidden_dim, output_dim, output_len, num_layers, 'RNN', device)

    def forward(self, x):
        batch_size, C, N, T = x.size()
        x = x.view(batch_size, C * N, T).contiguous()
        x = x.permute(0, 2, 1).contiguous()

        h0 = Variable(torch.zeros(self.num_layers * 1, batch_size, self.hidden_dim))
        h0 = h0.to(self.device)

        self.rnn.flatten_parameters()
        _, (hn, cn) = self.rnn(x, h0)
        h_out = hn[-1]
        h_out = h_out.view(batch_size, self.hidden_dim)

        out = self.fc(h_out)
        out = out.view(batch_size, self.output_len, self.output_dim)
        out = out.unsqueeze(-1)

        return out


class LSTM_MLP(RNNBaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, output_len, num_layers, device):
        super(LSTM_MLP, self).__init__(input_dim, hidden_dim, output_dim, output_len, num_layers, 'LSTM', device)

    def forward(self, x):
        batch_size, C, N, T = x.size()
        x = x.view(batch_size, C * N, T).contiguous()
        x = x.permute(0, 2, 1).contiguous()

        h0 = Variable(torch.zeros(self.num_layers * 1, batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(self.num_layers * 1, batch_size, self.hidden_dim))
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        self.rnn.flatten_parameters()
        _, (hn, cn) = self.rnn(x, (h0, c0))
        h_out = hn[-1]
        h_out = h_out.view(-1, self.hidden_dim)

        out = self.fc(h_out)
        out = out.view(batch_size, self.output_len, self.output_dim)
        out = out.unsqueeze(-1)

        return out

class GRU_MLP(RNNBaseModel):

    def __init__(self, input_dim, hidden_dim, output_dim, output_len, num_layers, device):
        super(GRU_MLP, self).__init__(input_dim, hidden_dim, output_dim, output_len, num_layers, 'GRU', device)

    def forward(self, x):
        batch_size, C, N, T = x.size()
        x = x.view(batch_size, C * N, T).contiguous()
        x = x.permute(0, 2, 1).contiguous()

        h0 = Variable(torch.zeros(self.num_layers * 1, batch_size, self.hidden_dim))
        h0 = h0.to(self.device)

        self.rnn.flatten_parameters()
        _, hn = self.rnn(x, h0)
        h_out = hn[-1]
        h_out = h_out.view(-1, self.hidden_dim)

        out = self.fc(h_out)
        out = out.view(batch_size, self.output_len, self.output_dim)
        out = out.unsqueeze(-1)

        return out


class ResRNN_Cell(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth, use_cuda=False):

        super(ResRNN_Cell, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.resDepth = resDepth
        self.use_cuda = use_cuda

        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        # self.ht2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):

        batchSize = x.size(0)

        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        if self.use_cuda:
            h0 = h0.cuda()
        ht = h0

        lag = x.data.size()[1]

        outputs = []

        for i in range(lag):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)

            if i == 0:
                hstart = hn
            elif i == lag - 2:
                h0 = nn.Tanh()(hn + hstart)
            else:
                if self.resDepth == 1:
                    h0 = nn.Tanh()(hn + h0)
                else:
                    if i % self.resDepth == 0:
                        h0 = nn.Tanh()(hn + ht)
                        ht = hn
                    else:
                        h0 = nn.Tanh()(hn)
            # act_hn = self.act(hn)
            outputs.append(hn)

        output_hiddens = torch.cat(outputs, 0)

        return output_hiddens


# ResRNN模型
class ResRNNModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth, use_cuda=False):

        super(ResRNNModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerNum = 1
        self.resDepth = resDepth
        self.use_cuda = use_cuda

        self.i2h = nn.Linear(self.inputDim, self.hiddenNum, bias=True)
        self.h2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)
        self.h2o = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim, bias=True)
        self.ht2h = nn.Linear(self.hiddenNum, self.hiddenNum, bias=True)

        self.i2h = self.i2h.cuda()
        self.h2h = self.h2h.cuda()
        self.h2o = self.h2o.cuda()
        self.fc = self.fc.cuda()
        self.ht2h = self.ht2h.cuda()

    def forward(self, x):

        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        if self.use_cuda:
            h0 = h0.cuda()
        lag = x.data.size()[1]
        ht = h0
        for i in range(lag):
            hn = self.i2h(x[:, i, :]) + self.h2h(h0)

            if i == 0:
                hstart = hn
            elif i == lag-1:
                h0 = nn.Tanh()(hn+hstart)
            else:
                if self.resDepth == 1:
                    h0 = nn.Tanh()(hn + h0)
                else:
                    if i % self.resDepth == 0:
                        h0 = nn.Tanh()(hn + ht)
                        ht = hn
                    else:
                        h0 = nn.Tanh()(hn)

        hn = hn.view(batchSize, self.hiddenNum)
        fcOutput = self.fc(hn)

        return fcOutput


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class RNN_Attention(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, resDepth, seq_len, merge="concat", use_cuda=True):

        super(RNN_Attention, self).__init__()
        self.att_fc = nn.Linear(hiddenNum, 1)

        self.time_distribut_layer = TimeDistributed(self.att_fc)
        if merge == "mean":
            self.dense = nn.Linear(hiddenNum, outputDim)
        if merge == "concat":
            self.dense = nn.Linear(hiddenNum * seq_len, outputDim)
        self.hiddenNum = hiddenNum
        self.merge = merge
        self.seq_len = seq_len
        self.use_cuda = use_cuda
        self.cell = ResRNN_Cell(inputDim, hiddenNum, outputDim, resDepth, use_cuda=use_cuda)
        if use_cuda:
            self.cell = self.cell.cuda()

    def forward(self, x):

        batchSize = x.size(0)

        rnnOutput = self.cell(x)

        attention_out = self.time_distribut_layer(rnnOutput)
        attention_out = attention_out.view((batchSize, -1))
        attention_out = F.softmax(attention_out)
        attention_out = attention_out.view(-1, batchSize, 1)

        rnnOutput = rnnOutput * attention_out

        if self.merge == "mean":
            sum_hidden = torch.mean(rnnOutput, 1)
            x = sum_hidden.view(-1, self.hiddenNum)
        if self.merge == "concat":
            rnnOutput = rnnOutput.contiguous()
            x = rnnOutput.view(-1, self.hiddenNum * self.seq_len)

        fcOutput = self.dense(x)

        return fcOutput

def get_model(args):
    if args.cell.lower() == 'rnn':
        model = RNN_MLP(input_dim=args.in_dim * args.num_nodes, hidden_dim=args.hidden_dim,
                        output_dim=args.num_nodes, output_len=args.out_len,
                        num_layers=args.layers, device=args.device)
    elif args.cell.lower() == 'lstm':
        model = LSTM_MLP(input_dim=args.in_dim * args.num_nodes, hidden_dim=args.hidden_dim,
                         output_dim=args.num_nodes, output_len=args.out_len,
                         num_layers=args.layers, device=args.device)
    elif args.cell.lower() == 'gru':
        model = GRU_MLP(input_dim=args.in_dim * args.num_nodes, hidden_dim=args.hidden_dim,
                        output_dim=args.num_nodes, output_len=args.out_len,
                        num_layers=args.layers, device=args.device)
    else:
        raise Exception(f"Unavailable RNN Cell type: {args.cell}.")

    if args.load_weights:
        print("Loading pretrained weights...")
        model.load_state_dict(torch.load(f'{args.weigth_path}'))
    else:
        print("Initializing model weights...")
        model.reset_parameters()

    return model