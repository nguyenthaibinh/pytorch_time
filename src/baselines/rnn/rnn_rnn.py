import torch as th
from torch import nn
from baselines.base_model import BaseModel


class RNNEncoder(nn.Module):
    """ Encodes time-series sequence """

    def __init__(self, input_dim, hidden_dim, num_layers, cell='lstm', bidirectional=False):
        """
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        """
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # define RNN layer
        if cell.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional)
        elif cell.lower() == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
        elif cell.lower() == 'rnn':
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
        else:
            raise Exception(f"Invalid rnn_type!! {cell}.")

    def forward(self, x_input):
        """
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        """
        x_input = x_input.view(x_input.shape[0], x_input.shape[1], self.input_dim)
        self.rnn.flatten_parameters()
        enc_output, enc_hidden = self.rnn(x_input)

        return enc_output, enc_hidden

    def init_hidden(self, batch_size):
        """
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        """
        return (th.zeros(self.num_layers, batch_size, self.hid_dim),
                th.zeros(self.num_layers, batch_size, self.hid_dim))

class RNNDecoder(nn.Module):
    """ Decodes hidden state output by encoder """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, cell='lstm', bidirectional=False):
        """
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        """
        super(RNNDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # define RNN layer
        if cell.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=bidirectional)
        elif cell.lower() == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
        elif cell.lower() == 'rnn':
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=bidirectional)
        else:
            raise Exception(f"Invalid rnn_type!! {cell}.")

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_input, encoder_hidden_states):
        """
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence
        """
        self.rnn.flatten_parameters()
        dec_output, dec_hidden = self.rnn(x_input, encoder_hidden_states)
        output = self.linear(dec_output.squeeze(0))

        return output, dec_hidden


class RNNSeq2Seq(BaseModel):
    """ Train LSTM encoder-decoder and make predictions """

    def __init__(self, input_dim, hidden_dim, output_dim, output_len, num_layers, cell, bidirectional, device):
        """
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        """
        super(RNNSeq2Seq, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_len = output_len
        self.device = device

        self.encoder = RNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, cell=cell,
                                  bidirectional=bidirectional)
        self.decoder = RNNDecoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  num_layers=num_layers, cell=cell, bidirectional=bidirectional)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        batch_size, C, N, T = x.size()
        x = x.view(batch_size, C * N, T).contiguous()
        x = x.permute(0, 2, 1).contiguous()

        enc_output, enc_hidden = self.encoder(x)
        dec_input = x[:, -1, :]  # shape: (batch_size, 1, input_size)
        dec_input = dec_input.unsqueeze(1)
        dec_hidden = enc_hidden

        out = th.zeros(batch_size, self.output_len, self.output_dim)

        for t in range(self.output_len):
            try:
                dec_output, dec_hidden = self.decoder(dec_input, dec_hidden)
                out[:, t, :] = dec_output[:, 0, :]
                dec_input = dec_output
            except Exception as e:
                msg = str(e)
                print(msg)
                raise Exception(e)

        out = out.unsqueeze(-1)
        out = out.to(self.device)

        return out

def get_model(args):
    model = RNNSeq2Seq(input_dim=args.in_dim * args.num_nodes, hidden_dim=args.hidden_dim, output_dim=args.num_nodes,
                       output_len=args.out_len, num_layers=args.layers, cell=args.cell,
                       bidirectional=args.bidirectional, device=args.device)

    if args.load_weights:
        print("Loading pretrained weights...")
        model.load_state_dict(th.load(f'{args.weigth_path}'))
    else:
        print("Initializing model weights...")
        model.reset_parameters()

    return model