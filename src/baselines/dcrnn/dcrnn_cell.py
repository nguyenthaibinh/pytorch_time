import numpy as np
import torch
import scipy.sparse as sp
import torch.nn as nn
from icecream import ic

from baselines.dcrnn import utils

class LayerParams:
    def __init__(self, rnn_network, layer_type, device):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.device = device

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=self.device))
            # nn_param = torch.nn.Parameter(torch.empty(*shape))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        # weights = self._rnn_network.__getattr__('{}_weight_{}'.format(self._type, str(shape)))
        return self._params_dict[shape]
        # return weights

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=self.device))
            # biases = torch.nn.Parameter(torch.empty(length))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        # biases = self._rnn_network.__getattr__('{}_biases_{}'.format(self._type, str(length)))
        return self._biases_dict[length]
        # return biases

class DCGRUCell(torch.nn.Module):
    def __init__(self, in_dim, num_units, adj_mx, max_diffusion_step, num_nodes,
                 nonlinearity='tanh', filter_type="laplacian", use_gc_for_ru=True, device="cuda"):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other non-linearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        # self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self.device = device
        supports = []
        _supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))

        for i, support in enumerate(supports):
            # L = self._build_sparse_matrix(support)
            # L = L.to(self.device)
            # self._supports.append(L)
            L = torch.from_numpy(support)
            # L = L.to(self.device)
            # L = torch.tensor(support, dtype=torch.float32, requires_grad=False)
            _supports.append(L)
            self.register_buffer(f'_support_{i}', L)

        self.support_len = len(supports)

        # self.register_buffer('_supports', _supports)

        self._fc_params = LayerParams(self, 'fc', device)
        self._gconv_params = LayerParams(self, 'gconv', device)

        self.num_matrices = self.support_len * self._max_diffusion_step + 1
        input_size = in_dim + num_units
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size * self.num_matrices, 2 * num_units)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(2 * num_units,)))

        self.weight_2 = nn.Parameter(torch.FloatTensor(size=(input_size * self.num_matrices, num_units)))
        self.biases_2 = nn.Parameter(torch.FloatTensor(size=(num_units,)))

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape)
        return L

    def forward(self, inputs, hx):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)
        :param hx: (B, num_nodes * rnn_units)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * rnn_units)`.
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv_2(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for i in range(self.support_len):
                # for support in self._supports:
                # x1 = torch.sparse.mm(support, x0)
                # x1 = torch.bmm(support, x0)
                support = self.__getattr__(f'_support_{i}')
                x1 = torch.matmul(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = self.support_len * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        # weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        # x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, self.weight)

        # biases = self._gconv_params.get_biases(output_size, bias_start)
        # x += biases
        x += self.biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)

        x = torch.reshape(x, [batch_size, self._num_nodes * output_size])
        return x

    def _gconv_2(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for i in range(self.support_len):
                # for support in self._supports:
                # x1 = torch.sparse.mm(support, x0)
                # x1 = torch.bmm(support, x0)
                support = self.__getattr__(f'_support_{i}')
                x1 = torch.matmul(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = self.support_len * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        # weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        # x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, self.weight_2)

        # biases = self._gconv_params.get_biases(output_size, bias_start)
        # x += biases
        x += self.biases_2
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)

        x = torch.reshape(x, [batch_size, self._num_nodes * output_size])
        return x