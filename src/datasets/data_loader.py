import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
from torch import FloatTensor
from icecream import ic
from datasets.normalization import StandardScaler, MinMax01Scaler, MinMax11Scaler, NScaler
from datasets.data_utils import normalize_dataset

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, train_batch_size, val_batch_size,
                 test_batch_size, normalize=2):
        self.P = window
        self.h = horizon
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

        self._get_dataloader()

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if normalize == 0:
            self.dat = self.rawdat

        if normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)

        # normalized by the maximum value of each row(sensor).
        if normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _get_dataloader(self):
        self.train_loader = data_loader(self.train[0], self.train[1], batch_size=self.train_batch_size, shuffle=True,
                                        drop_last=True)
        self.val_loader = data_loader(self.valid[0], self.valid[1], batch_size=self.val_batch_size, shuffle=False,
                                      drop_last=True)
        self.test_loader = data_loader(self.test[0], self.test[1], batch_size=self.test_batch_size, shuffle=False,
                                       drop_last=True)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0

        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    # cuda = True if torch.cuda.is_available() else False
    # TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # X, Y = TensorFloat(X), TensorFloat(Y)
    X, Y = FloatTensor(X), FloatTensor(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, mean=None, std=None,
                 column_wise=False, normalizer='std', normalize_all=True):
    data = {}
    data_all = []
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        # data_all.append(data['x_' + category])
        # data_all.append(data['y_' + category])

    # data_all = np.stack(data_all, axis=0)

    """
    if normalizer == 'std':
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    elif normalizer == 'minmax01':
        scaler = MinMax01Scaler(mean=data['x_train'][..., 0].min(), std=data['x_train'].max())
    elif normalizer == 'minmax11':
        scaler = MinMax11Scaler(mean=data['x_train'][..., 0].min(), std=data['x_train'].max())
    elif normalizer == 'None':
        scaler = NScaler()
    else:
        raise Exception(f"Normalizer not available!! {normalizer}")
    """
    if mean is None:
        _, scaler = normalize_dataset(data['x_train'][..., 0], normalizer=normalizer, column_wise=column_wise)
    else:
        scaler = StandardScaler(mean=mean, std=std)

    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = data_loader(data['x_train'], data['y_train'], batch_size=batch_size, shuffle=True,
                                       drop_last=False)
    data['val_loader'] = data_loader(data['x_val'], data['y_val'], batch_size=valid_batch_size, shuffle=False,
                                     drop_last=False)
    data['test_loader'] = data_loader(data['x_test'], data['y_test'], batch_size=test_batch_size, shuffle=False,
                                      drop_last=False)
    # data['scaler'] = scaler
    data['scaler'] = scaler
    ic(data['x_train'].shape)
    ic(data['y_train'].shape)
    ic(data['x_val'].shape)
    ic(data['y_val'].shape)
    ic(data['x_test'].shape)
    ic(data['y_test'].shape)
    return data

def load_dataset_1(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, normalizer='std',
                 column_wise=False):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    """
    if normalizer == 'std':
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    elif normalizer == 'minmax01':
        scaler = MinMax01Scaler(mean=data['x_train'][..., 0].min(), std=data['x_train'].max())
    elif normalizer == 'minmax11':
        scaler = MinMax11Scaler(mean=data['x_train'][..., 0].min(), std=data['x_train'].max())
    elif normalizer == 'None':
        scaler = NScaler()
    else:
        raise Exception(f"Normalizer not available!! {normalizer}")
    """
    data['x_train'][..., 0], train_scaler = normalize_dataset(data['x_train'][..., 0], normalizer=normalizer,
                                                              column_wise=column_wise)
    data['x_val'][..., 0], val_scaler = normalize_dataset(data['x_val'][..., 0], normalizer=normalizer,
                                                          column_wise=column_wise)
    data['x_test'][..., 0], test_scaler = normalize_dataset(data['x_test'][..., 0], normalizer=normalizer,
                                                            column_wise=column_wise)

    # Data format
    """
    for category in ['val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    """

    data['train_loader'] = data_loader(data['x_train'], data['y_train'], batch_size=batch_size, shuffle=True,
                                       drop_last=False)
    data['val_loader'] = data_loader(data['x_val'], data['y_val'], batch_size=valid_batch_size, shuffle=False,
                                     drop_last=False)
    data['test_loader'] = data_loader(data['x_test'], data['y_test'], batch_size=test_batch_size, shuffle=False,
                                      drop_last=False)
    # data['scaler'] = scaler
    data['train_scaler'] = train_scaler
    data['val_scaler'] = val_scaler
    data['test_scaler'] = test_scaler
    ic(data['x_train'].shape)
    ic(data['y_train'].shape)
    ic(data['x_val'].shape)
    ic(data['y_val'].shape)
    ic(data['x_test'].shape)
    ic(data['y_test'].shape)
    return data

def load_dataset_single(args):
    Data = DataLoaderS(args.data_file, args.train_ratio, args.val_ratio, args.device, args.horizon, args.window,
                       args.normalize)
    data = dict()

    data['train_loader'] = data_loader(Data.train[0], Data.train[1], batch_size=args.batch_size, shuffle=True,
                                       drop_last=True)
    data['val_loader'] = data_loader(Data.valid[0], Data.valid[1], batch_size=args.val_batch_size, shuffle=False,
                                     drop_last=True)
    data['test_loader'] = data_loader(Data.test[0], Data.test[1], batch_size=args.test_batch_size, shuffle=False,
                                      drop_last=True)
    data['scale'] = Data.scale
    return data

def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = torch.tensor((x - mean) / std, dtype=torch.float)
    return z
