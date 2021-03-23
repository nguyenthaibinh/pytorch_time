import torch
import numpy as np
import torch.utils.data
from pathlib import Path
from utils import ConfigLoader
import csv

def add_window_horizon(data, window=3, horizon=1, interval=1, single=False):
    """
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :param single:
    :return: X is [B, W, ...], Y is [B, H, ...]
    """
    length = len(data)
    end_index = length - horizon * interval - window * interval + 1
    X = []      # windows
    Y = []      # horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index::interval][:window])
            Y.append(data[index::interval][window + horizon - 1:window + horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index::interval][:window])
            Y.append(data[index::interval][window:window + horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def split_data_by_days(data, val_days, test_days, interval=60):
    """
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    """
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    # A = np.reshape(num_of_vertices, num_of_vertices)

    return A

def create_data(args):
    conf = ConfigLoader(config_file=args.cfg_file).config
    data_file = conf['data_file']
    distance_file = conf['adj_file']
    num_nodes = conf['num_nodes']

    data_dir = Path(data_file).parents[0]

    data = np.load(data_file)['data']
    data_train, data_val, data_test = split_data_by_ratio(data, val_ratio=0.2, test_ratio=0.2)

    # add time window
    x_train, y_train = add_window_horizon(data_train, window=12, horizon=12, interval=1)
    x_val, y_val = add_window_horizon(data_val, window=12, horizon=12, interval=1)
    x_test, y_test = add_window_horizon(data_test, window=12, horizon=12, interval=1)

    train_npz_path = Path(data_dir, 'train.npz')
    np.savez(train_npz_path, x=x_train, y=y_train)
    val_npz_path = Path(data_dir, 'val.npz')
    np.savez(val_npz_path, x=x_val, y=y_val)
    test_npz_path = Path(data_dir, 'test.npz')
    np.savez(test_npz_path, x=x_test, y=y_test)

    print(f'# full data shape:  {data.shape}     (B, T, N, C)')
    print(f'# train/val/set: 6/2/2')
    print(f'# train x, y.shape: {x_train.shape}')
    print(f'# val x, y.shape:   {x_val.shape}')
    print(f'# test x, y.shape:  {x_test.shape}')
    print(f'data_dir: \"{data_dir}\"')

def main():
    import argparse
    # MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--cfg-file', default='cfg/pems/pems04.yaml', type=str)
    args = parser.parse_args()
    create_data(args)

if __name__ == '__main__':
    main()