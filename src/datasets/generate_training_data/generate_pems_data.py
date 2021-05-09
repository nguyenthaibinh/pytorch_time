import torch
import numpy as np
import torch.utils.data
from pathlib import Path
from utils import ConfigLoader
from utils import get_root_dir
import argparse
import csv
from icecream import ic

def add_window_horizon(data, window=3, horizon=1, interval=1, single=False):
    """
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :param interval:
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

def generate_train_val_test(args):
    data_file = args.data_file
    adj_file = args.adj_file
    cfg_file = args.cfg_file

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = 1 - train_ratio - val_ratio

    data = np.load(data_file)['data']
    data_train, data_val, data_test = split_data_by_ratio(data, val_ratio=0.2, test_ratio=0.2)

    # add time window
    x_train, y_train = add_window_horizon(data_train, window=args.window, horizon=args.horizon)
    x_val, y_val = add_window_horizon(data_val, window=args.window, horizon=args.horizon)
    x_test, y_test = add_window_horizon(data_test, window=args.window, horizon=args.horizon)

    train_npz_path = Path(args.out_dir, 'train_no_split.npz')
    np.savez(train_npz_path, x=data_train)
    train_npz_path = Path(args.out_dir, 'train.npz')
    np.savez(train_npz_path, x=x_train, y=y_train)
    val_npz_path = Path(args.out_dir, 'val.npz')
    np.savez(val_npz_path, x=x_val, y=y_val)
    test_npz_path = Path(args.out_dir, 'test.npz')
    np.savez(test_npz_path, x=x_test, y=y_test)

    f = open(cfg_file, 'w')
    print(f"dataset: {args.dataset_name}", file=f)
    print(f'# full data shape:  {data.shape}     (B, T, N, C)', file=f)
    print(f'# train/val/set: {int(args.train_ratio * 10)}/{int(args.val_ratio * 10)}/{int(test_ratio * 10)}', file=f)
    print(f'# train x.shape:    {x_train.shape},   y.shape:    {y_train.shape}', file=f)
    print(f'# val x.shape:      {x_val.shape},     y.shape:    {y_val.shape}', file=f)
    print(f'# test x.shape:     {x_test.shape},    y.shape:    {y_test.shape}', file=f)
    print(f'\ndata_dir: \"{args.out_dir}\"', file=f)
    print(f'adj_file: {adj_file}', file=f)
    print(f'normalized: {args.normalizer}', file=f)
    print(f'column_wise: {str(args.column_wise)}', file=f)
    print(f'mean/std:           {data.mean():.4f}/{data.std():.4f}', file=f)
    print(f'train_mean/std:     {x_train.mean():.4f}/{x_train.std():.4f}', file=f)
    print(f'val_mean/std:       {x_val.mean():.4f}/{x_val.std():.4f}', file=f)
    print(f'test_mean/std:      {x_test.mean():.4f}/{x_test.std():.4f}', file=f)
    print(f'num_nodes: {x_train.shape[2]}', file=f)
    print(f'channels: {x_train.shape[3]}', file=f)
    print(f'in_dim: {x_train.shape[-1]}', file=f)
    print(f'out_dim: {y_train.shape[-1]}', file=f)
    print(f'window: {x_train.shape[1]}', file=f)
    print(f'horizon: {args.horizon}', file=f)
    print(f'output_length: {y_train.shape[1]}', file=f)

    f.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='PEMS08')
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--normalizer", type=str, default='None')
    parser.add_argument("--column-wise", type=eval, default=False)
    args = parser.parse_args()

    dataset = args.dataset

    root_dir = f'/home/nbinh/datasets/traffic_flow/{dataset}'

    train_ratio = int(args.train_ratio * 10)
    val_ratio = int(args.val_ratio * 10)
    test_ratio = int(10 - train_ratio - val_ratio)

    split = f'{train_ratio}x{val_ratio}x{test_ratio}'

    if args.normalizer == 'None':
        dataset_name = f'{dataset.lower()}_original_{args.window}x{args.horizon}_split_{split}'
    else:
        dataset_name = f'{dataset.lower()}_scaled_{args.normalizer}_{args.window}x{args.horizon}_split_{split}'

    data_file = Path(root_dir, f'{dataset.lower()}.npz')
    adj_file = Path(root_dir, f'distance.csv')

    args.dataset_name = dataset_name
    args.data_file = data_file
    args.adj_file = adj_file
    args.out_dir = str(Path(root_dir, dataset_name))
    args.cfg_dir = Path(get_root_dir(), f'cfg/{dataset.lower()}')
    args.cfg_file = str(Path(args.cfg_dir, f'{dataset_name}.yaml'))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cfg_dir).mkdir(parents=True, exist_ok=True)

    ic(args)

    generate_train_val_test(args)

if __name__ == '__main__':
    main()