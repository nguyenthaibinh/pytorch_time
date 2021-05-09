from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
from datasets.generate_training_data.generate_pems_data import split_data_by_ratio, add_window_horizon
from datasets.data_utils import normalize_dataset
from utils import get_root_dir
from icecream import ic
from matplotlib.dates import strpdate2num
from datetime import datetime as dt
import pandas as pd

def split(data, train_ratio, valid_ratio, window, horizon):
    N, M = data.shape
    train_set = range(window + horizon - 1, int(train_ratio * N))
    valid_set = range(int(train_ratio * N), int((train_ratio + valid_ratio) * N))
    test_set = range(int((train_ratio + valid_ratio) * N), N)

    train = _batchify(data, train_set, M, window, horizon)
    val = _batchify(data, valid_set, M, window, horizon)
    test = _batchify(data, test_set, M, window, horizon)

    return train, val, test

def _batchify(data, idx_set, M, window, horizon):
    N = len(idx_set)
    X = np.zeros((N, window, M, 1))
    Y = np.zeros((N, 1, M, 1))
    for i in range(N):
        end = idx_set[i] - horizon + 1
        start = end - window
        X[i, :, :, 0] = data[start:end, :]
        Y[i, 0, :, 0] = data[idx_set[i], :]
    return [X, Y]

def generate_train_val_test(args):
    data_file = args.data_file
    out_dir = args.out_dir
    cfg_file = Path(args.cfg_dir, f'{args.dataset_name}.yaml')
    ic(cfg_file)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cfg_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file, header=None, names=['datetime_str', 'price'])
    df['ts'] = df['datetime_str'].apply(lambda x: pd.Timestamp(x))
    # df['date_time'] = df['ts'].apply(lambda x: dt.fromtimestamp(x).strftime('%Y-%m-%d'))
    print(df.head())

    data = df[['price', 'ts']].to_numpy()

    data = np.expand_dims(data, axis=1)

    data, _ = normalize_dataset(data, normalizer=args.normalizer, column_wise=args.column_wise)

    test_ratio = 1 - args.train_ratio - args.val_ratio
    data_train, data_val, data_test = split_data_by_ratio(data, val_ratio=args.val_ratio, test_ratio=test_ratio)

    # add time window
    x_train, y_train = add_window_horizon(data_train, window=args.window, horizon=args.horizon, interval=1)
    x_val, y_val = add_window_horizon(data_val, window=args.window, horizon=args.horizon, interval=1)
    x_test, y_test = add_window_horizon(data_test, window=args.window, horizon=args.horizon, interval=1)

    # scaler = StandardScaler(mean=data.mean(), std=data.std())
    # train_
    train_npz_path = Path(args.out_dir, 'train_no_split.npz')
    np.savez(train_npz_path, x=data_train)
    train_npz_path = Path(out_dir, 'train.npz')
    np.savez(train_npz_path, x=x_train, y=y_train)
    val_npz_path = Path(out_dir, 'val.npz')
    np.savez(val_npz_path, x=x_val, y=y_val)
    test_npz_path = Path(out_dir, 'test.npz')
    np.savez(test_npz_path, x=x_test, y=y_test)

    f = open(cfg_file, 'w')
    print(f"dataset: {args.dataset_name}", file=f)
    print(f'# full data shape:  {data.shape}     (B, T, N, C)', file=f)
    print(f'# train/val/set: {int(args.train_ratio*10)}/{int(args.val_ratio*10)}/{int(test_ratio*10)}', file=f)
    print(f'# train x.shape:    {x_train.shape},   y.shape:    {y_train.shape}', file=f)
    print(f'# val x.shape:      {x_val.shape},     y.shape:    {y_val.shape}', file=f)
    print(f'# test x.shape:     {x_test.shape},    y.shape:    {y_test.shape}', file=f)
    print(f'\ndata_dir: \"{out_dir}\"', file=f)
    print(f'normalized: {args.normalizer}', file=f)
    print(f'column_wise: {str(args.column_wise)}', file=f)
    print(f'mean/std:           {data[:, :, 0].mean():.2f}/{data[:, :, 0].std():.2f}', file=f)
    print(f'train_mean/std:     {x_train[:, :, :, 0].mean():.2f}/{x_train[:, :, :, 0].std():.2f}', file=f)
    print(f'val_mean/std:       {x_val[:, :, :, 0].mean():.2f}/{x_val[:, :, :, 0].std():.2f}', file=f)
    print(f'test_mean/std:      {x_test[:, :, :, 0].mean():.2f}/{x_test[:, :, :, 0].std():.2f}', file=f)
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
    dataset = 'bitcoin'
    filename = 'btc_price_20210419.csv'
    data_root = f'/home/nbinh/datasets/time_series/{dataset}'
    data_file = str(Path(data_root, filename))
    parser.add_argument("--data-file", type=str, default=data_file, help="Raw traffic readings.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--normalizer", type=str, default='None')
    parser.add_argument("--column-wise", type=eval, default=False)
    args = parser.parse_args()

    train_ratio = int(args.train_ratio * 10)
    val_ratio = int(args.val_ratio * 10)
    test_ratio = int(10 - train_ratio - val_ratio)

    split = f'{train_ratio}x{val_ratio}x{test_ratio}'

    if args.normalizer == 'None':
        dataset_name = f'{Path(data_file).stem}_original_{args.window}x{args.horizon}_split_{split}'
    else:
        dataset_name = f'{Path(data_file).stem}_scaled_{args.normalizer}_{args.window}x{args.horizon}_split_{split}'

    out_dir = Path(data_root, dataset_name)
    args.dataset_name = dataset_name
    args.out_dir = out_dir
    args.cfg_dir = Path(get_root_dir(), f'cfg/{dataset}')

    ic(args)
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    main()
