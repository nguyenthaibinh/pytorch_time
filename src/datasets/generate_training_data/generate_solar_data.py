from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path

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
    file_name = args.raw_file_path
    out_dir = Path(args.output_dir, f'horizon_{args.horizon:02}')
    out_dir.mkdir(parents=True, exist_ok=True)

    fin = open(file_name)
    data = np.loadtxt(fin, delimiter=',')
    train, val, test = split(data, args.train_ratio, args.val_ratio, args.window, args.horizon)

    train_npz_path = Path(out_dir, 'train.npz')
    np.savez(train_npz_path, x=train[0], y=train[1])
    val_npz_path = Path(out_dir, 'val.npz')
    np.savez(val_npz_path, x=val[0], y=val[1])
    test_npz_path = Path(out_dir, 'test.npz')
    np.savez(test_npz_path, x=test[0], y=test[1])

    print(f"dataset: {Path(file_name).stem}")
    print(f'# full data shape:  {data.shape}     (B, T, N, C)')
    print(f'# train/val/set: 6/2/2')
    print(f'# train x.shape:    {train[0].shape},   y.shape:    {train[1].shape}')
    print(f'# val x.shape:      {val[0].shape},     y.shape:    {val[1].shape}')
    print(f'# test x.shape:     {test[0].shape},    y.shape:    {test[1].shape}')
    print(f'\ndata_dir: \"{out_dir}\"')
    print(f'num_nodes: {train[0].shape[2]}')
    print(f'channels: {train[0].shape[3]}')
    print(f'in_dim: 1')
    print(f'out_dim: 1')
    print(f'lag: {train[0].shape[1]}')
    print(f'horizon: {args.horizon}')
    print(f'output_length: {train[1].shape[1]}')

def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    data_dir = '/home/nbinh/datasets/time_series/traffic'
    file_name = 'traffic.txt'
    raw_file_path = str(Path(data_dir, file_name))
    parser.add_argument("--output-dir", type=str, default=data_dir, help="Output directory.")
    parser.add_argument("--raw-file-path", type=str, default=raw_file_path, help="Raw traffic readings.")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    main(args)
