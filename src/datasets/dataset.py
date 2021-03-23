import torch as th
from pathlib import Path
import numpy as np
from torch import FloatTensor

class Dataset(th.utils.data.Dataset):
    def __init__(self, data_dir, subset='train', debug=False):
        data_path = Path(data_dir, f'{subset}.npz')
        data = np.load(data_path)
        self.X = data['x']
        self.Y = data['y']

        if debug:
            self.X = self.X[:100]
            self.Y = self.Y[:100]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    def __len__(self):
        return self.X.shape[0]

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    X, Y = FloatTensor(X), FloatTensor(Y)
    data = th.utils.data.TensorDataset(X, Y)
    dataloader = th.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader
