# torch
import torch as th
import numpy as np
from pathlib import Path


class KineticDataset(th.utils.data.Dataset):
    def __init__(self, data_root, subset='train', debug=False):
        super(KineticDataset, self).__init__()
        data_file_name = f'{subset}_data.npy'
        label_file_name = f'{subset}_label.pkl'
        data_file_path = Path(data_root, data_file_name)
        label_file_path = Path(data_root, label_file_name)

        print("data_file_path:", data_file_path)
        self.data = np.load(data_file_path)[:, :, :, :]     # B, C, T, V
        # self.data = np.transpose(data, (0, 2, 3, 1))   # B, T, V, C
        labels = np.load(label_file_path, allow_pickle=True)[1]
        self.labels = np.asarray(labels)
        self.num_classes = max(self.labels) + 1

        if debug:
            self.data_seq_len = 30
            self.data = self.data[:50, :, :self.data_seq_len, :]
            self.labels = self.labels[:50]
        else:
            self.data_seq_len = self.data.shape[2]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return self.data.shape[0]

    def seq_len(self):
        return self.data_seq_len

    def number_of_classes(self):
        return self.num_classes


class NtuDataset(th.utils.data.Dataset):
    def __init__(self, data_root, classes, subset='train', debug=False):
        super(NtuDataset, self).__init__()
        data_file_name = f'{subset}_data.npy'
        label_file_name = f'{subset}_label.pkl'

        data_file_path = Path(data_root, data_file_name)
        label_file_path = Path(data_root, label_file_name)

        data = np.load(data_file_path)[:, :, :, :, 0]
        labels = np.load(label_file_path, allow_pickle=True)[1]
        labels = np.asarray(labels)
        indices = [i for i in range(len(labels)) if labels[i] in classes]
        self.data = data[indices]
        self.labels = []

        labels = labels[indices]
        label_dict = dict()
        label_idx = 0

        for label in labels:
            if label not in label_dict.keys():
                label_dict[label] = label_idx
                label_idx += 1
            class_inner_id = label_dict[label]
            self.labels.append(class_inner_id)

        self.labels = np.asarray(self.labels)
        self.label_map = label_dict

        if debug:
            self.data = self.data[:100, :, :, :]
            self.labels = self.labels[:100]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return self.data.shape[0]

    def number_of_classes(self):
        return len(self.labels)