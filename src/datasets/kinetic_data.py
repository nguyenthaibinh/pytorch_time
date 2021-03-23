import argparse
import os
import numpy as np
import json
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from pathlib import Path
import tools


class Kinetics(Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    # Joint index:
    # {0,  "Nose"}
    # {1,  "Neck"},
    # {2,  "RShoulder"},
    # {3,  "RElbow"},
    # {4,  "RWrist"},
    # {5,  "LShoulder"},
    # {6,  "LElbow"},
    # {7,  "LWrist"},
    # {8,  "RHip"},
    # {9,  "RKnee"},
    # {10, "RAnkle"},
    # {11, "LHip"},
    # {12, "LKnee"},
    # {13, "LAnkle"},
    # {14, "REye"},
    # {15, "LEye"},
    # {16, "REar"},
    # {17, "LEar"},
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self, data_root, subset='train', window_size=-1, random_choose=False,
                 random_move=False, mmap=False, debug=False):
        data_file_name = f'{subset}_data.npy'
        label_file_name = f'{subset}_label.pkl'

        data_file_path = str(Path(data_root, data_file_name))
        label_file_path = str(Path(data_root, label_file_name))

        print("data_file_path:", data_file_path)
        if mmap:
            self.data = np.load(data_file_path, mmap_mode='r')  # B, C, T, V, M
        else:
            self.data = np.load(data_file_path)

        labels = np.load(label_file_path, allow_pickle=True)[1]
        self.labels = np.asarray(labels)
        self.num_classes = max(self.labels) + 1

        self.window_size = window_size
        self.random_choose = random_choose
        self.random_move = random_move

        if debug:
            self.labels = self.labels[0:100]
            self.data = self.data[0:100]

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.labels[index]

        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def __len__(self):
        return self.data.shape[0]
