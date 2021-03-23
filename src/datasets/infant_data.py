from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

class DeakinInfant(Dataset):
    def __init__(self, data_root, label_file, channels, joints, length, file_name='raw_skel.npy',
                 pos_label=1, normalize=False, data_size=-1):
        super(DeakinInfant, self).__init__()
        self.data_paths = []
        self.infant_list = []
        self.label_list = []
        label_dict = load_labels(label_file)
        infant_dir_list = [e for e in Path(data_root).rglob('*') if e.is_dir()]
        for infant_dir in infant_dir_list:
            dir_name = str(infant_dir.stem)
            infant_id = dir_name[:13]
            if infant_id not in label_dict.keys():
                continue
            infant_label = label_dict[infant_id]
            if pos_label == 0:
                infant_label = 1 - infant_label
            data_path = Path(data_root, dir_name, f'{file_name}')
            if not data_path.exists():
                continue
            self.data_paths.append(data_path)
            self.infant_list.append(dir_name)
            self.label_list.append(infant_label)
        if data_size != -1:
            self.data_paths = self.data_paths[:data_size]
            self.infant_list = self.infant_list[:data_size]
            self.label_list = self.infant_list[:data_size]
        self.channels = channels
        self.joints = joints
        self.length = length
        self.normalize = normalize

    def __getitem__(self, index):
        data_file_path = self.data_paths[index]
        infant_dir = self.infant_list[index]
        label = self.label_list[index]
        data = load_skel_data(file_path=data_file_path, channels=self.channels, joints=self.joints,
                              length=self.length, normalize=self.normalize)
        if self.length == -1:
            length = data.shape[1]
        else:
            length = self.length
        data = data[:, :length, :]
        return data, label

    def labels(self):
        return self.label_list

    def __len__(self):
        return len(self.data_paths)

def load_labels(label_file_path):
    label_dict = dict()
    f = open(label_file_path, 'r')
    lines = f.readlines()[1:]
    f.close()

    for line in lines:
        infant_id, label = line.split(',')
        label_dict[infant_id] = int(label)
    return label_dict

def load_skel_data(file_path, channels=[0, 1], joints=list(range(18)), length=-1, normalize=False):
    n_channels = len(channels)

    data_i = np.load(file_path, allow_pickle=True)

    width = len(joints)
    if length == -1:
        length = data_i.shape[1]

    data = np.zeros((n_channels, length, width))

    # print("dataset.py::load_skel_data::data_i.shape:", data_i.shape)
    for j, c in enumerate(channels):
        for k, joint in enumerate(joints):
            try:
                if normalize:
                    m = data_i[c, :length, joint].mean()
                    std = data_i[c, :length, joint].std()
                    ts = data_i[c, :length, joint].reshape((len(data_i[c, :length, joint]), 1))
                    scaler = StandardScaler()
                    scaler = scaler.fit(ts)
                    normalized = scaler.transform(ts)
                    normalized = normalized.squeeze(axis=1)
                    data[j, :, k] = normalized
                else:
                    data[j, :, k] = data_i[c, :length, joint]
            except Exception as e:
                print(e)
                print("file_path:", file_path)
                print("data_shape:", data.shape)
                print("data_i.shape:", data_i.shape)
                raise e
    return data