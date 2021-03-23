from pathlib import Path
import pandas as pd
from datasets.normalization import StandardScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np
from icecream import ic

def get_ref_adj_mx(data_name='metr_la', k=10):
    data_path = Path(f'/home/nbinh/datasets/traffic_flow/{data_name}/neighbor_graphs', f'neighbor_graph_k_{k}.npz')
    g = np.load(data_path)['g']
    ic(g.mean())
    ic(g.shape)
    return g

def main(data_name='metr_la'):
    data_dir = Path('/home/nbinh/datasets/traffic_flow/', data_name)
    out_dir = Path(data_dir, 'neighbor_graphs')
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = Path(data_dir, f'{data_name}.h5')

    df = pd.read_hdf(h5_path)
    num_samples = df.shape[0]
    num_train = round(num_samples * 0.7)
    df = df[:num_train].values
    scaler = StandardScaler(mean=df.mean(), std=df.std())
    train_feat = scaler.transform(df)
    # self._train_feas = torch.Tensor(train_feas).to(device)
    # print(self._train_feas.shape)

    knn_metric = 'cosine'
    for k in [5, 10, 20, 30, 50]:
        out_file = Path(out_dir, f'neighbor_graph_k_{k}.npz')
        g = kneighbors_graph(train_feat.T, k, metric=knn_metric)
        g = np.array(g.todense(), dtype=np.float32)
        np.savez(out_file, g=g)
        ic(out_file)
        ic(g.shape)
        ic(g.sum())
        ic(g.mean())

if __name__ == '__main__':
    data_name = 'pems_bay'
    main(data_name)
    get_ref_adj_mx(data_name)