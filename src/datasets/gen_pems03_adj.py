import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def get_adjacency_matrix(distance_df_filename, num_of_vertices, out_file, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        f_out = open(out_file, 'w')
        f_out.write("from,to,cost")
        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
                    f_out.write(f"\n{id_dict[i]},{id_dict[j]},{distance}")
        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
                    f_out.write(f"\n{i},{j},{distance}")
        f_out.close()
        return A, distaneA

if __name__ == '__main__':
    adj_file = '/home/nbinh/datasets/traffic_flow/PEMS03/raw_cost.csv'
    id_file_name = '/home/nbinh/datasets/traffic_flow/PEMS03/PEMS03.txt'
    output_file = '/home/nbinh/datasets/traffic_flow/PEMS03/distance.csv'
    num_nodes = 358
    get_adjacency_matrix(distance_df_filename=adj_file, num_of_vertices=num_nodes, out_file=output_file,
                         id_filename=id_file_name)