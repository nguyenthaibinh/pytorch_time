import numpy as np
import csv
from datasets.dataset import Dataset
from pathlib import Path

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

def get_dataloader(args):
    adj_matrix = get_adjacency_matrix(distance_df_filename=args.adj_file, num_of_vertices=args.num_nodes)
    train_file = Path(args.data_dir, 'train.npz')
    train_file = Path(args.data_dir, 'train.npz')
