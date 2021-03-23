from pathlib import Path
import numpy as np
import configparser
import csv
from scipy.sparse.linalg import eigs
from utils import ConfigLoader
from datasets.data_utils import get_dataloader

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

def normalize_adj_matrix(A, max_hop, dilation=1):
    # create a symmetric matrix from A
    A = np.tril(A) + np.tril(A, -1).T
    num_nodes = A.shape[0]
    hop_dis = get_hop_distance(A, max_hop=max_hop)
    valid_hop = range(0, max_hop + 1, dilation)
    adjacency = np.zeros((num_nodes, num_nodes))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalized_adjacency = normalize_digraph(adjacency)

    A = np.zeros((1, num_nodes, num_nodes))
    A[0] = normalized_adjacency

    return A

def get_hop_distance(A, max_hop=1):
    num_nodes = A.shape[0]

    # compute hop steps
    hop_dis = np.zeros((num_nodes, num_nodes)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

def load_pems_data(args, load_adj_matrix=True):
    # load the first dimension (traffic flow data) only.
    data = np.load(args.data_file)['data'][:, :, 0]
    if load_adj_matrix:
        adj_matrix = get_adjacency_matrix(distance_df_filename=args.adj_file, num_of_vertices=args.num_nodes)
    else:
        adj_matrix = None

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)

    print('Load %s Dataset shaped: ' % args.data_file, data.shape, data.max(), data.min(), data.mean(), np.median(data))

    return data, adj_matrix

def get_pems_data_loader(args):
    data, adj_matrix = load_pems_data(args)
    if args.debug:
        data = data[:500]
    train_loader, val_loader, test_loader, scaler = get_dataloader(args, data, normalizer=args.normalizer,
                                                                   tod=args.tod, dow=False, weather=False, single=False)
    return train_loader, val_loader, test_loader, scaler, adj_matrix