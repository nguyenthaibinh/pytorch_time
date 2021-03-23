from datasets.pems_data import load_pems_dataset
from datasets.data_utils import get_dataloader
from utils import ConfigLoader

def load_pems04():
    conf = ConfigLoader(config_file='cfg/pems04.yaml').config
    data_file = conf["data_file"]
    adj_file = conf["adj_file"]
    num_of_vertices = conf["num_of_vertices"]

    data, adj_matrix = load_pems_dataset(data_file=data_file, adj_file=adj_file, num_of_vertices=num_of_vertices)
    return data, adj_matrix

def get_pems04_data_loader(args):
    data, adj_matrix = load_pems04()
    train_loader, val_loader, test_loader, scaler = get_dataloader(args, data, normalizer=args.normalizer, tod=args.tod,
                                                                   dow=False, weather=False, single=False)
    return train_loader, val_loader, test_loader, scaler, adj_matrix
