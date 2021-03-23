from pathlib import Path
import torch as th
from baselines.agcrn.agcrn import get_model
from utils import print_model_parameters
from utils import ConfigLoader, get_root_dir, get_hostname, get_this_filepath, get_this_filename, get_basic_parser, \
    process_args
from datasets.data_loader import load_dataset, load_adj
from datasets.pems_data import get_adjacency_matrix
from trainer import Trainer
from icecream import ic

def add_arguments(parser):
    parser.add_argument('--rnn-units', type=int, default=64, help='Dimensionality of the rnn hidden layers.')
    parser.add_argument('--embed-dim', type=int, default=10, help='Dimensionality of the node embedding.')
    parser.add_argument('--cheb-k', type=int, default=3)
    return parser

def main():
    parser = get_basic_parser()
    parser = add_arguments(parser)

    args = parser.parse_args()
    args, conf = process_args(args)

    args.run_file_name = get_this_filename()
    args.run_file_path = get_this_filepath()

    root_dir = get_root_dir()
    args.model_dir = Path(root_dir, 'src/baselines/agcrn')

    # load dataset
    dataloader = load_dataset(args.data_dir, args.batch_size, args.val_batch_size, args.test_batch_size,
                              normalizer=args.normalizer)
    ref_adj = None
    if args.dataset in ['metr_la', 'pems_bay']:
        args.adj_file = conf['adj_file']
        predefined_A = load_adj(args.adj_file)
    elif args.dataset in ['pems03', 'pems04', 'pems07', 'pems08']:
        args.adj_file = conf['adj_file']
        predefined_A = get_adjacency_matrix(args.adj_file, args.num_nodes)
    else:
        # print("This is neither pems03-08, metr_la nor pems_bay dataset!!!")
        # return
        predefined_A = None

    train_loader = dataloader['train_loader']
    val_loader = dataloader['val_loader']
    test_loader = dataloader['test_loader']
    scaler = dataloader['scaler']

    # predefined_A = load_adj(args.adj_file)

    model = get_model(args)
    args.model_class = model.__class__
    args.model_class_name = model.__class__.__name__
    model = model.to(args.device, dtype=th.float)

    args.num_parameters = print_model_parameters(model, only_num=True)

    ic(args)
    ic(len(train_loader))
    ic(len(val_loader))
    ic(len(test_loader))
    trainer = Trainer(args, model, scaler, scaler, scaler)
    trainer.fit(args, train_loader, val_loader, test_loader, ref_adj=ref_adj, epochs=args.epochs)
    # test(args, model, test_loader, scaler)

    print("args:", args)

if __name__ == "__main__":
    main()
