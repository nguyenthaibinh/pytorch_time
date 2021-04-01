from pathlib import Path
import torch as th
from baselines.dcrnn.dcrnn import get_model
from utils import print_model_parameters
from utils import get_root_dir, get_this_filepath, get_this_filename, get_basic_parser, process_args
from datasets.data_loader import load_dataset, load_adj, load_pems_adj
from trainer import Trainer
from icecream import ic
import numpy as np

def add_arguments(parser):
    parser.add_argument('--rnn-units', type=int, default=64, help='Dimensionality of the rnn hidden layers.')
    return parser

def main():
    parser = get_basic_parser()
    parser = add_arguments(parser)

    args = parser.parse_args()
    args, conf = process_args(args)

    args.run_file_name = get_this_filename()
    args.run_file_path = get_this_filepath()

    root_dir = get_root_dir()
    args.model_dir = Path(root_dir, 'src/baselines/dcrnn')

    # load dataset
    dataloader = load_dataset(args.data_dir, args.batch_size, args.val_batch_size, args.test_batch_size,
                              normalizer=args.normalizer)
    if args.dataset in ['metr_la', 'pems_bay']:
        args.adj_file = conf['adj_file']
        A = load_adj(args.adj_file)
    elif args.dataset[:6] in ['pems03', 'pems04', 'pems07', 'pems08']:
        args.adj_file = conf['adj_file']
        A = load_pems_adj(args.adj_file, args.num_nodes)
    else:
        A = None

    """
    if A is not None:
        A = np.tril(A) + np.tril(A, -1).T
        A = th.from_numpy(A)
    """

    train_loader = dataloader['train_loader']
    val_loader = dataloader['val_loader']
    test_loader = dataloader['test_loader']
    scaler = dataloader['scaler']

    model = get_model(args, A)
    args.model_class = model.__class__
    args.model_class_name = model.__class__.__name__
    model = model.to(args.device, dtype=th.float)

    args.num_parameters = print_model_parameters(model, only_num=True)

    ic(args)
    ic(len(train_loader))
    ic(len(val_loader))
    ic(len(test_loader))
    trainer = Trainer(args, model, scaler, scaler, scaler)
    trainer.fit(args, train_loader, val_loader, test_loader, epochs=args.epochs)
    # test(args, model, test_loader, scaler)

    print("args:", args)

if __name__ == "__main__":
    main()
