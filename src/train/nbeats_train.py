from pathlib import Path
import torch as th
from baselines.n_beats.nbeats import get_model
from datasets.pems_data import get_pems_data_loader
import torch.nn as nn
from utils import print_model_parameters
from utils import ConfigLoader, get_root_dir, get_hostname, get_this_filepath, get_this_filename, get_basic_parser, \
    process_args
from datasets.data_loader import load_dataset, load_adj
from datasets.pems_data import load_pems_adj
from trainer import Trainer
from icecream import ic
from datasets.generate_training_data.gen_ref_adj_mx import get_ref_adj_mx

def add_arguments(parser):
    parser.add_argument('--buildA_true', type=eval, default=True, help='Build the graph or not.')
    parser.add_argument('--subgraph-size', type=int, default=20)
    parser.add_argument('--conv-channels', type=int, default=32)
    parser.add_argument('--residual-channels', type=int, default=32)
    parser.add_argument('--skip-channels', type=int, default=64)
    parser.add_argument('--end-channels', type=int, default=128)
    return parser

def main():
    parser = get_basic_parser()
    # parser = add_arguments(parser)

    args = parser.parse_args()
    args, conf = process_args(args)

    args.run_file_name = get_this_filename()
    args.run_file_path = get_this_filepath()

    args.model_dir = Path(get_root_dir(), 'src/baselines/mtgnn')

    args.kernel_set = [2, 3, 6, 7]

    args.multi_gpus = False

    # load dataset
    dataloader = load_dataset(args.data_dir, args.batch_size, args.val_batch_size, args.test_batch_size,
                              normalizer=args.normalizer)
    ref_adj = None
    if args.dataset in ['metr_la', 'pems_bay']:
        args.adj_file = conf['adj_file']
        predefined_A = load_adj(args.adj_file)
        ref_adj = get_ref_adj_mx(args.dataset)
    elif args.dataset in ['pems03', 'pems04', 'pems07', 'pems08']:
        args.adj_file = conf['adj_file']
        predefined_A = get_adjacency_matrix(args.adj_file, args.num_nodes)
    else:
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
