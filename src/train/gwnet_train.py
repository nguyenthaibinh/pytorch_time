from pathlib import Path
import torch as th
from baselines.graph_wavenet.graph_wavenet import get_model
from datasets.pems_data import get_pems_data_loader
import torch.nn as nn
from utils import print_model_parameters
from utils import ConfigLoader, get_root_dir, get_hostname, get_this_filepath, get_this_filename, get_basic_parser, \
    process_args
from datasets.data_loader import load_dataset, load_adj
from datasets.pems_data import get_adjacency_matrix
from trainer import Trainer
from icecream import ic

def add_arguments(parser):
    parser.add_argument('--adj-data', type=str,
                        default='/home/nbinh/datasets/traffic_flow/metr_la/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    # parser.add_argument('--adj-type', type=str, default='doubletransition', help='adj type')
    parser.add_argument('--gcn-bool', action='store_true', help='whether to add graph convolution layer')
    parser.add_argument('--apt-only', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--add-apt-adj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--random-adj', action='store_true', help='whether random initialize adaptive adj')
    parser.add_argument('--residual-channels', type=int, default=32)
    parser.add_argument('--dilation-channels', type=int, default=32)
    parser.add_argument('--skip-channels', type=int, default=256)
    parser.add_argument('--end-channels', type=int, default=512)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--kernel-size', type=int, default=3)
    return parser

def main():
    parser = get_basic_parser()
    parser = add_arguments(parser)

    args = parser.parse_args()
    args, conf = process_args(args)

    args.run_file_name = get_this_filename()
    args.run_file_path = get_this_filepath()

    args.kernel_set = [2, 3, 6, 7]

    args.multi_gpus = False

    # load dataset
    dataloader = load_dataset(args.data_dir, args.batch_size, args.val_batch_size, args.test_batch_size,
                              normalizer=args.normalizer)
    ref_adj = None
    ic(args.dataset)
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

    root_dir = get_root_dir()
    args.model_dir = Path(root_dir, 'src/baselines/graph_wavenet')

    model = get_model(args, predefined_A)
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
