from pathlib import Path
import torch as th
from baselines.lstnet.lstnet import get_model
from datasets.pems_data import get_pems_data_loader
import torch.nn as nn
from utils import print_model_parameters
from utils import ConfigLoader, get_root_dir, get_hostname, get_this_filepath, get_this_filename, get_basic_parser, \
    process_args
from datasets.data_loader import load_dataset, load_adj
from datasets.pems_data import get_adjacency_matrix
from trainer import Trainer
from icecream import ic
from datasets.generate_training_data.gen_ref_adj_mx import get_ref_adj_mx

def add_arguments(parser):
    parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
    parser.add_argument('--window', type=int, default=7, help='window size')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=0, help='The window size of the highway component')
    parser.add_argument('--hidSkip', type=int, default=5)
    parser.add_argument('--skip', type=float, default=0)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    return parser

def main():
    parser = get_basic_parser()
    parser = add_arguments(parser)

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

    train_loader = dataloader['train_loader']
    val_loader = dataloader['val_loader']
    test_loader = dataloader['test_loader']
    scaler = dataloader['scaler']

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
    trainer.fit(args, train_loader, val_loader, test_loader, epochs=args.epochs)

    print("args:", args)

if __name__ == "__main__":
    main()
