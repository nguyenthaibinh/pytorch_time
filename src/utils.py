import yaml
from pathlib import Path
import traceback
import torch as th
import random
import numpy as np
import socket
from torch import nn
from torch.autograd import Variable
import argparse

def get_root_dir():
    root_dir = Path(__file__).resolve().parents[1]
    return root_dir

def get_this_filename():
    file_path = get_this_filepath()
    filename = str(Path(file_path).resolve().stem)
    return filename

def get_this_filepath():
    stack = traceback.extract_stack()
    file_path = str(stack[-2].filename)
    return file_path

def get_hostname():
    hostname = socket.gethostname()
    return hostname

class ConfigLoader(object):
    def __init__(self, config_file='config.yaml'):
        # Load common configs
        self.root_dir = get_root_dir()
        CONFIG_FILE = Path(self.root_dir, config_file)
        with open(CONFIG_FILE, 'r') as stream:
            self.config = yaml.safe_load(stream)

class GlobalConfig(object):
    debug = False

def init_seed(seed):
    """
    Disable cudnn to maximize reproducibility
    """
    th.cuda.cudnn_enabled = False
    th.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

def generate_square_subsequent_mask(b_size, seq_len, device):
    """
        Create the look-ahead mask for the target sequence
        :param b_size: the batch size
        :param seq_len:   the sequence length
        :param device:  the device ("cuda" or "cpu")
        :return:    the look-ahead mask
        """
    mask = th.ones((b_size, seq_len), dtype=th.bool).unsqueeze(-2)

    np_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype('uint8')
    np_mask = Variable(th.from_numpy(np_mask) == 0)

    mask = mask & np_mask
    mask = mask.to(device)
    return mask

def create_target_mask(trg_seq, device):
    """
    Create the look-ahead mask for the target sequence
    :param trg_seq: the target sequence
    :param device:  the device ("cuda" or "cpu")
    :return:    the look-ahead mask
    """
    b_size = trg_seq.size(0)
    length = trg_seq.size(1)
    trg_mask = th.ones((b_size, length), dtype=th.bool).unsqueeze(-2)

    np_mask = np.triu(np.ones((1, length, length)), k=1).astype('uint8')
    np_mask = Variable(th.from_numpy(np_mask) == 0)

    trg_mask = trg_mask & np_mask
    trg_mask = trg_mask.to(device)
    return trg_mask

def init_weights(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model

def print_model_parameters(model, only_num=True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print(f'Total params num: {num_parameters}')
    print('*****************Finish Parameter****************')
    return num_parameters

def array_concat(arr1, arr2, axis=0):
    if arr1 is None:
        arr1 = arr2
    else:
        arr1 = np.concatenate((arr1, arr2), axis=axis)
    return arr1

def get_basic_parser():
    # parser
    parser = argparse.ArgumentParser(description='arguments')
    # general
    parser.add_argument('--cfg-file', type=str, default='cfg/metr_la/metr_la.yaml')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--exp-name', type=str, default="pems08_prediction")
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--log-prediction', type=eval, default=True)
    # data
    parser.add_argument('--in-dim', type=int)
    parser.add_argument('--out-dim', type=int, default=1)
    parser.add_argument('--normalizer', type=str, default="std")
    parser.add_argument('--column-wise', type=eval, default=False)
    parser.add_argument('--tod', default=False, type=eval)
    parser.add_argument('--eval-on', type=str, default='original', help="Evaluate on normalized data or original data")
    # model
    parser.add_argument('--no-gcn', action='store_true', default=False, help='whether to add graph convolution layer')
    parser.add_argument('--load-static-feature', type=eval, default=False, help='whether to load static feature')
    parser.add_argument('--gcn-depth', type=int, default=2, help='graph convolution depth')
    parser.add_argument('--dropout', type=int, default=0.3)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--n-groups', type=int, default=4)
    parser.add_argument('--t-kernel-size', type=int, default=3)
    parser.add_argument('--kernel-set', type=int, nargs='+', default=[3, 3, 3, 3])
    parser.add_argument('--dilation-exponential', type=int, default=1)
    # train
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--multi-gpus', type=eval, default=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--accumulate-steps', type=int, default=5)
    parser.add_argument('--val-batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--load-weights', action='store_true', default=False)
    parser.add_argument('--train-loss', type=str, default='masked_mae')
    parser.add_argument('--test-mae-thresh', type=float, default=None)
    parser.add_argument('--test-mape-thresh', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr-init', type=float, default=0.001)
    parser.add_argument('--lr-decay-ratio', type=float, default=0.1)
    parser.add_argument('--eps', type=float, default=1.0e-3)
    parser.add_argument('--adjust-lr', type=eval, default=False)
    parser.add_argument('--clip', type=int, default=5)
    parser.add_argument('--nesterov', type=eval, default=True, help='use nesterov or not')
    parser.add_argument('--steps', type=int, default=[20, 50], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--real-value', default=True, type=eval, help='use real value for loss calculation')

    return parser

def process_args(args):
    args.device = th.device("cuda" if th.cuda.is_available() and not args.no_cuda else "cpu")
    init_seed(args.seed)

    conf = ConfigLoader(config_file=args.cfg_file).config
    args.dataset = conf['dataset']
    args.exp_name = f"exp_{args.dataset}"
    args.data_dir = conf['data_dir']
    args.num_nodes = conf['num_nodes']
    if args.in_dim is None:
        args.in_dim = conf['in_dim']
    if args.out_dim is None:
        args.out_dim = conf['out_dim']
    args.window = conf['window']
    args.out_len = conf['output_length']
    args.horizon = conf['horizon']

    args.gcn_true = not args.no_gcn

    root_dir = get_root_dir()
    args.model_dir = Path(root_dir, 'src/evolve_gnn')

    if args.debug:
        args.exp_name = f"debug_{args.dataset}"
        # args.log_dir = str(Path(get_root_dir(), 'logs/kinetics400_action_recognition/test/mlruns'))
    args.log_dir = str(Path(get_root_dir(), 'logs/mlruns'))
    args.cfg_file_path = Path(get_root_dir(), args.cfg_file)

    if args.dataset in ['electricity', 'solar_AL']:
        args.eval_on = 'scaled'
    return args, conf