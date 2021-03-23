import torch as th
import numpy as np
"""
def adjust_lr(args, optim, epoch):
    if args.optimizer.lower() == 'sgd' and args.steps:
        lr = args.lr_init * (
            args.lr_decay_rate**np.sum(epoch >= np.array(args.steps)))
        for param_group in optim.param_groups:
            param_group['lr'] = lr
    else:
        lr = args.lr_init
    return lr
"""

def adjust_lr(args, optim, epoch):
    if args.adjust_lr and args.steps:
        lr = args.lr_init * (
            args.lr_decay_ratio**np.sum(epoch >= np.array(args.steps)))
        for param_group in optim.param_groups:
            param_group['lr'] = lr
    else:
        lr = args.lr_init
    return lr

def load_optimizer(args, model):
    if args.optimizer.lower() == 'sgd':
        optim = th.optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9, nesterov=args.nesterov)
    elif args.optimizer.lower() == 'adam':
        optim = th.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)
    return optim

def get_lr(optim):
    for param_group in optim.param_groups:
        return param_group['lr']