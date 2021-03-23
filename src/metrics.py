"""
Always evaluate the baselines with MAE, RMSE, MAPE, RRSE, PNBI, and oPNBI.
Why add mask to MAE and RMSE?
    Filter the 0 that may be caused by error (such as loop sensor)
Why add mask to MAPE and MARE?
    Ignore very small values (e.g., 0.5/0.5=100%)
"""

import numpy as np
import torch as th
from torch.utils.data import DataLoader
from utils import array_concat
from sklearn import metrics
from utils import create_target_mask

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~th.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= th.mean(mask)
    mask = th.where(th.isnan(mask), th.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = th.where(th.isnan(loss), th.zeros_like(loss), loss)
    return th.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return th.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~th.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= th.mean(mask)
    mask = th.where(th.isnan(mask), th.zeros_like(mask), mask)
    loss = th.abs(preds-labels)
    loss = loss * mask
    loss = th.where(th.isnan(loss), th.zeros_like(loss), loss)
    return th.mean(loss)

"""
def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~th.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mean_mask = th.mean(mask)
    mask /= mean_mask
    mask = th.where(th.isnan(mask), th.zeros_like(mask), mask)
    loss = th.abs((preds-labels)/labels)
    loss = loss * mask
    loss = th.where(th.isnan(loss), th.zeros_like(loss), loss)
    return th.mean(loss)
"""
def masked_mape(preds, labels, null_val=np.nan):
    nz = th.where(labels > 0)
    Pz = preds[nz]
    Az = labels[nz]

    return th.mean(th.abs(Az - Pz) / th.abs(Az))

def metric(preds, real):
    mae = masked_mae(preds, real, 0.0).item()
    mape = masked_mape(preds, real, 0.0).item()
    rmse = masked_rmse(preds, real, 0.0).item()
    return mae, mape, rmse

def accuracy(pred_values, true_values, mask=None, metric="mae"):
    if mask is not None:
        mask = th.gt(true_values, mask)
        pred_values = th.masked_select(pred_values, mask)
        true_values = th.masked_select(true_values, mask)

    if metric == "mae":
        acc = th.mean(th.abs(true_values - pred_values))
    elif metric == "mse":
        acc = th.mean((pred_values - true_values) ** 2)
    elif metric == "rmse":
        acc = th.sqrt(th.mean((pred_values - true_values) ** 2))
    elif metric == "rrse":
        acc = th.sqrt(th.sum((pred_values - true_values) ** 2)) / \
              th.sqrt(th.sum((pred_values - true_values.mean()) ** 2))
    elif metric == "mape":
        acc = th.mean(th.abs(th.div((true_values - pred_values), true_values)))
    elif metric == "pnbi":
        indicator = th.gt(pred_values - true_values, 0).float()
        acc = indicator.mean()
    elif metric == "opnbi":
        bias = (true_values + pred_values) / (2 * true)
        acc = bias.mean()
    elif metric == "mare":
        acc = th.div(th.sum(th.abs((true_values - pred_values))), th.sum(true_values))
    elif metric == "smape":
        acc = th.mean(th.abs(true_values - pred_values) / (th.abs(true_values) + th.abs(pred_values)))
    else:
        raise Exception(f"Invalid metric name: {metric}!!!")
    return acc

def compute_cls_accuracy(net, dataset, batch_size=8, device="cuda", use_src_mask=None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    label_list = []
    score_arr = None
    net = net.to(device, dtype=th.float)
    net.eval()
    for i, data_batch in enumerate(data_loader):
        X_cpu, y_cpu = data_batch
        X_gpu = X_cpu.to(device, dtype=th.float)
        """
        if use_src_mask:
            src_mask = create_target_mask(X_cpu, device)
        else:
            src_mask = None
        """
        label_list.extend(y_cpu)
        scores = net(X_gpu)
        scores = th.softmax(scores, dim=1)
        scores = scores.detach().cpu().numpy()
        score_arr = array_concat(score_arr, scores)
    y_true = np.asarray(label_list)
    top1 = topk_acc(y_true, score_arr, k=1)
    top5 = topk_acc(y_true, score_arr, k=5)
    ret_metric = {'top_1': top1, 'top_5': top5}
    # confusion_matrix = metrics.confusion_matrix(y_true, y_pred, normalize='true')
    # additional_info = dict({'feature_vectors': context_arr, 'labels': label_list, 'confusion': confusion_matrix})
    return ret_metric

def binary_cls_accuracy(net, dataset, label_set, batch_size=64, device="cuda", use_src_mask=None):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    label_list = []
    score_arr = []
    net = net.to(device, dtype=th.float)
    net.eval()
    for i, data_batch in enumerate(data_loader):
        X_cpu, y_cpu = data_batch
        X_gpu = X_cpu.to(device, dtype=th.float)
        if use_src_mask:
            src_mask = create_target_mask(X_cpu, device)
        else:
            src_mask = None
        label_list.extend(y_cpu)
        scores = net(X_gpu)
        scores = th.softmax(scores, dim=1)
        scores = scores.detach().cpu().numpy()
        score_arr.extend(scores[:, 1])
    y_true = np.asarray(label_list)
    score_arr = np.asarray(score_arr)
    roc_auc = metrics.roc_auc_score(y_true, score_arr, average="weighted")
    ret_metric = {'roc_auc': roc_auc}
    # confusion_matrix = metrics.confusion_matrix(y_true, y_pred, normalize='true')
    # additional_info = dict({'feature_vectors': context_arr, 'labels': label_list, 'confusion': confusion_matrix})
    return ret_metric

def cls_accuracy(logits, y_true):
    scores = th.softmax(logits, dim=1)
    scores = scores.detach().cpu().numpy()

    top1 = topk_acc(y_true, scores, k=1)
    top5 = topk_acc(y_true, scores, k=5)
    ret_metric = {'top_1': top1, 'top_5': top5}
    return ret_metric

def topk_acc(y_true, scores, k):
    rank = scores.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(y_true)]
    acc = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return acc

def All_Metrics(y_pred, y_true, mask1, mask2):
    assert type(y_pred) == type(y_true)
    mae = accuracy(y_pred, y_true, mask1, metric='mae')
    rmse = accuracy(y_pred, y_true, mask1, metric='rmse')
    mape = accuracy(y_pred, y_true, mask2, metric='mape')

    return mae, rmse, mape

if __name__ == '__main__':
    pred = th.Tensor([1, 2, 3,4])
    true = th.Tensor([2, 1, 4,5])
    print(All_Metrics(pred, true, None, None))

