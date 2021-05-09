import torch.optim as optim
import torch as th
from optim_utils import load_optimizer, adjust_lr, get_lr
import metrics
import mlflow as mf
import mlflow.pytorch
from utils import get_root_dir, get_hostname, get_this_filepath
from pathlib import Path
from timeit import default_timer as timer
import numpy as np
from icecream import ic
from metrics import masked_mae
import copy
from torch import nn
import json

def get_loss_func(loss_func):
    if loss_func == 'masked_mae':
        loss = masked_mae
    elif loss_func == 'mae':
        loss = th.nn.L1Loss()
    elif loss_func == 'mse':
        loss = th.nn.MSELoss()
    else:
        raise Exception(f"Loss function unavailable!! {loss_func}.")
    return loss


class Trainer:
    def __init__(self, args, model, train_scaler, val_scaler, test_scaler, cl=True):
        self.train_scaler = train_scaler
        self.val_scaler = val_scaler
        self.test_scaler = test_scaler
        self.model = model
        self.model.to(args.device)
        self.optimizer = load_optimizer(args, model)
        self.loss = get_loss_func(args.train_loss)
        self.clip = args.clip
        self.seq_out_len = args.out_len
        self.cl = cl
        self.log_dir = args.log_dir
        # self.scaler = scaler
        self.exp_name = args.exp_name
        self.real_value = args.real_value
        self.device = args.device
        self.debug = args.debug
        self.model_class = args.model_class
        ic(train_scaler.scaler_info())
        ic(val_scaler.scaler_info())
        ic(test_scaler.scaler_info())

        # self.model_wrapper = ModelWrapper(model=self.model, scaler=self.scaler, loss=self.loss)
        # self.best_model = copy.deepcopy(self.model)
        if args.multi_gpus:
            gpu_count = th.cuda.device_count()
            args.gpu_count = gpu_count
            if gpu_count > 1:
                print("Let's use", gpu_count, "GPUs!")
                self.model = nn.DataParallel(self.model)
        self.model = self.model.to(args.device, dtype=th.float)

    def fit(self, args, train_loader, val_loader, test_loader, epochs):
        self.model = self.model.to(self.device, dtype=th.float)
        mf.set_tracking_uri(uri=self.log_dir)
        mf.set_experiment(experiment_name=self.exp_name)

        print("tracking_uri:", self.log_dir)
        print("exp_name:    ", self.exp_name)

        with mf.start_run() as mf_run:
            mf.log_param('host', get_hostname())
            mf.log_param('run_file_path', args.run_file_path)
            mf.log_param('cfg_file', args.cfg_file)
            mf.log_param('gcn_true', args.gcn_true)
            try:
                mf.log_param('adjust_lr_true', args.adjust_lr)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('dynamic_graph', args.dynamic_graph)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('static_graph', args.static_graph)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('dilation_exponential', args.dilation_exponential)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('global_graph', args.global_graph)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('predefined_graph', args.predefined_graph)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('temporal_model', args.temporal_model)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('attention_side', args.attention_side)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('init_state', args.init_state)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('lr_decay_ratio', args.lr_decay_ratio)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('lr_decay_steps', args.steps)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('graph_regularization', args.graph_regularization)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('cell', args.cell)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('adj_type', args.cell)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('multi_gpus', args.multi_gpus)
            except Exception as e:
                ic(e)
            try:
                mf.log_param('clip', args.clip)
            except Exception as e:
                ic(e)

            mf.log_param('kernel_set', args.kernel_set)
            mf.log_param('normalizer', args.normalizer)
            mf.log_param('train_loss', args.train_loss)
            mf.log_param('layers', args.layers)
            mf.log_param("in_dim", args.in_dim)
            mf.log_param("out_dim", args.out_dim)
            mf.log_param("trainer", get_this_filepath())
            mf.log_param('model', self.model_class)
            mf.log_param('window', args.window)
            mf.log_param('horizon', args.horizon)
            mf.log_param('out_len', args.out_len)
            mf.log_param("batch_size", args.batch_size)
            mf.log_param('clip', args.clip)
            mf.log_param('lr_init', args.lr_init)
            mf.log_param('weight_decay', args.weight_decay)
            mf.log_param("param", args)
            mf.log_param("trainer", get_this_filepath())

            mf.log_artifact(args.run_file_path)
            mf.log_artifact(args.cfg_file_path)
            mf.log_artifact(get_this_filepath())
            root_dir = get_root_dir()
            src_dir = Path(root_dir, f'src')
            mlflow.log_artifacts(src_dir, artifact_path="src")

            min_val_mae = np.inf
            not_improved_count = 0
            best_state_dict = None

            num_batches = len(train_loader)
            batches_seen = num_batches * epochs

            for epoch in range(1, epochs + 1):
                start = timer()
                epoch_loss = []
                self.model.train()
                lr = adjust_lr(args, self.optimizer, epoch)

                for i, (input, target) in enumerate(train_loader):
                    input = input[..., :args.in_dim]         # B, T, N, C   (scaled data)
                    input = input.permute(0, 3, 2, 1).contiguous()     # B, C, N, T
                    target = target[..., :args.out_dim]     # B, T, N, 1   (raw data)
                    input = input.to(args.device, dtype=th.float)
                    target = target.to(args.device, dtype=th.float)
                    self.optimizer.zero_grad()

                    B, T, _, _ = target.size()
                    labels = target.permute(1, 0, 2, 3).contiguous()
                    labels = labels.view(T, B, -1)

                    preds = self.model(input)               # B, T, N, 1

                    preds = self.train_scaler.inverse_transform(preds)    # scale to raw data to compare with target

                    loss = self.loss(preds, target, 0.0)

                    loss.backward()

                    if self.clip is not None:
                        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()

                    epoch_loss.append(loss.item())
                    if self.debug and i == 4:
                        ic("Run only five mini-batches in debug mode!")
                        break
                end = timer()
                elapsed_time = end - start
                epoch_loss = np.asarray(epoch_loss)
                train_loss = epoch_loss.mean()
                val_loss, _ = evaluate(args, self.model, val_loader, self.val_scaler, self.device, self.real_value)
                val_mae = val_loss['mae']
                val_rmse = val_loss['rmse']
                val_mape = val_loss['mape']

                if val_mae <= 0.0:
                    break

                if min_val_mae > val_mae > 0.0:
                    min_val_mae = val_mae
                    mf.log_metric("best_val_mae", val_loss['mae'])
                    mf.log_metric("best_val_rmse", val_loss['rmse'])
                    mf.log_metric("best_val_mape", val_loss['mape'])
                    # mf.pytorch.log_model(self.model, f"best_model")

                    ret_metrics, best_pred = evaluate(args, self.model, test_loader, self.test_scaler, self.device,
                                                      self.real_value)
                    for t in range(len(ret_metrics['mae_horizon'])):
                        mf.log_metric(f'best_test_mae_step_{t+1:02}', ret_metrics['mae_horizon'][t])
                        mf.log_metric(f'best_test_rmse_step_{t+1:02}', ret_metrics['rmse_horizon'][t])
                        mf.log_metric(f'best_test_mape_step_{t+1:02}', ret_metrics['mape_horizon'][t])

                    for h in list(range(1, len(ret_metrics['mae_horizon']) + 1)):
                        mf.log_metric(f'best_test_mae_horizon_{h:02}', np.mean(ret_metrics['mae_horizon'][:h]))
                        mf.log_metric(f'best_test_rmse_horizon_{h:02}', np.mean(ret_metrics['rmse_horizon'][:h]))
                        mf.log_metric(f'best_test_mape_horizon_{h:02}', np.mean(ret_metrics['mape_horizon'][:h]))

                    mf.log_metric('best_test_mae', ret_metrics['mae'])
                    mf.log_metric('best_test_rmse', ret_metrics['rmse'])
                    mf.log_metric('best_test_mape', ret_metrics['mape'])

                    mf.log_metric('best_test_mae_epoch', ret_metrics['mae'], step=epoch)
                    mf.log_metric('best_test_rmse_epoch', ret_metrics['rmse'], step=epoch)
                    mf.log_metric('best_test_mape_epoch', ret_metrics['mape'], step=epoch)

                    not_improved_count = 0
                    best_state = True
                else:
                    not_improved_count += 1
                    best_state = False

                if epoch % args.print_every == 0:
                    print(f"Epoch {epoch:03}/{args.epochs:03} | time: {elapsed_time:.1f}s | "
                          f"train mae loss: {train_loss:.4f} | val mae: {val_mae:.4f} | "
                          f"val rmse: {val_rmse:.4f} | val mape: {val_mape:.4f} | lr: {lr:.9f}")

                    mf.log_metric("train_loss", train_loss, epoch)
                    mf.log_metric("val_loss", val_mae, epoch)
                    mf.log_metric("elapsed_time", elapsed_time, epoch)
                    mf.log_metric('lr_epoch', lr, epoch)
                    mf.log_metric('finished_epoch', epoch)

                # save the best state
                if best_state:
                    # self.logger.info('*********************************Current best model saved!')
                    best_state_dict = copy.deepcopy(self.model.state_dict())

                if self.debug and epoch == 5:
                    ic("Run only five epochs in debug mode!")
                    break

            if best_state_dict is not None:
                ic("Log the best model.")
                self.model.load_state_dict(best_state_dict)
                mf.pytorch.log_model(self.model, f"best_model")
                # best_model_uri = "runs:/{}/best_model".format(mf_run.info.run_id)
                # loaded_model = mf.pytorch.load_model(best_model_uri)
                # if isinstance(loaded_model, th.nn.DataParallel):
                #     loaded_model = loaded_model.module
                # loaded_model = loaded_model.to(self.device)
                if args.log_prediction:
                    self.model.eval()
                    best_results, best_preds = evaluate(args, self.model, test_loader, self.test_scaler, self.device,
                                                        self.real_value)
                    ic(mf_run.info.experiment_id)
                    ic(mf_run.info.run_id)

                    pred_file = str(Path(get_root_dir(), f'prediction_logs',
                                         f'{args.model_class_name}_pred_{mf_run.info.run_id}.npz'))

                    np.savez(pred_file, best_pred.cpu())
                    mf.log_artifact(pred_file)

                    acc_file = str(Path(get_root_dir(), f'prediction_logs',
                                        f'{args.model_class_name}_acc_{mf_run.info.run_id}.json'))
                    with open(acc_file, 'w') as f:
                        json.dump(best_results, f)
                    mf.log_artifact(acc_file)
                    try:
                        A = self.model.get_dynamic_adj()
                        if A is not None:
                            ic(A.size())
                            A_file = str(Path(get_root_dir(), f'prediction_logs',
                                              f'{args.model_class_name}_dynamic_A_{mf_run.info.run_id}.npz'))
                            np.savez(A_file, A.detach().cpu().numpy())
                            mf.log_artifact(A_file)
                    except Exception as e:
                        print("Exception:", e)

                    try:
                        A = self.model.get_init_adj()
                        if A is not None:
                            ic(A.size())
                            A_file = str(Path(get_root_dir(), f'prediction_logs',
                                              f'{args.model_class_name}_init_A_{mf_run.info.run_id}.npz'))
                            np.savez(A_file, A.detach().cpu().numpy())
                            mf.log_artifact(A_file)
                    except Exception as e:
                        print("Exception:", e)

                    try:
                        A = self.model.get_static_adj()
                        if A is not None:
                            ic(A.size())
                            A_file = str(Path(get_root_dir(), f'prediction_logs',
                                              f'{args.model_class_name}_static_A_{mf_run.info.run_id}.npz'))
                            np.savez(A_file, A.detach().cpu().numpy())
                            mf.log_artifact(A_file)
                    except Exception as e:
                        print("Exception:", e)
                # self.model =

            """
            model_uri = f"runs:/{mf_run.info.run_id}/best_model"
            best_model = mf.pytorch.load_model(model_uri)
            ret_metrics = evaluate(best_model, test_loader, self.scaler, self.device, self.real_value)
            for t in range(len(ret_metrics['mae_horizon'])):
                mf.log_metric('best_test_mae_horizon', ret_metrics['mae_horizon'][t], step=t + 1)
                mf.log_metric('best_test_rmse_horizon', ret_metrics['rmse_horizon'][t], step=t + 1)
                mf.log_metric('best_test_mape_horizon', ret_metrics['mape_horizon'][t], step=t + 1)

            mf.log_metric('best_test_mae', ret_metrics['mae'])
            mf.log_metric('best_test_rmse', ret_metrics['rmse'])
            mf.log_metric('best_test_mape', ret_metrics['mape'])
            """


def evaluate(args, model, data_loader, scaler, device, real_value=True):
    # model = model.to(device)
    model.eval()
    y_pred = []
    y_true = []
    with th.no_grad():
        for batch_idx, (input, target) in enumerate(data_loader):
            input = input[..., :args.in_dim]            # B, T, N, C   (scaled data)
            input = input.permute(0, 3, 2, 1).contiguous()           # B, C, N, T
            target = target[..., :args.out_dim]         # B, T, N, 1   (raw data)
            input = input.to(device, dtype=th.float)
            target = target.to(device, dtype=th.float)
            preds = model(input)                        # B, T, N, 1    (scaled data)
            y_true.append(target)
            y_pred.append(preds)

            if args.debug and batch_idx == 4:
                ic("Evaluate only five mini-batches in debug mode!!")
                break
    # y_true = scaler.inverse_transform(th.cat(y_true, dim=0))
    # if real_value:
    #    y_pred = th.cat(y_pred, dim=0)
    # else:
    y_true = th.cat(y_true, dim=0)
    y_pred = scaler.inverse_transform(th.cat(y_pred, dim=0))    # scaled to raw data to compare with y_true

    ret_metrics = dict()
    ret_metrics['mae_horizon'] = []
    ret_metrics['rmse_horizon'] = []
    ret_metrics['mape_horizon'] = []
    for t in range(y_true.size(1)):
        mae = metrics.masked_mae(y_pred[:, t, ...], y_true[:, t, ...], 0.0)
        rmse = metrics.masked_rmse(y_pred[:, t, ...], y_true[:, t, ...], 0.0)
        mape = metrics.masked_mape(y_pred[:, t, ...], y_true[:, t, ...], 0.0)
        ret_metrics['mae_horizon'].append(mae.item())
        ret_metrics['rmse_horizon'].append(rmse.item())
        ret_metrics['mape_horizon'].append(mape.item())

    mae = metrics.masked_mae(y_pred, y_true, 0.0)
    rmse = metrics.masked_rmse(y_pred, y_true, 0.0)
    mape = metrics.masked_mape(y_pred, y_true, 0.0)
    ret_metrics['mae'] = mae.item()
    ret_metrics['rmse'] = rmse.item()
    ret_metrics['mape'] = mape.item()
    return ret_metrics, y_pred
