from pathlib import Path
import torch as th
from baselines.var.var_model import VARModel
from utils import print_model_parameters
from utils import ConfigLoader, get_root_dir, get_hostname, get_this_filepath, get_this_filename, get_basic_parser, \
    process_args
from datasets.data_loader import load_var_dataset
import numpy as np
from trainer import Trainer
from icecream import ic
import mlflow as mf
import torch as th
from metrics import masked_mae, masked_rmse, masked_mape

def train(args, data):
    train_data = data['train_data'][..., 0]
    x_test = data['x_test'][..., 0]
    y_test = data['y_test'][..., 0]
    scaler = data['scaler']

    y_pred = []

    model = VARModel(args.window)
    results = model.fit(train_data)

    mf.set_tracking_uri(uri=args.log_dir)
    mf.set_experiment(experiment_name=args.exp_name)

    print("tracking_uri:", args.log_dir)
    print("exp_name:    ", args.exp_name)

    with mf.start_run() as mf_run:
        mf.log_param('host', get_hostname())
        mf.log_param('run_file_path', args.run_file_path)
        mf.log_param('cfg_file', args.cfg_file)

        mf.log_param('normalizer', args.normalizer)
        mf.log_param("in_dim", args.in_dim)
        mf.log_param("out_dim", args.out_dim)
        mf.log_param("trainer", get_this_filepath())
        mf.log_param('model', model.__class__)
        mf.log_param('window', args.window)
        mf.log_param('horizon', args.horizon)
        mf.log_param('out_len', args.out_len)
        mf.log_param("param", args)
        mf.log_param("trainer", get_this_filepath())

        mf.log_artifact(args.run_file_path)
        mf.log_artifact(args.cfg_file_path)
        mf.log_artifact(get_this_filepath())
        root_dir = get_root_dir()
        src_dir = Path(root_dir, f'src')
        mf.log_artifacts(src_dir, artifact_path="src")

        for i in range(len(x_test)):
            x_i = x_test[i, :, :]
            y_hat = results.forecast(x_i, args.out_len)
            y_pred.append(y_hat)

        y_pred = np.asarray(y_pred)
        y_pred = scaler.inverse_transform(y_pred)
        y_test = th.from_numpy(y_test)
        y_pred = th.from_numpy(y_pred)

        for t in range(args.out_len):
            mae_t = masked_mae(y_pred[:, t, :], y_test[:, t, :], 0.0)
            rmse_t = masked_rmse(y_pred[:, t, :], y_test[:, t, :], 0.0)
            mape_t = masked_mape(y_pred[:, t, :], y_test[:, t, :], 0.0)
            mf.log_metric(f'best_test_mae_step_{t + 1:02}', mae_t.item())
            mf.log_metric(f'best_test_rmse_step_{t + 1:02}', rmse_t.item())
            mf.log_metric(f'best_test_mape_step_{t + 1:02}', mape_t.item())

        for h in list(range(1, args.out_len + 1)):
            mae_h = masked_mae(y_pred[:, :h, :], y_test[:, :h, :], 0.0)
            rmse_h = masked_rmse(y_pred[:, :h, :], y_test[:, :h, :], 0.0)
            mape_h = masked_mape(y_pred[:, :h, :], y_test[:, :h, :], 0.0)
            mf.log_metric(f'best_test_mae_horizon_{h:02}', mae_h.item())
            mf.log_metric(f'best_test_rmse_horizon_{h:02}', rmse_h.item())
            mf.log_metric(f'best_test_mape_horizon_{h:02}', mape_h.item())

        mae = masked_mae(y_pred, y_test, 0.0)
        rmse = masked_rmse(y_pred, y_test, 0.0)
        mape = masked_mape(y_pred, y_test, 0.0)
        mf.log_metric('best_test_mae', mae.item())
        mf.log_metric('best_test_rmse', rmse.item())
        mf.log_metric('best_test_mape', mape.item())

        ic(y_pred.shape)

def main():
    parser = get_basic_parser()

    args = parser.parse_args()
    args, conf = process_args(args)

    args.run_file_name = get_this_filename()
    args.run_file_path = get_this_filepath()

    # load dataset
    data = load_var_dataset(args.data_dir, normalizer=args.normalizer, debug=args.debug)

    train(args, data)

    print("args:", args)

if __name__ == "__main__":
    main()
