# Pytorch_Time: Multi-variate time series forecasting using Pytorch.

## Generate training data:
```shell script
python datasets/generate_training_data/generate_bitcoin_data.py \
    --data-dir path_to_the_data_folder \
    --filename name_of_the_data_file \
    --train-ratio 0.7 \
    --val-ratio 0.1 \
    --window 24 --horizon 12
```

## Run the model:
```shell script
python rnn_mlp_train.py --cfg-file cfg/bitcoin/btc_price_20210419_original_24x12_split_7x1x2.yaml --cell gru
```