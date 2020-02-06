#!/bin/bash
set -e

python train.py --batch_size 32 --nepochs 10 --lr 1e-04 --dropout 0.2 --encoder TRANSFORMER --aug 0.5 --seed 7 --data_seed 7 --seq_len 100 --k 0
python train.py --batch_size 32 --nepochs 10 --lr 1e-04 --dropout 0.2 --encoder TRANSFORMER --aug 0.5 --seed 7 --data_seed 7 --seq_len 100 --k 1
python train.py --batch_size 32 --nepochs 10 --lr 1e-04 --dropout 0.2 --encoder TRANSFORMER --aug 0.5 --seed 7 --data_seed 7 --seq_len 100 --k 2
python train.py --batch_size 32 --nepochs 10 --lr 1e-04 --dropout 0.2 --encoder TRANSFORMER --aug 0.5 --seed 7 --data_seed 7 --seq_len 100 --k 3
python train.py --batch_size 32 --nepochs 10 --lr 1e-04 --dropout 0.2 --encoder TRANSFORMER --aug 0.5 --seed 7 --data_seed 7 --seq_len 100 --k 4
