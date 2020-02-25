


Below you can find a outline of how to reproduce my solution for the 2019 Data Science Bowl competition.
If you run into any trouble with the setup/code or have any questions please contact me at kfaceapi@gmail.com

## Archive contents
- 3_solution.tgz : contains original code, trained models etc
```
3rd_solution/
├── input/
│   ├── processed/
│   └── data-science-bowl-2019/
├── models/ # 
└── code/
```
- `input/processed/` : will be created by the preprocessing step (by running `python prepare_data.py`)
- `input/data-science-bowl-2019/` : raw data dir of the competition (should contain 'train.csv', 'test.csv',  etc)
- `models/` : contains trained models used to generate a 'submission.csv' file
- `code/` : contains codes for training and prediction

### Requirements 
- Ubuntu 18.04.4 LTS
- Python 3.7.5
- pytorch 1.3
- transformers 2.3.0 (or pytorch-transformers 1.2.0)

You can use the `pip install -r requirements.txt` to install the necessary packages.

#### Hardware
- CPU: 3 x AMD® Ryzen 9 3900X (3 PCs)
- GPU: 5 x NVIDIA RTX2080Ti 11G (2 GPUs in 1 PC)
- RAM: 64G

The above is just my PC spec. In fact, GTX 1080 is enough for training.

## Prepare Data
You can generate `bowl.pt`, `bowl_info.pt` files by running `prepare_data.py`
``` 
.../3rd_solution/code$ python prepare_data.py
```

## Train Model (GPU needed)
You can reproduce models already in the `model/` directory by running the `train.sh` script.
The `train.sh` reproduce 5 models (5-fold)

``` 
.../3rd_solution/code$ bash train.sh
```

## Predict
You can reproduce the `submissions/submission.csv` by running the `predict.py`

``` 
.../3rd_solution/code$ python predict.py
```


## Option - Validate (GPU needed)
You can reproduce the local CV score and coefficients by running the `validate.py`
- VALID KAPPA_SCORE:0.5684061832361282
- coefficients=[0.53060865, 1.66266655, 2.31145611]
