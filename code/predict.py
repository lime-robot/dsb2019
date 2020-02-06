import os
import sys
import torch
import json
import gc
import time
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from numba import jit 
from functools import partial
from scipy import optimize
from torch.utils.data import DataLoader
from pytorch_transformers.modeling_bert import BertConfig, BertEncoder

import warnings
warnings.filterwarnings(action='ignore')


TARGET = ['accuracy_group', 'num_correct', 'num_incorrect']
GAME_TARGET = ['accuracy_group_game', 'num_correct_game', 'num_incorrect_game']
#TARGET = ['accuracy_group']

from torch.utils.data import Dataset


class BowlDataset(Dataset):
    def __init__(self, cfg, df, sample_indices, aug=0.0, aug_p=0.5, padding_front=True, use_tta=False):
        self.cfg = cfg
        self.df = df.copy()    
        self.sample_indices = sample_indices
        self.seq_len = self.cfg.seq_len
        self.aug = aug
        self.aug_p = aug_p
        self.use_tta = use_tta
        self.padding_front = padding_front
         
        self.cate_cols = self.cfg.cate_cols
        self.cont_cols = self.cfg.cont_cols
        
        self.cate_df = self.df[self.cate_cols]
        self.cont_df = np.log1p(self.df[self.cont_cols])                
        if 'accuracy_group' in self.df:
            self.df['num_incorrect'][self.df['num_incorrect']==1] = 0.5
            self.df['num_incorrect'][self.df['num_incorrect']>1] = 1.0            
            self.df['num_correct'][self.df['num_correct']>1] = 1.0
            self.target_df = self.df[TARGET]
        else:
            self.target_df = None
            
        if 'accuracy_group_game' in self.df:
            self.df['num_incorrect_game'][self.df['num_incorrect_game']==1] = 0.5
            self.df['num_incorrect_game'][self.df['num_incorrect_game']>1] = 1.0            
            self.df['num_correct_game'][self.df['num_correct_game']>1] = 1.0
            self.target_game_df = self.df[GAME_TARGET]
        else:
            self.target_game_df = None
        
    def __getitem__(self, idx):
        indices = self.sample_indices[idx]
        
        seq_len = min(self.seq_len, len(indices))
        
        if self.aug > 0:
            if len(indices)>30:
                if np.random.binomial(1, self.aug_p) == 1:
                    cut_ratio = random.random()
                    if cut_ratio > self.aug:
                        cut_ratio = self.aug
                    #cut_ratio = self.aug
                    start_idx = max(int(len(indices)*cut_ratio), 30)
                    indices = indices[start_idx:]
                    seq_len = min(self.seq_len, len(indices))
        
        tmp_cate_x = torch.LongTensor(self.cate_df.iloc[indices].values)
        cate_x = torch.LongTensor(self.seq_len, len(self.cate_cols)).zero_()
        if self.padding_front:
            cate_x[-seq_len:] = tmp_cate_x[-seq_len:]
        else:
            cate_x[:seq_len] = tmp_cate_x[-seq_len:]
        
        tmp_cont_x = torch.FloatTensor(self.cont_df.iloc[indices].values)
        tmp_cont_x[-1] = 0
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        if self.padding_front:            
            cont_x[-seq_len:] = tmp_cont_x[-seq_len:]
        else:
            cont_x[:seq_len] = tmp_cont_x[-seq_len:]
        
        mask = torch.ByteTensor(self.seq_len).zero_()
        if self.padding_front:
            mask[-seq_len:] = 1
        else:
            mask[:seq_len] = 1
        
        if self.target_df is not None:
            target = torch.FloatTensor(self.target_df.iloc[indices[-1]].values)
            if target.sum() == 0:                
                target = torch.FloatTensor(self.target_game_df.iloc[indices[-1]].values)            
        else:
            target = 0
        
        return cate_x, cont_x, mask, target

    def __len__(self):
        return len(self.sample_indices)


class TransfomerModel(nn.Module):
    def __init__(self, cfg):
        super(TransfomerModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertEncoder(self.config)        
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.target_size),            
        )        
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)     
        cont_emb = self.cont_emb(cont_x)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)
        
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        
        encoded_layers = self.encoder(seq_emb, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:, -1]
        
        pred_y = self.reg_layer(sequence_output)
        return pred_y

    
class LSTMATTNModel(nn.Module):
    def __init__(self, cfg):
        super(LSTMATTNModel, self).__init__()
        self.cfg = cfg
        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)        
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size*cate_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )        
        self.cont_emb = nn.Sequential(                
            nn.Linear(cont_col_size, cfg.hidden_size//2),
            nn.LayerNorm(cfg.hidden_size//2),
        )
        
        self.encoder = nn.LSTM(cfg.hidden_size, 
                            cfg.hidden_size, 1, dropout=cfg.dropout, batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=1,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.attn = BertEncoder(self.config)                 
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.target_size),            
        )           
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        
        cate_emb = self.cate_emb(cate_x).view(batch_size, self.cfg.seq_len, -1)
        cate_emb = self.cate_proj(cate_emb) 
        cont_emb = self.cont_emb(cont_x)
        
        seq_emb = torch.cat([cate_emb, cont_emb], 2)        
        
        output, _ = self.encoder(seq_emb)
        
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.config.num_hidden_layers
        
        encoded_layers = self.attn(output, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        sequence_output = sequence_output[:, -1]
        pred_y = self.reg_layer(sequence_output)
        return pred_y

    
ENCODERS = {    
    'TRANSFORMER':TransfomerModel,
    'LSTMATTN':LSTMATTNModel,
}


def replace_4110_4100(df):
    rep_code4110_bool = (df['title']=='Bird Measurer (Assessment)')&(df['event_code']==4110)
    rep_code4100_bool = (df['title']=='Bird Measurer (Assessment)')&(df['event_code']==4100)
    df['event_code'][rep_code4110_bool] = 4100
    df['event_code'][rep_code4100_bool] = 5110


def get_agged_session(df):    
    event_code = pd.crosstab(df['game_session'], df['event_code'])
    event_id = pd.crosstab(df['game_session'], df['event_id'])
    event_num_correct = pd.pivot_table(df[(~df['correct'].isna())], index='game_session', columns='event_code', values='num_correct', aggfunc='sum')
    event_num_incorrect = pd.pivot_table(df[(~df['correct'].isna())], index='game_session', columns='event_code', values='num_incorrect', aggfunc='sum')
    event_accuracy = event_num_correct/(event_num_correct+event_num_incorrect[event_num_correct.columns])
    event_accuracy = event_accuracy.add_prefix('accuray_')    
    del event_num_correct, event_num_incorrect    
    
    event_round = pd.pivot_table(df[~df['correct'].isna()], index='game_session', columns='event_code', values='round', aggfunc='max')
    event_round = event_round.add_prefix('round_')
    
    print('max_game_time')    
    df['elapsed_time'] = df[['game_session', 'game_time']].groupby('game_session')['game_time'].diff()
    game_time = df.groupby('game_session', as_index=False)['elapsed_time'].agg(['mean', 'max']).reset_index()
    game_time.columns = ['game_session', 'mean_game_time', 'max_game_time']    
    df = df.merge(game_time, on='game_session', how='left')    
    event_max_game_time = pd.pivot_table(df, index='game_session', columns='event_code', values='elapsed_time', aggfunc='max')
    event_max_game_time = event_max_game_time.add_prefix('max_game_time_')
    del df['elapsed_time'] 
    
    print('session_extra_df')
    session_extra_df = pd.concat([event_code, event_id, event_accuracy, event_round], 1)
    session_extra_df.index.name = 'game_session'
    session_extra_df.reset_index(inplace=True)
    del event_code, event_id, event_accuracy, event_round
    
    print('session_df')
    session_df = df.drop_duplicates('game_session', keep='last').reset_index(drop=True)
    session_df['row_id'] = session_df.index
    session_df = session_df.merge(session_extra_df, how='left', on='game_session')
    return session_df

def gen_label(df):
    num_corrects = []
    for inst_id, one_df in df.groupby('installation_id'):
        one_df = one_df[(one_df['type']=='Assessment')&(one_df['event_code']==4100)]
        for game_session, title_df in one_df.groupby('game_session'):            
            num_correct = title_df['event_data'].str.contains('"correct":true').sum()
            num_incorrect = title_df['event_data'].str.contains('"correct":false').sum()            
            num_corrects.append([inst_id, game_session, num_correct, num_incorrect])
    label_df = pd.DataFrame(num_corrects, columns=['installation_id', 'game_session', 'num_correct', 'num_incorrect'])
    label_df['accuracy'] = label_df['num_correct'] / (label_df['num_correct']+label_df['num_incorrect'])
    label_df['accuracy_group'] = 3
    label_df['accuracy_group'][label_df['accuracy']==0.5] = 2    
    label_df['accuracy_group'][label_df['accuracy']<0.5] = 1
    label_df['accuracy_group'][label_df['accuracy']==0] = 0    
    return label_df


def extract_data_from_event_code(df, columns=['correct', 'round']):
    for col in columns:
        col_bool = df['event_data'].str.contains(col)
        df[col] = np.nan
        df[col][col_bool] = df['event_data'][col_bool].apply(lambda x: json.loads(x).get(col)).astype(float)

        
def get_train_sample_indices(df):
    sample_indices = []
    inst_indiecs = []    
    df_groups = df.groupby('installation_id').groups
    for inst_idx, indices in enumerate(df_groups.values()):
        one_df = df.iloc[indices].reset_index(drop=True)
        assessment_start_indices = one_df[(one_df['type']=='Assessment')&
                                          (one_df['accuracy_group']>=0)
                                         ].index
        for num, start_index in enumerate(assessment_start_indices):
            sample_indices.append( one_df.iloc[:start_index+1]['row_id'].tolist() )
            inst_indiecs.append(inst_idx)            
    return sample_indices, inst_indiecs

def choose_one(train_samples, train_groups, random_state):    
    random.seed(random_state)    
    group_dict = {}
    for row_id, group in zip(train_samples, train_groups):
        if group not in group_dict:
            group_dict[group] = []
        group_dict[group].append(row_id)
    new_train_samples = []    
    for v in group_dict.values():        
        new_train_samples.append(random.choice(v))         
    
    return np.array(new_train_samples)

def preprocessing(df, train_columns, mappers_dict, cate_offset, cate_cols, cont_cols, extra_cont_cls):
    print('preprocessing ... ')
    replace_4110_4100(df)
    
    print('generating label ...')
    label_df = gen_label(df)
    
    print('extract_data_from_event_code ...')
    extract_data_from_event_code(df)
    df['num_incorrect'] = np.where(df['correct']==0, 1, np.nan)
    df['num_correct'] = np.where(df['correct']==1, 1, np.nan)
    
    df['game_time'] = df['game_time'] // 1000
    
    df = get_agged_session(df)
    df = df.drop(['correct', 'round', 'num_correct', 'num_incorrect'], axis=1)
    
    df = df.merge(label_df, on=['game_session', 'installation_id'], how='left')
    
    samples, groups = get_train_sample_indices(df)
    
    df = df.append(pd.DataFrame(columns=train_columns))[train_columns]
    df = df.fillna(0)
    
    for col in cate_cols:
        df[col] = df[col].map(mappers_dict[col]).fillna(0).astype(int)
    
    print('preprocessing ... done')        
    return df, samples, groups

@jit
def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / (e+1e-08)


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

        return -qwk3(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']

def get_optimized_kappa_score(predictions, groundtruth):
    optR = OptimizedRounder()
    optR.fit(predictions, groundtruth)
    coefficients = optR.coefficients()
    #print(coefficients)
    temp_predictions = predictions.copy()
    temp_predictions[temp_predictions < coefficients[0]] = 0
    temp_predictions[(coefficients[0]<=temp_predictions)&(temp_predictions< coefficients[1])] = 1
    temp_predictions[(coefficients[1]<=temp_predictions)&(temp_predictions< coefficients[2])] = 2
    temp_predictions[(coefficients[2]<=temp_predictions)] = 3

    kappa_score = qwk3(temp_predictions, groundtruth)
    return kappa_score, coefficients 

class CFG:
    learning_rate=1.0e-4
    batch_size=64
    num_workers=4
    print_freq=100
    test_freq=1
    start_epoch=0
    num_train_epochs=1
    warmup_steps=30
    max_grad_norm=1000
    gradient_accumulation_steps=1
    weight_decay=0.01    
    dropout=0.2
    emb_size=100
    hidden_size=500
    nlayers=2
    nheads=8    
    device='cpu'
    #device='cuda:0'
    seed=7
    ntta = [0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6] # TEST KAPPA_SCORE:0.5990772768904306
    wtta = [0.8]
CFG.wtta += [ (1-CFG.wtta[0])/(len(CFG.ntta)-1) for _ in range(len(CFG.ntta)-1)]

    
def main():
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)    
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True       
    
    settings = json.load(open('SETTINGS.json'))
    
    test_df = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'test.csv'))

    [train_columns, mappers_dict, cate_offset, 
     cate_cols, cont_cols, extra_cont_cls] = torch.load(os.path.join(settings['CLEAN_DATA_DIR'], 'bowl_info.pt'))
    test_df, test_samples, test_groups = preprocessing(test_df, train_columns, mappers_dict, cate_offset, 
                            cate_cols, cont_cols, extra_cont_cls)    
    
    CFG.target_size = 3
    CFG.total_cate_size = cate_offset
    print(CFG.__dict__)
    CFG.cate_cols = cate_cols
    CFG.cont_cols = cont_cols+extra_cont_cls    
        
    base_model_path_list = [
        ['bowl.pt', [
            [1.0, os.path.join(settings['MODEL_DIR'], 'b-32_a-TRANSFORMER_e-100_h-500_d-0.2_l-2_hd-10_s-7_len-100_aug-0.5_da-bowl.pt_k-0.pt')],            
        ]],
    ]

    ################################################
    # find the coefficients
    ################################################
    rand_seed_list = [7, 77, 777, 1, 2]
    #rand_seed_list = [110798, 497274, 885651, 673327, 599183, 272713, 582394, 180043, 855725, 932850]    
    sum_coefficients = 0
    sum_cnt = 0
    for _, base_model_paths in base_model_path_list:        
        for model_w, base_model_path in base_model_paths:        
            path = base_model_path.split('/')[-1]
            path = path.replace('bowl_', '')
            cfg_dict = dict([tok.split('-') for tok in path.split('_')])
            CFG.encoder = cfg_dict['a']
            CFG.seq_len = int(cfg_dict['len'])
            CFG.emb_size = int(cfg_dict['e'])
            CFG.hidden_size = int(cfg_dict['h'])
            CFG.nlayers = int(cfg_dict['l'])
            CFG.nheads = int(cfg_dict['hd'])
            CFG.seed = int(cfg_dict['s'])
            CFG.data_seed = int(cfg_dict['s'])
            
            for k in range(5):
                model = ENCODERS[CFG.encoder](CFG)
                model_path = base_model_path.replace('k-0', f'k-{k}')
                
                checkpoint = torch.load(model_path, map_location=CFG.device)        
                model.load_state_dict(checkpoint['state_dict'])
                model.to(CFG.device)
                print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))            
                
                for rand_seed in rand_seed_list:
                    chosen_samples = choose_one(test_samples, test_groups, random_state=rand_seed)
                    predictions = 0    
                    for w, tta in zip(CFG.wtta, CFG.ntta):
                        padding_front = False if CFG.encoder=='LSTM' else True
                        valid_db = BowlDataset(CFG, test_df, chosen_samples, aug=tta, aug_p=1.0, 
                                               padding_front=padding_front, use_tta=True)
                        valid_loader = DataLoader(
                                valid_db, batch_size=CFG.batch_size, shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True)                
                        prediction, groundtruths = validate(valid_loader, model)
                        predictions += w*prediction                                            
                    try:
                        valid_kappa, valid_coefficients = get_optimized_kappa_score(predictions, groundtruths)
                        print(f'k[{k}]-s2[{rand_seed}]: valid_kappa:{valid_kappa} - {valid_coefficients}') 
                        sum_coefficients += np.array(valid_coefficients)
                        sum_cnt += 1
                    except Exception as e:
                        print(e)
                        print(f'k[{k}]-s2[{rand_seed}]: valid_kappa: Failed!')
                        pass
                del model
    ################################################
    test_samples = list(test_df.groupby(['installation_id']).groups.values())    
    
    coefficients = 0.2*sum_coefficients/sum_cnt + 0.8*np.array([0.53060865, 1.66266655, 2.31145611])       
    print('=======================================')
    print(f'coefficients - {coefficients}')
    print('=======================================')
    
    random.seed(CFG.seed)
    
    submission_df = test_df.groupby('installation_id').tail(1)[['installation_id']]
    submission_df['accuracy_group'] = 0
    
    for _, base_model_paths in base_model_path_list:
        for model_w, base_model_path in base_model_paths:        
            path = base_model_path.split('/')[-1]
            path = path.replace('bowl_', '')
            cfg_dict = dict([tok.split('-') for tok in path.split('_')])
            CFG.encoder = cfg_dict['a']
            CFG.seq_len = int(cfg_dict['len'])
            CFG.emb_size = int(cfg_dict['e'])
            CFG.hidden_size = int(cfg_dict['h'])
            CFG.nlayers = int(cfg_dict['l'])
            CFG.nheads = int(cfg_dict['hd'])
            CFG.seed = int(cfg_dict['s'])
            CFG.data_seed = int(cfg_dict['s'])
            
            for k in range(5):
                model = ENCODERS[CFG.encoder](CFG)
                model_path = base_model_path.replace('k-0', f'k-{k}')
                
                checkpoint = torch.load(model_path, map_location=CFG.device)        
                model.load_state_dict(checkpoint['state_dict'])
                model.to(CFG.device)
                print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))            
                                      
                for w, tta in zip(CFG.wtta, CFG.ntta):
                    padding_front = False if CFG.encoder=='LSTM' else True
                    valid_db = BowlDataset(CFG, test_df, test_samples, aug=tta, aug_p=1.0, 
                                           padding_front=padding_front, use_tta=True)
                    valid_loader = DataLoader(
                            valid_db, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True)                
                    predictions = test(valid_loader, model)
                    submission_df['accuracy_group'] += w*predictions*model_w*(1/5)
                del model
    
    submission_df['accuracy_group'] /= len(base_model_path_list)
    compute_th_acc_gp(submission_df['accuracy_group'], coefficients) 
    submission_df['accuracy_group'] = submission_df['accuracy_group'].astype(int)
    os.makedirs(settings['SUBMISSION_DIR'], exist_ok=True)
    submission_df.to_csv(os.path.join(settings['SUBMISSION_DIR'], 'submission.csv'), index=False)
    print('done')

def compute_th_acc_gp(temp, coef):
    temp[temp < coef[0]] = 0
    temp[(coef[0]<=temp)&(temp< coef[1])] = 1
    temp[(coef[1]<=temp)&(temp< coef[2])] = 2
    temp[(coef[2]<=temp)] = 3    

def compute_acc_gp(pred):
    #batch_size = pred.size(0)
    pred = (3*pred[:, 0] - 2*pred[:, 1])    
    pred[pred < 0] = 0    
    return pred


def validate(valid_loader, model):
    model.eval()    
    
    predictions = []
    groundtruths = []
    for step, (cate_x, cont_x, mask, y) in enumerate(valid_loader):
        
        cate_x, cont_x, mask = cate_x.to(CFG.device), cont_x.to(CFG.device), mask.to(CFG.device)        
        
        k = 0.5
        with torch.no_grad():        
            pred = model(cate_x, cont_x, mask)
          
        # record accuracy
        pred_y = (1-k)*pred[:, 0] + (k)*compute_acc_gp(pred[:, 1:])
        predictions.append(pred_y.detach().cpu())        
        groundtruths.append(y[:, 0])

    predictions = torch.cat(predictions).numpy()
    groundtruths = torch.cat(groundtruths).numpy()
    
    return predictions, groundtruths


def test(valid_loader, model):
    model.eval()    
    
    predictions = []
    for step, (cate_x, cont_x, mask, _) in enumerate(valid_loader):
        
        cate_x, cont_x, mask = cate_x.to(CFG.device), cont_x.to(CFG.device), mask.to(CFG.device)        
        
        k = 0.5
        with torch.no_grad():        
            pred = model(cate_x, cont_x, mask)
          
        # record accuracy
        pred_y = (1-k)*pred[:, 0] + (k)*compute_acc_gp(pred[:, 1:])
        predictions.append(pred_y.detach().cpu())        

    predictions = torch.cat(predictions).numpy()
    
    return predictions


if __name__ == '__main__':
    main()