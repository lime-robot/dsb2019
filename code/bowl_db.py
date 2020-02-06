#import warnings
#warnings.filterwarnings('ignore')

import sys
import os
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random


TARGET = ['accuracy_group', 'num_correct', 'num_incorrect']
GAME_TARGET = ['accuracy_group_game', 'num_correct_game', 'num_incorrect_game']
#TARGET = ['accuracy_group']


class BowlDataset(Dataset):
    def __init__(self, cfg, df, sample_indices, aug=0.0, aug_p=0.5):
        self.cfg = cfg
        self.df = df.copy()    
        self.sample_indices = sample_indices
        self.seq_len = self.cfg.seq_len
        self.aug = aug
        self.aug_p = aug_p
         
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
                    cut_ratio = np.random.rand()
                    if cut_ratio > self.aug:
                        cut_ratio = self.aug
                    #cut_ratio = self.aug
                    start_idx = max(int(len(indices)*cut_ratio), 30)
                    indices = indices[start_idx:]
                    seq_len = min(self.seq_len, len(indices))
        
        tmp_cate_x = torch.LongTensor(self.cate_df.iloc[indices].values)
        cate_x = torch.LongTensor(self.seq_len, len(self.cate_cols)).zero_()
        cate_x[-seq_len:] = tmp_cate_x[-seq_len:]        
        
        tmp_cont_x = torch.FloatTensor(self.cont_df.iloc[indices].values)
        tmp_cont_x[-1] = 0
        cont_x = torch.FloatTensor(self.seq_len, len(self.cont_cols)).zero_()
        cont_x[-seq_len:] = tmp_cont_x[-seq_len:]
        
        mask = torch.ByteTensor(self.seq_len).zero_()
        mask[-seq_len:] = 1
        
        if self.target_df is not None:
            target = torch.FloatTensor(self.target_df.iloc[indices[-1]].values)
            if target.sum() == 0:                
                target = torch.FloatTensor(self.target_game_df.iloc[indices[-1]].values)            
        else:
            target = 0
        
        return cate_x, cont_x, mask, target

    def __len__(self):
        return len(self.sample_indices)

DB_PATH='../../input/data-science-bowl-2019'
def main():    
    (train_df, test_df, mappers_dict, cate_offset, cate_cols, 
     cont_cols, extra_cont_cls, train_samples, train_groups, test_samples) = (
        torch.load(os.path.join(DB_PATH, 'bowl_v28.pt')))    
    
    train_db = BowlDataset(train_df, train_samples, mappers_dict)

    for cate_x, cont_x, mask, target in train_db:
        a = 0
    
    

if __name__ == '__main__':
    main()