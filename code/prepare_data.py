import torch
import numpy as np
import pandas as pd
import os, gc, sys, warnings, random, math, psutil, pickle

from tqdm import tqdm as tqdm_notebook
import json

warnings.filterwarnings('ignore')

settings = json.load(open('SETTINGS.json'))

# DATA LOAD
print('loading data ...')
train_df = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'train.csv'))
test_df = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'test.csv'))
train_label_df = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'train_labels.csv'))
specs_df = pd.read_csv(os.path.join(settings['RAW_DATA_DIR'], 'specs.csv'))
print('loading ... done')

def replace_4110_4100(df):
    rep_code4110_bool = (df['title']=='Bird Measurer (Assessment)')&(df['event_code']==4110)
    rep_code4100_bool = (df['title']=='Bird Measurer (Assessment)')&(df['event_code']==4100)
    df['event_code'][rep_code4110_bool] = 4100
    df['event_code'][rep_code4100_bool] = 5110 
replace_4110_4100(train_df)
replace_4110_4100(test_df)

# Create additional columns from event_code
def extract_data_from_event_code(df, columns=['correct', 'round']):
    for col in columns:
        col_bool = df['event_data'].str.contains(col)
        df[col] = np.nan
        df[col][col_bool] = df['event_data'][col_bool].apply(lambda x: json.loads(x).get(col)).astype(float)

print('extract_data_from_event_code ...')
extract_data_from_event_code(train_df)
extract_data_from_event_code(test_df)
        
train_df['num_incorrect'] = np.where(train_df['correct']==0, 1, np.nan)
train_df['num_correct'] = np.where(train_df['correct']==1, 1, np.nan)
test_df['num_incorrect'] = np.where(test_df['correct']==0, 1, np.nan)
test_df['num_correct'] = np.where(test_df['correct']==1, 1, np.nan)

# Convert game_time to seconds
train_df['game_time'] = train_df['game_time'] // 1000
test_df['game_time'] = test_df['game_time'] // 1000

# Aggregation by game_session
def get_agged_session(df):
    event_code = pd.crosstab(df['game_session'], df['event_code'])
    event_id = pd.crosstab(df['game_session'], df['event_id'])
        
    event_num_correct = pd.pivot_table(df[(~df['correct'].isna())], index='game_session', columns='event_code', values='num_correct', aggfunc='sum')
    event_num_incorrect = pd.pivot_table(df[(~df['correct'].isna())], index='game_session', columns='event_code', values='num_incorrect', aggfunc='sum')
    event_accuracy = event_num_correct/(event_num_correct+event_num_incorrect[event_num_correct.columns])
    event_accuracy = event_accuracy.add_prefix('accuray_')    
    
    event_round = pd.pivot_table(df[~df['correct'].isna()], index='game_session', columns='event_code', values='round', aggfunc='max')
    event_round = event_round.add_prefix('round_')    
    
    df['elapsed_time'] = df[['game_session', 'game_time']].groupby('game_session')['game_time'].diff()
    game_time = df.groupby('game_session', as_index=False)['elapsed_time'].agg(['mean', 'max']).reset_index()
    game_time.columns = ['game_session', 'mean_game_time', 'max_game_time']    
    df = df.merge(game_time, on='game_session', how='left')     
    del df['elapsed_time']
    
    session_extra_df = pd.concat([event_code, event_id, event_accuracy, event_round], 1)
    session_extra_df.index.name = 'game_session'
    session_extra_df.reset_index(inplace=True)
    
    session_df = df.drop_duplicates('game_session', keep='last').reset_index(drop=True)
    session_df['row_id'] = session_df.index
    session_df = session_df.merge(session_extra_df, how='left', on='game_session')
    return session_df

print('get_agged_session ...')
agged_train_df = get_agged_session(train_df)
agged_test_df = get_agged_session(test_df)

agged_train_df = agged_train_df.drop(['correct', 'round', 'num_correct', 'num_incorrect'], axis=1)
agged_test_df = agged_test_df.drop(['correct', 'round', 'num_correct', 'num_incorrect'], axis=1)

agged_test_df = agged_test_df.append(pd.DataFrame(columns=agged_train_df.columns))

#Additional training data generation
def gen_game_label(df):
    num_corrects = []
    for inst_id, one_df in tqdm_notebook(df.groupby('installation_id'), leave=False):
        one_df = one_df[(one_df['type']=='Game')&(one_df['event_code'].isin([4020, 4025]) )]
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
print('gen_game_label ...')
train_game_label_df = gen_game_label(train_df)
test_game_label_df = gen_game_label(test_df)

# Generate&Merge label
def gen_label(df):
    num_corrects = []
    for inst_id, one_df in tqdm_notebook(df.groupby('installation_id'), leave=False):
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
print('gen_label ...')
train_label_df = gen_label(train_df)
test_label_df = gen_label(test_df)

agged_train_df = agged_train_df.merge(train_label_df, on=['game_session', 'installation_id'], how='left')
agged_train_df = agged_train_df.merge(train_game_label_df, on=['game_session', 'installation_id'], how='left', suffixes=('', '_game'))
agged_test_df = agged_test_df.merge(test_label_df, on=['game_session', 'installation_id'], how='left')
agged_test_df = agged_test_df.merge(test_game_label_df, on=['game_session', 'installation_id'], how='left', suffixes=('', '_game'))
agged_test_df = agged_test_df[agged_train_df.columns]
print(agged_train_df.shape)
print(agged_test_df.shape)

agged_train_df[(agged_train_df['accuracy_group'] >= 0)&(agged_train_df['type']=='Assessment')].shape

# Generate sample_indices
def get_train_sample_indices(df):
    sample_indices = []
    inst_indiecs = []    
    df_groups = df.groupby('installation_id').groups
    for inst_idx, indices in enumerate(tqdm_notebook(df_groups.values())):
        one_df = df.iloc[indices].reset_index(drop=True)
        assessment_start_indices = one_df[(one_df['type']=='Assessment')&
                                          (one_df['accuracy_group']>=0)
                                         ].index
        for num, start_index in enumerate(assessment_start_indices):
            sample_indices.append( one_df.iloc[:start_index+1]['row_id'].tolist() )
            inst_indiecs.append(inst_idx)            
    return sample_indices, inst_indiecs

train_samples, train_groups = get_train_sample_indices(agged_train_df)
test_samples, test_groups = get_train_sample_indices(agged_test_df)
print(len(train_samples), len(test_samples))

def get_train_game_sample_indices(df):
    sample_indices = []
    inst_indiecs = []    
    df_groups = df.groupby('installation_id').groups
    for inst_idx, indices in enumerate(tqdm_notebook(df_groups.values())):
        one_df = df.iloc[indices].reset_index(drop=True)
        assessment_start_indices = one_df[(one_df['type']=='Game')&
                                          (one_df['accuracy_group_game']>=0)
                                         ].index
        for num, start_index in enumerate(assessment_start_indices):
            sample_indices.append( one_df.iloc[:start_index+1]['row_id'].tolist() )
            inst_indiecs.append(inst_idx)            
    return sample_indices, inst_indiecs

print('get_train_game_sample_indices ...')
train_game_samples, train_game_groups = get_train_game_sample_indices(agged_train_df)
test_game_samples, test_game_groups = get_train_game_sample_indices(agged_test_df)
print(len(train_game_samples), len(test_game_samples))

agged_train_df = agged_train_df.fillna(0)
agged_test_df = agged_test_df.fillna(0)

# Convert categorical data to corresponding index
all_df = pd.concat([agged_train_df, agged_test_df])
cate_cols = ['title', 'type', 'world']
cont_cols = ['event_count', 'game_time', 'max_game_time']
extra_cont_cls = list(agged_train_df.columns[15:-4]) # except 2000
mappers_dict = {}

cate_offset = 1
for col in tqdm_notebook(cate_cols):    
    cate2idx = {}
    for v in all_df[col].unique():
        if (v != v) | (v == None): continue 
        cate2idx[v] = len(cate2idx)+cate_offset
    mappers_dict[col] = cate2idx    
    agged_train_df[col] = agged_train_df[col].map(cate2idx).fillna(0).astype(int)
    agged_test_df[col] = agged_test_df[col].map(cate2idx).fillna(0).astype(int)
    cate_offset += len(cate2idx)
del all_df

os.makedirs(settings['CLEAN_DATA_DIR'], exist_ok=True)
torch.save([agged_train_df, agged_test_df, mappers_dict, cate_offset, cate_cols, cont_cols, extra_cont_cls, 
            train_samples, train_groups, test_samples, train_game_samples, test_game_samples],
           os.path.join(settings['CLEAN_DATA_DIR'], 'bowl.pt'))

torch.save([agged_train_df.columns, mappers_dict, cate_offset, cate_cols, cont_cols, extra_cont_cls],
           os.path.join(settings['CLEAN_DATA_DIR'], 'bowl_info.pt'))
