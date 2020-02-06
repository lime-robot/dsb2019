import os
import sys
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import os
import math
import copy
import torch
import bowl_db
import bowl_model
import bowl_utils
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings(action='ignore')

import argparse
import logging
import json
import collections

settings = json.load(open('SETTINGS.json'))

DB_PATH=settings['CLEAN_DATA_DIR']


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
    ntta = [0.0, 0.3, 0.5]
    wtta = [0.6, 0.2, 0.2]
    #ntta = [0.0]
    #wtta = [1.0]
    


def main():    
    parser = argparse.ArgumentParser("")
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)    
    parser.add_argument("--nepochs", type=int, default=CFG.num_train_epochs)
    parser.add_argument("--seed", type=int, default=7)    
    parser.add_argument("--data_seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)        
    args = parser.parse_args()
    print(args) 
    
    CFG.batch_size=args.batch_size   
    CFG.seed =  args.seed
    CFG.data_seed =  args.data_seed
    CFG.target_size = 3
    print(CFG.__dict__)    
    
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)    
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True        
    
    
    base_model_path_list = [        
        ['bowl.pt', os.path.join(settings['MODEL_DIR'], 'b-32_a-TRANSFORMER_e-100_h-500_d-0.2_l-2_hd-10_s-7_len-100_aug-0.5_da-bowl.pt_k-0.pt')],        
    ]
    
    
    rand_seed_list = [7, 77, 777, 1, 2]
    
    total_predictions = []
    total_groundtruth = []      
    
    for k in range(5):
        mean_predictions = 0
        mean_groundtruth = 0
        prev_filename = ''
        for filename, base_model_path in base_model_path_list:            
            if prev_filename != filename:
                (train_df, test_df, mappers_dict, cate_offset, cate_cols, 
                 cont_cols, extra_cont_cls, train_samples, train_groups, test_samples) = (
                    torch.load(os.path.join(DB_PATH, filename)))[:10]
            prev_filename = filename
               
            
            CFG.total_cate_size = cate_offset
            CFG.cate_cols = cate_cols
            CFG.cont_cols = cont_cols+extra_cont_cls
            
            path = base_model_path.split('/')[-1]
            path = path.replace('bowl_', '')
            cfg_dict = dict([tok.split('-') for tok in path.split('_')])
            CFG.encoder = cfg_dict['a']
            CFG.seq_len = int(cfg_dict['len'])
            CFG.emb_size = int(cfg_dict['e'])
            CFG.hidden_size = int(cfg_dict['h'])
            CFG.nlayers = int(cfg_dict['l'])
            CFG.nheads = int(cfg_dict['hd'])
            
            model_path = base_model_path.replace('k-0', f'k-{k}')
            model = bowl_model.encoders[CFG.encoder](CFG)   
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['state_dict'])        
            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
            model.cuda()
            
            rand_predictions = []
            rand_groundtruth = []
            
            for rand_seed in rand_seed_list:
                _, valid_samples = bowl_utils.train_valid_split(train_samples, train_groups, k, 
                                                                random_state=CFG.data_seed, random_state2=rand_seed, choice=True)                                
                predictions = 0
                for w, tta in zip(CFG.wtta, CFG.ntta):                    
                    valid_db = bowl_db.BowlDataset(CFG, train_df, valid_samples, aug=tta, aug_p=1.0)
                    valid_loader = DataLoader(
                            valid_db, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True)                
                    _, valid_kappa, _, prediction, groundtruth = validate(valid_loader, model)
                    predictions += w*prediction                                

                rand_predictions.append(predictions)
                rand_groundtruth.append(groundtruth)                    
                
                valid_kappa = bowl_utils.get_optimized_kappa_score(predictions, groundtruth)
                print(f'k[{k}]-s2[{rand_seed}]: valid_kappa:{valid_kappa}')                
            del model
            mean_predictions += np.concatenate(rand_predictions)
            mean_groundtruth += np.concatenate(rand_groundtruth)
        
        total_predictions.append(mean_predictions/len(base_model_path_list))
        total_groundtruth.append(mean_groundtruth/len(base_model_path_list))
        
    total_predictions = np.concatenate(total_predictions)
    total_groundtruth = np.concatenate(total_groundtruth)    
    
    print(total_predictions.shape)
    
    optR = bowl_utils.OptimizedRounder()
    optR.fit(total_predictions, total_groundtruth)
    coefficients = optR.coefficients()
    #coefficients = [0.50755102, 1.64870448, 2.23524805]
    #coefficients = [0.49057894, 1.66282769, 2.26743377]
    #print('FIXED COEEFICIENT !!!!')
    
    total_predictions[total_predictions < coefficients[0]] = 0
    total_predictions[(coefficients[0]<=total_predictions)&(total_predictions< coefficients[1])] = 1
    total_predictions[(coefficients[1]<=total_predictions)&(total_predictions< coefficients[2])] = 2
    total_predictions[(coefficients[2]<=total_predictions)] = 3

    kappa_score = bowl_utils.qwk3(total_predictions, total_groundtruth)
    print('==============================')    
    print(f'VALID KAPPA_SCORE:{kappa_score} - {coefficients}')
    print('==============================')
    
    
    if len(test_samples)>1000:        
        predictions = 0
        accum_count = 0
        for filename, base_model_path in base_model_path_list:
            if prev_filename != filename:
                (train_df, test_df, mappers_dict, cate_offset, cate_cols, 
                 cont_cols, extra_cont_cls, train_samples, train_groups, test_samples) = (
                    torch.load(os.path.join(DB_PATH, filename)))[:10]
            prev_filename = filename
            
            CFG.total_cate_size = cate_offset
            CFG.cate_cols = cate_cols
            CFG.cont_cols = cont_cols+extra_cont_cls            
            
            path = base_model_path.split('/')[-1]
            path = path.replace('bowl_', '')
            cfg_dict = dict([tok.split('-') for tok in path.split('_')])
            CFG.encoder = cfg_dict['a']
            CFG.seq_len = int(cfg_dict['len'])
            CFG.emb_size = int(cfg_dict['e'])
            CFG.hidden_size = int(cfg_dict['h'])
            CFG.nlayers = int(cfg_dict['l'])
            CFG.nheads = int(cfg_dict['hd'])
                
            for k in range(5):            
                model = bowl_model.encoders[CFG.encoder](CFG)
                model_path = base_model_path.replace('k-0', f'k-{k}')
                checkpoint = torch.load(model_path)        
                model.load_state_dict(checkpoint['state_dict'])        
                print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
                model.cuda()
                                                
                for w, tta in zip(CFG.wtta, CFG.ntta):
                    valid_db = bowl_db.BowlDataset(CFG, test_df, test_samples, aug=tta, aug_p=1.0)
                    valid_loader = DataLoader(
                            valid_db, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True)                
                    _, valid_kappa, _, prediction, groundtruth = validate(valid_loader, model)
                    predictions += w*prediction
                accum_count += 1
                del model                
        kappa_score = bowl_utils.get_optimized_kappa_score(predictions/accum_count, groundtruth)
        print('==============================')            
        print(f'TEST KAPPA_SCORE:{kappa_score}')            
        
    #ass_bool = (test_df['event_code']!=2000)&(test_df['num_correct']==0)&(test_df['num_incorrect']==0)&(test_df['type']==mappers_dict['type']['Assessment'])
    #test_df = test_df[~ass_bool].reset_index(drop=True)
        
    submission_df = test_df.groupby('installation_id').tail(1)[['installation_id']]
    submission_df['accuracy_group'] = 0
    accum_count = 0
    
    for filename, base_model_path in base_model_path_list:
        if prev_filename != filename:
                (train_df, test_df, mappers_dict, cate_offset, cate_cols, 
                 cont_cols, extra_cont_cls, train_samples, train_groups, _) = (
                    torch.load(os.path.join(DB_PATH, filename)))[:10]
        prev_filename = filename
        
        test_samples = list(test_df.groupby(['installation_id']).groups.values())
        
        CFG.total_cate_size = cate_offset
        CFG.cate_cols = cate_cols
        CFG.cont_cols = cont_cols+extra_cont_cls        
        
        path = base_model_path.split('/')[-1]
        path = path.replace('bowl_', '')
        cfg_dict = dict([tok.split('-') for tok in path.split('_')])
        CFG.encoder = cfg_dict['a']
        CFG.seq_len = int(cfg_dict['len'])
        CFG.emb_size = int(cfg_dict['e'])
        CFG.hidden_size = int(cfg_dict['h'])
        CFG.nlayers = int(cfg_dict['l'])
        CFG.nheads = int(cfg_dict['hd'])
        
        for k in range(5):
            model = bowl_model.encoders[CFG.encoder](CFG)
            model_path = base_model_path.replace('k-0', f'k-{k}')
            
            checkpoint = torch.load(model_path)        
            model.load_state_dict(checkpoint['state_dict'])        
            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
            model.cuda()
                                  
            for w, tta in zip(CFG.wtta, CFG.ntta):       
                valid_db = bowl_db.BowlDataset(CFG, test_df, test_samples, aug=tta, aug_p=1.0)
                valid_loader = DataLoader(
                        valid_db, batch_size=CFG.batch_size, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=True)                
                predictions = test(valid_loader, model)
                submission_df['accuracy_group'] += w*predictions
            accum_count += 1
            
                            
            del model
    
    submission_df['accuracy_group'] /= accum_count    
    compute_th_acc_gp(submission_df['accuracy_group'], coefficients) 
    submission_df['accuracy_group'] = submission_df['accuracy_group'].astype(int)
    submission_df.to_csv('submission.csv', index=False)
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


def test(test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    sent_count = AverageMeter()
    #meter = bowl_utils.Meter()
    
    # switch to evaluation mode
    model.eval()

    start = end = time.time()
    
    predictions = []    
    for step, (cate_x, cont_x, mask, y) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)        
        
        cate_x, cont_x, mask = cate_x.cuda(), cont_x.cuda(), mask.cuda()
        batch_size = cate_x.size(0)        
        
        # compute loss
        k = 0.5
        with torch.no_grad():        
            pred = model(cate_x, cont_x, mask)
        # record accuracy
        pred_y = (1-k)*pred[:, 0] + (k)*compute_acc_gp(pred[:, 1:])
        predictions.append(pred_y.detach().cpu())        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)
        """
        if step % CFG.print_freq == 0 or step == (len(test_loader)-1):            
            print('TEST: {0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '                  
                  'sent/s {sent_s:.0f} '
                  .format(
                   step, len(test_loader), batch_time=batch_time,                   
                   data_time=data_time,                    
                   remain=timeSince(start, float(step+1)/len(test_loader)),
                   sent_s=sent_count.avg/batch_time.avg
                   ))
        """
    predictions = torch.cat(predictions).numpy()    
    
    return predictions


def validate(valid_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    sent_count = AverageMeter()
    #meter = bowl_utils.Meter()
    
    # switch to evaluation mode
    model.eval()

    start = end = time.time()
    
    predictions = []
    groundtruth = []
    for step, (cate_x, cont_x, mask, y) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)        
        
        cate_x, cont_x, mask, y = cate_x.cuda(), cont_x.cuda(), mask.cuda(), y.cuda()
        batch_size = cate_x.size(0)        
        
        # compute loss
        k = 0.5
        with torch.no_grad():        
            pred = model(cate_x, cont_x, mask)
            loss1 = F.mse_loss(pred[:, 0].contiguous().view(-1), y[:, 0].contiguous().view(-1))
            loss2 = F.mse_loss(pred[:, 1].contiguous().view(-1), y[:, 1].contiguous().view(-1))
            loss = (1-k)*loss1+k*loss2

            # record loss
            losses.update(loss.item(), batch_size)
        
        # record accuracy
        pred_y = (1-k)*pred[:, 0] + (k)*compute_acc_gp(pred[:, 1:])
        predictions.append(pred_y.detach().cpu())        
        pred_y = (pred_y+0.5).int()
        pred_y[pred_y > 3] = 3        
        y = y[:, 0]
        
        pred_y = pred_y.detach().cpu()
        y = y.cpu()
        
        groundtruth.append(y)
        
        kappa_score = bowl_utils.qwk3(pred_y.numpy(), y.numpy())        
        accuracies.update( kappa_score, batch_size)
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps    

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)
        
        """
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):            
            print('TEST: {0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Acc: {acc.val:.4f}({acc.avg:.4f}) '
                  'sent/s {sent_s:.0f} '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,                   
                   data_time=data_time, loss=losses,
                   acc=accuracies,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   sent_s=sent_count.avg/batch_time.avg
                   ))
        """
    predictions = torch.cat(predictions).numpy()
    groundtruth = torch.cat(groundtruth).numpy()
    
    """
    try:
        optR = bowl_utils.OptimizedRounder()
        optR.fit(predictions, groundtruth)
        coefficients = optR.coefficients()
        print(coefficients)
        temp_predictions = predictions.copy()
        temp_predictions[temp_predictions < coefficients[0]] = 0
        temp_predictions[(coefficients[0]<=temp_predictions)&(temp_predictions< coefficients[1])] = 1
        temp_predictions[(coefficients[1]<=temp_predictions)&(temp_predictions< coefficients[2])] = 2
        temp_predictions[(coefficients[2]<=temp_predictions)] = 3

        kappa_score = bowl_utils.qwk3(temp_predictions, groundtruth)
    except:
        kappa_score = 0
        coefficients = [0.5, 1.5, 2.5]
    """
    coefficients = [0.5, 1.5, 2.5]  
    
    return losses.avg, kappa_score, coefficients, predictions, groundtruth


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger

logger = get_logger()


def save_checkpoint(state, model_path, model_filename, is_best=False):
    print('saving cust_model ...')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(state, os.path.join(model_path, model_filename))
    if is_best:
        torch.save(state, os.path.join(model_path, 'best_' + model_filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def adjust_learning_rate(optimizer, epoch):  
    #lr  = CFG.learning_rate     
    lr = (CFG.lr_decay)**(epoch//10) * CFG.learning_rate    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
    return lr




if __name__ == '__main__':
    main()
