import os
import math
import copy
import torch
import json
import time
import random
import logging
import argparse
import collections
import numpy as np
import pandas as pd
import bowl_db
import bowl_model
import bowl_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings(action='ignore')

settings = json.load(open('SETTINGS.json'))
DB_PATH=settings['CLEAN_DATA_DIR']
FINETUNED_MODEL_PATH=settings['MODEL_DIR']


class CFG:
    learning_rate=1.0e-4
    batch_size=64
    num_workers=4
    print_freq=100
    test_freq=1
    start_epoch=0
    num_train_epochs=10
    warmup_steps=30
    max_grad_norm=1000
    gradient_accumulation_steps=1
    weight_decay=0.01    
    dropout=0.2
    emb_size=100
    hidden_size=500
    nlayers=2
    nheads=10
    seq_len=100


def main():    
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--data", type=str, default='bowl.pt')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--use_test", action='store_true')
    parser.add_argument("--aug", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--grad_accums", type=int, default=CFG.gradient_accumulation_steps)
    parser.add_argument("--nepochs", type=int, default=CFG.num_train_epochs)    
    parser.add_argument("--wsteps", type=int, default=CFG.warmup_steps)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data_seed", type=int, default=7)
    parser.add_argument("--seq_len", type=int, default=CFG.seq_len)
    parser.add_argument("--nlayers", type=int, default=CFG.nlayers)
    parser.add_argument("--nheads", type=int, default=CFG.nheads)
    parser.add_argument("--hidden_size", type=int, default=CFG.hidden_size)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)
    parser.add_argument("--encoder", type=str, default='TRANSFORMER')
    args = parser.parse_args()
    print(args) 
    
    CFG.batch_size=args.batch_size
    CFG.gradient_accumulation_steps = args.grad_accums
    CFG.batch_size = CFG.batch_size // CFG.gradient_accumulation_steps
    CFG.num_train_epochs=args.nepochs
    CFG.warmup_steps=args.wsteps    
    CFG.learning_rate=args.lr
    CFG.dropout=args.dropout
    CFG.seed =  args.seed
    CFG.data_seed =  args.data_seed
    CFG.seq_len =  args.seq_len
    CFG.nlayers =  args.nlayers
    CFG.nheads =  args.nheads
    CFG.hidden_size =  args.hidden_size
    CFG.res_dir=f'res_dir_{args.k}'
    CFG.target_size = 3
    CFG.encoder = args.encoder
    CFG.aug = args.aug
    print(CFG.__dict__)    
    
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)    
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True 
    
    data_path = os.path.join(DB_PATH, args.data)
    (train_df, test_df, mappers_dict, cate_offset, cate_cols,  
     cont_cols, extra_cont_cls, train_samples, train_groups, test_samples,
     train_game_samples, test_game_samples) = (
        torch.load(data_path))
    print(data_path)
    print(cate_cols, cont_cols)
    
    CFG.total_cate_size = cate_offset
    CFG.cate_cols = cate_cols
    CFG.cont_cols = cont_cols + extra_cont_cls    
    
    model = bowl_model.encoders[CFG.encoder](CFG)
    if args.model != "":
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        
        state_dict = collections.OrderedDict([(k, v) for k, v in checkpoint['state_dict'].items() if 'reg.' not in k])
        CFG.start_epoch = checkpoint['epoch']        
        model.load_state_dict(state_dict, strict=False)        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model, checkpoint['epoch']))        
    
    model.cuda()
    model._dropout = CFG.dropout
    print('model.dropout:', model._dropout)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('parameters: ', count_parameters(model))
    
    #n_gpu = torch.cuda.device_count()
    #if n_gpu > 1:
    #    model = torch.nn.DataParallel(model)  
    
    train_samples, valid_samples = bowl_utils.train_valid_split(train_samples, train_groups, args.k, 
                                                                random_state=CFG.data_seed, random_state2=CFG.data_seed, choice=True)
    print(train_samples.shape, valid_samples.shape)
    if args.use_test:        
        extra_samples = np.array([np.array(indices) + len(train_df) for indices in test_samples])
        train_df = train_df.append(test_df).reset_index(drop=True)
        train_df['row_id'] = train_df.index
        train_samples = np.concatenate([train_samples, extra_samples])
        print(train_samples.shape, valid_samples.shape)    
    
    last_indices = [indices[-1] for indices in valid_samples]
    valid_installation_ids = train_df.iloc[last_indices]['installation_id'].unique()
    
    # remove samples for validation 
    print(len(train_game_samples))
    last_game_indices = [indices[-1] for indices in train_game_samples]
    train_game_samples = [indices for indices, inst_id in zip(train_game_samples, train_df.iloc[last_game_indices]['installation_id']) if inst_id not in valid_installation_ids]
    print(len(train_game_samples))
    
    
    ext_train_db = bowl_db.BowlDataset(CFG, train_df, train_game_samples+list(train_samples), aug=CFG.aug)
    train_db = bowl_db.BowlDataset(CFG, train_df, train_samples, aug=CFG.aug)
    valid_db = bowl_db.BowlDataset(CFG, train_df, valid_samples)
    
    num_train_optimization_steps = int(
        len(ext_train_db) / CFG.batch_size / CFG.gradient_accumulation_steps) * (3)
    num_train_optimization_steps += int(
        len(train_db) / CFG.batch_size / CFG.gradient_accumulation_steps) * (7)
    print('num_train_optimization_steps', num_train_optimization_steps)    

    ext_train_loader = DataLoader(
        ext_train_db, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True)
    train_loader = DataLoader(
        train_db, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True)
    valid_loader = DataLoader(
        valid_db, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True)
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                           lr=CFG.learning_rate,
                           weight_decay=CFG.weight_decay,                           
                           )

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps,
                                        num_training_steps=num_train_optimization_steps
                                     )                                   
  
    print('use WarmupLinearSchedule ...')
    
    def get_lr():
        return scheduler.get_lr()[0]
    
    if args.model != "":
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])        
        log_df = checkpoint['log']
        del checkpoint
    else:
        log_df = pd.DataFrame(columns=(['EPOCH']+['LR']+['TRAIN_LOSS', 'TRAIN_KAPPA']+
                                       ['VALID_LOSS', 'VALID_KAPPA']) )     
    os.makedirs('log', exist_ok=True)
    
    curr_lr = get_lr()    
    
    print(f'initial learning rate:{curr_lr}')
        
    submission_df = train_df.iloc[[indices[-1] for indices in valid_samples]]
    print(submission_df.shape)
    
    best_kappa = 0
    best_model = None
    best_epoch = 0
    
    model_list = []        
            
    for epoch in range(CFG.start_epoch, CFG.num_train_epochs):
        # train for one epoch
        
        if epoch < 3:
            train_loss, train_kappa = train(ext_train_loader, model, optimizer, epoch, scheduler)
        else:
            train_loss, train_kappa = train(train_loader, model, optimizer, epoch, scheduler)
        
        valid_loss, valid_kappa, _, _, _ = validate(valid_loader, model)        
    
        curr_lr = get_lr()       
        print(f'set the learning_rate: {curr_lr}')
        
        model_list.append(copy.deepcopy(model))
        if epoch % CFG.test_freq == 0 and epoch >= 0:
            log_row = {'EPOCH':epoch, 'LR':curr_lr,
                       'TRAIN_LOSS':train_loss, 'TRAIN_KAPPA':train_kappa,
                       'VALID_LOSS':valid_loss, 'VALID_KAPPA':valid_kappa,
                       } 
                           
            log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)                        
            print(log_df.tail(20))           
            
            batch_size = CFG.batch_size*CFG.gradient_accumulation_steps
                        
            if (best_kappa < valid_kappa):
                best_model = copy.deepcopy(model)
                best_kappa = valid_kappa
                best_epoch = epoch
            
    model_list = model_list[6:]    
    last_model = best_model
    last_params = dict(last_model.named_parameters())
    for i in range(len(model_list)-1):
        curr_params = dict(model_list[i].named_parameters())        
        for name, param  in last_params.items():      
            param.data += curr_params[name].data
    for name, param  in last_params.items():
        param.data /= len(model_list)
    model = last_model
    
    valid_loss, valid_acc, coefficients, _, _ = validate(valid_loader, model)
    print(valid_loss, valid_acc)
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the cust_model it-self    
    
    input_filename = args.data.split('/')[-1]
    curr_model_name = (f'b-{batch_size}_a-{CFG.encoder}_e-{CFG.emb_size}_h-{CFG.hidden_size}_'                               
                               f'd-{CFG.dropout}_l-{CFG.nlayers}_hd-{CFG.nheads}_'
                               f's-{CFG.seed}_len-{CFG.seq_len}_aug-{CFG.aug}_da-{input_filename}_k-{args.k}.pt')
    save_checkpoint({
        'epoch': best_epoch + 1,
        'arch': 'transformer',
        'state_dict': model_to_save.state_dict(),
        'log': log_df,
        'coefficients': coefficients,
        },        
        FINETUNED_MODEL_PATH, curr_model_name,
    )    
    print('done')


def compute_acc_gp(pred):
    pred = (3*pred[:, 0] - 2*pred[:, 1])
    pred[pred < 0] = 0    
    return pred


def train(train_loader, model, optimizer, epoch, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    sent_count = AverageMeter()
    #meter = bowl_utils.Meter()
    
    # switch to train mode
    model.train()

    start = end = time.time()
    global_step = 0
    
    for step, (cate_x, cont_x, mask, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        cate_x, cont_x, mask, y = cate_x.cuda(), cont_x.cuda(), mask.cuda(), y.cuda()
        batch_size = cate_x.size(0)        
        
        # compute loss
        k = 0.5
        pred = model(cate_x, cont_x, mask)
        loss = F.mse_loss(pred.view(-1), y.view(-1))
        
        # record loss
        losses.update(loss.item(), batch_size)
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:      
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1    

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            # record accuracy
            pred_y = (1-k)*pred[:, 0] + (k)*compute_acc_gp(pred[:, 1:])
            pred_y = (pred_y+0.5).int()
            pred_y[pred_y > 3] = 3
            
            kappa_score = bowl_utils.qwk3(pred_y.detach().cpu().numpy(), y[:, 0].cpu().numpy())        
            accuracies.update( kappa_score, batch_size)
        
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Acc: {acc.val:.4f}({acc.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  'sent/s {sent_s:.0f} '
                  .format(
                   epoch, step, len(train_loader), batch_time=batch_time,                   
                   data_time=data_time, loss=losses,
                   acc=accuracies,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   lr=scheduler.get_lr()[0],
                   #lr=scheduler.optimizer.param_groups[0]['lr'],
                   sent_s=sent_count.avg/batch_time.avg
                   ))
    return losses.avg, accuracies.avg


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
            loss = F.mse_loss(pred.view(-1), y.view(-1))

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
    predictions = torch.cat(predictions).numpy()
    groundtruth = torch.cat(groundtruth).numpy()
    
    try:
        optR = bowl_utils.OptimizedRounder()
        optR.fit(predictions, groundtruth)
        coefficients = optR.coefficients()
        print(coefficients)
        predictions[predictions < coefficients[0]] = 0
        predictions[(coefficients[0]<=predictions)&(predictions< coefficients[1])] = 1
        predictions[(coefficients[1]<=predictions)&(predictions< coefficients[2])] = 2
        predictions[(coefficients[2]<=predictions)] = 3

        kappa_score = bowl_utils.qwk3(predictions, groundtruth)
    except:
        kappa_score = 0
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
