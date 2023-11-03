 
from __future__ import print_function, absolute_import
from ast import Return
import os
import sys 
import pdb
import argparse
import operator
import logging
import numpy as np
import os.path as osp
from datetime import datetime
import time
from collections import OrderedDict
from collections import defaultdict
from typing import Optional, Callable, Dict, List, Tuple

from sklearn.metrics import f1_score,confusion_matrix,roc_curve, auc,roc_auc_score 
from sklearn.preprocessing import label_binarize 

import copy
# torch
import torch
from torch import optim 
import torch.distributed as dist
from torch import nn 
from torchvision import transforms as T
from yacs.config import CfgNode

# custom
import utils
from models import create_model 
from utils import create_scheduler, set_seed, adjust_learning_rate, set_bn_track_running_stats 
from utils import  AverageMeter,SupConLoss,SupConLoss_adv, FormatterNoInfo 

from config.default_config import get_cfg
from dataset import   get_dataloader_from_hdf5,Normalize_layer,get_external_dataloader_from_hdf5
from vis import Visualizer,plot_confusion_matrix 
from vis import  sensitivity_specificity_multiclass
from einops import rearrange
# timm
import timm
from timm.utils import   get_outdir, CheckpointSaver, update_summary, accuracy 
from timm.optim import create_optimizer_v2 as create_optimizer
from timm.models import load_checkpoint 
   
has_apex = False 
has_native_amp = False  
_logger = logging.getLogger('train') 
best_prec1 = 0

  
def main(cfg: CfgNode ):
     
    """Get default settings""" 
    output_dir = get_outdir( cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else './output/train', cfg.EXP_NAME) 
    default_level=logging.INFO
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    log_path = osp.join(output_dir, 'log.txt')
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)  
    # global best_prec1
    if utils.is_master_rank(cfg):
        _logger.info(cfg)
    _logger.info('is_master_rank(cfg): {}'.format(utils.is_master_rank(cfg)))
  
    """Build Models""" 
    model = create_model(cfg.MODEL.NAME, cfg.MODEL).to(cfg.DEVICE) 
    model.cuda()  
 
    if utils.is_master_rank(cfg): 
        _logger.info( 'Model %s created, param count: %d' %
            (cfg.MODEL.NAME, sum([m.numel() for m in model.parameters()])))
    

    """Build data loader"""
    if cfg.Test != "external":
        train_loader, val_loader, test_loader = get_dataloader_from_hdf5(hdf5_path=cfg.DATA.DIR, cfg = cfg.DATA) 
    else:
        test_loader = get_external_dataloader_from_hdf5(hdf5_path=cfg.DATA.DIR, cfg = cfg.DATA) 
    f1_type = 'macro' 
    tumor_idx = ['choroid', 'ependymoma', 'glioma', 'mb']
    
    """Define loss function (criterion)"""
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    """Validate"""  
    if cfg.EVAL_CHECKPOINT  : 
        try: 
            load_checkpoint(model, cfg.EVAL_CHECKPOINT, use_ema=False,strict=False) 
        except RuntimeError as e:
            print('---------------------------------------------------------------------main---------')
            print('Ignoring "' + str(e) + '"') 
        eval_metrics_val = validate(model, test_loader, validate_loss_fn, f1_type, tumor_idx,cfg)
        if cfg.Test == "internal" or cfg.Test == "external":  return 0 
    
    
    """Build Optimizers"""  
    optimizer = create_optimizer(model, cfg.TRAIN.OPTIM,  
                                 weight_decay = cfg.TRAIN.WEIGHT_DECAY,momentum = cfg.TRAIN.MOMENTUM)
                     
    """ DataParallel """ 
    model = torch.nn.DataParallel(model).cuda() 
    
    """Setup Logger, Saver and Visualizer"""
    eval_metric = cfg.EVAL_METRIC
    best_metric = None
    best_epoch = None
    saver = None
    viser = None
    if utils.is_master_rank(cfg):
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(model=model,
                                optimizer=optimizer,
                                args=None,
                                model_ema=None,
                                amp_scaler=None,
                                checkpoint_dir=output_dir,
                                recovery_dir=output_dir,
                                decreasing=decreasing,
                                max_history=cfg.CHECKPOINT_HISTORY)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())
        viser = Visualizer(use_tb=cfg.DEBUG.VIS_USE_TB, root=output_dir)

    """Scheduler""" 
    lr_scheduler, num_epochs = create_scheduler(
            optimizer,
            cfg.TRAIN.EPOCHS,
            sched=cfg.TRAIN.SCHED,
            min_lr=cfg.TRAIN.MIN_LR,
            decay_rate=cfg.TRAIN.DECAY_RATE,
            decay_epochs=cfg.TRAIN.DECAY_EPOCHS,
            decay_epochs_list=cfg.TRAIN.DECAY_EPOCHS_LIST,
            warmup_lr=cfg.TRAIN.WARMUP_LR,
            warmup_epochs=cfg.TRAIN.WARMUP_EPOCHS) 
    start_epoch = 0
    best_f1_macro = 0 
    best_acc = 0 
   
    """Training Loops"""
 
    adv_image = None  
    for epoch in range(start_epoch, num_epochs):
        # Medulloblastoma was sampled every epoch to ensure a balanced train set 
        train_metrics  = train_one_epoch(epoch,
                                        model,
                                        train_loader, #val_loader, #
                                        optimizer,
                                        train_loss_fn,
                                        cfg,
                                        lr_scheduler=lr_scheduler,
                                        saver=saver, 
                                        adv_image = adv_image)  
        for param_group in optimizer.param_groups:
            print('Adjusted learning rate to {:.5f}'.format(param_group['lr']))
  
        eval_metrics_val = validate(model, val_loader, validate_loss_fn, f1_type, tumor_idx,cfg)
  
        for param_group in optimizer.param_groups:
            print('Adjusted learning rate to {:.5f}'.format(param_group['lr']))
        if output_dir is not None:
            update_summary(epoch, train_metrics, eval_metrics_val, os.path.join(output_dir, 'summary.csv'),
                            write_header=best_metric is None)

        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = eval_metrics_val[eval_metric]
            best_metric, best_epoch = saver.save_checkpoint(
                epoch, metric=save_metric)
            if eval_metrics_val['f1-macro'] > best_f1_macro:
                best_f1_macro = eval_metrics_val['f1-macro']
                best_f1_macro_epoch = epoch

            if eval_metrics_val['acc'] > best_acc:
                best_acc = eval_metrics_val['acc']
                best_acc_epoch = epoch
 
        if viser is not None:
            viser.visualize(epoch, train_metrics, eval_metrics_val)
    
    
    """Testing """
    _logger.info('now testing!')
    best_model_path = os.path.join(output_dir, 'model_best.pth.tar')
    
    load_checkpoint(model.module, best_model_path, use_ema=False)

    eval_metrics_test = validate(model, test_loader, validate_loss_fn, f1_type, tumor_idx,cfg)
    plot_confusion_matrix(model,test_loader,title= 'test', labels_name=tumor_idx,save_path=output_dir,cfg=cfg)
 
    store=cfg.EXP_NAME+':*** test AUC: {0}(epoch {1})'.format(
            round(eval_metrics_test['auroc'],2),    best_epoch, '.2f')+'test  f1-macro : {0} (epoch {1})'.format(
            round(eval_metrics_test['f1-macro'],2), best_f1_macro_epoch, '.2f')+'test acc  : {0} (epoch {1})'.format(
            round(eval_metrics_test['acc'],2),      best_acc_epoch, '.2f') + '*** val acc  : {0} (epoch {1})'.format(
            round(best_acc,2), best_acc_epoch, '.2f')
    print(store) 
    os.system("rm -r "+os.path.join(output_dir, 'checkpoint*'))
    os.system("rm -r "+os.path.join(output_dir, 'last.pth.tar'))
    logging.root.removeHandler(file_handler) 


   
def validate(model, loader, loss_fn, f1_type,tumor_idx,cfg,print_predicting_wrong=False) -> OrderedDict:
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    accs_m = AverageMeter() 
    model.eval() 
    last_idx = len(loader) - 1
    target_all = []
    pred_all = []
    patient_all = []
    wrong = defaultdict(list) 
    with torch.no_grad():
        for batch_idx, (input, num, target,xlsx) in enumerate(loader):
            # print(num) 
            # pdb.set_trace()
            last_batch = batch_idx == last_idx
            
            input = input.cuda()
            target = target.cuda()
            # pdb.set_trace()
            
            output = model(input)   
            if isinstance(output, (tuple, list)):
                output = output[0]  
            loss = loss_fn(output, target)
            acc = accuracy(output, target)[0]
                 
            pred = torch.argmax(output,1) 
            mm = torch.nn.Softmax(dim=1)
            output_softmax = mm(output)
            if not batch_idx:
                output_softmax_all = output_softmax
            else:
                output_softmax_all = torch.cat((output_softmax_all, output_softmax), 0)
             
            target_all.extend(target.cpu().numpy())
            pred_all.extend(pred.cpu().numpy())
            patient_all.extend(num)
            torch.cuda.synchronize()
            
            losses_m.update(loss.data.item(), input.size(0))
            accs_m.update(acc.item(), output.size(0)) 
            
            if utils.is_master_rank(cfg) and (last_batch or
                                              batch_idx % cfg.PRINT_FREQ == 0):
                log_name = 'Test'
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc: {acc.val:>7.4f} ({acc.avg:>7.4f})'.format(
                        log_name,
                        batch_idx,
                        last_idx, 
                        batch_time=batch_time_m,
                        loss=losses_m,
                        acc=accs_m))  
        wrong_sample = []
        true_sample = []
        print(target_all)
        for i in range(len(target_all)):
            if target_all[i] != pred_all[i]: 
                #wrong_sample.append(tumor_idx[target_all[i]]+"_"+patient_all[i]+"_to_"+tumor_idx[pred_all[i]])
                wrong_sample.append( patient_all[i] )
            else:
                # true_sample.append(tumor_idx[target_all[i]]+"_"+patient_all[i]+"_to_"+tumor_idx[pred_all[i]])
                true_sample.append( patient_all[i] )
        print(wrong_sample)
        print("true_sample:")
        print(true_sample)
        # Compute sensitivity and specificity
        sensitivity_list, specificity_list, mean_sensitivity, mean_specificity = sensitivity_specificity_multiclass(target_all,pred_all)
        
        # Compute F1 Scores
        f1_new_macro = f1_score(target_all,pred_all,average = f1_type)
        print(tumor_idx)
        
        # Compute AUC
        target_all_binarize = label_binarize(target_all, classes=torch.arange(0,len(tumor_idx))) 
        auroc_macro = roc_auc_score(target_all_binarize,output_softmax_all.cpu(),average="macro")
         
    show_list = [('loss', losses_m.avg), ('acc', accs_m.avg),('f1-macro',f1_new_macro),('auroc',auroc_macro)]
    print("sensitivity: ",mean_sensitivity, " specificity:",mean_specificity, " F1:", f1_new_macro, " AUC:", auroc_macro, " ACC:", accs_m.avg)
    
    metrics = OrderedDict(show_list) 
    return metrics
 
def train_one_epoch(epoch,
                    model,
                    loader,
                    optimizer,
                    loss_fn,
                    cfg,
                    lr_scheduler=None,
                    saver=None, 
                    adv_image = None) -> OrderedDict: 
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses0_m = AverageMeter()
    losses1_m = AverageMeter()
    losses2_m = AverageMeter() 
    accs_m = AverageMeter()
    #pdb.set_trace()
      
    model.train()
    criterion = nn.CrossEntropyLoss()
    contrast_criterion = SupConLoss(temperature=cfg.temperature,contrast_mode='one')
    contrast_criterion_adv = SupConLoss_adv(temperature = cfg.temperature,contrast_mode='one')
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    MSE_loss = torch.nn.MSELoss()  
    for batch_idx, (input, num,  target,xlsx) in enumerate(loader): 
         
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
 
        
        if cfg.method == "coca": 
            adv_image = Inconsistentcy_Instantiation(xlsx,target, model,times=cfg.times,epsilon = cfg.epsilon  ,cfg = cfg)
              
         
        input, target = input.cuda(), target.cuda()
        loss_entro,loss_contrast,loss1_entro,loss1_contrast = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0) 
        loss2_entro, loss2_contrast = torch.tensor(0) ,torch.tensor(0) 
        
                  
        if cfg.method == "coca":  
            output, feature_vector = model( input, adv = cfg.adv ) 
        else:   
            output, feature_vector = model( input  ) 

 
        
        loss_contrast = contrast_criterion(feature_vector.unsqueeze(1), target )  if cfg.method == "coca" else torch.tensor(0) 
        
        loss_entro = loss_fn(output, target)  
        loss = loss_contrast * 0.1 + loss_entro 
        if adv_image != None: 
            
            set_bn_track_running_stats(model, track_running_stats = False) 
            input = adv_image["adv1"].cuda() 
            output_adv1, feature_vector1 = model(input, adv = cfg.adv)
            input = adv_image["adv2"].cuda() 
            output_adv2, feature_vector2 = model(input, adv = cfg.adv)
            set_bn_track_running_stats(model, track_running_stats = True) 
            
            loss1_entro = loss_fn(output_adv1, target)  
            loss2_entro = loss_fn(output_adv2, target)   
            loss1_contrast = contrast_criterion_adv(feature_vector1.unsqueeze(1),feature_vector.detach().data.unsqueeze(1),target)
            loss2_contrast =  contrast_criterion_adv(feature_vector2.unsqueeze(1),feature_vector.detach().data.unsqueeze(1),target) 
            loss1 = loss1_contrast* cfg.lamda[1] + loss1_entro* cfg.lamda[0]
            
            
             
            loss2 = loss2_contrast * cfg.lamda[3] + loss2_entro* cfg.lamda[2] 
            loss = loss  + loss1 * cfg.beta12 + loss2 * cfg.beta21 
         
        acc = accuracy(output, target)[0]
              
             
        if not cfg.DIST:
            losses_m.update(loss.item(), input.size(0))
            losses0_m.update(loss_entro.item(), input.size(0))
            losses1_m.update(loss1_entro.item(), input.size(0))
            losses2_m.update(loss2_entro.item(), input.size(0)) 
            accs_m.update(acc.item(), input.size(0))
            
        
        optimizer.zero_grad()
        loss.backward()
        if cfg.TRAIN.CLIP_GRADIENT is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),  cfg.TRAIN.CLIP_GRADIENT)
        optimizer.step()  
        
        torch.cuda.synchronize()
        num_updates += 1 
        
        if last_batch or batch_idx % cfg.PRINT_FREQ == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)  
            if utils.is_master_rank(cfg):
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Loss1: {loss1.val:>9.6f} ({loss1.avg:>6.4f})  '
                    'Loss2: {loss2.val:>9.6f} ({loss2.avg:>6.4f})  ' 
                    'Acc: {acc.val:>7.4f} ({acc.avg:>7.4f})  ' 
                    'LR1: {lr1:.3e}  '
                    'LR2: {lr2:.3e}  ' 
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx,
                        len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        loss1=losses1_m,
                        loss2=losses2_m,
                        acc=accs_m, 
                        lr1=lr, 
                        lr2=lr, 
                        data_time=data_time_m)) 
        if saver is not None and cfg.RECOVERY_FREQ and (
                last_batch or (batch_idx + 1) % cfg.RECOVERY_FREQ == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates,
                                     metric=losses_m.avg)
        else:
            adjust_learning_rate(cfg, optimizer, epoch)
        end = time.time()
    
    return OrderedDict([('loss', losses_m.avg),('loss0', losses0_m.avg),( 'loss1', losses1_m.avg),( 'loss2', losses2_m.avg)   ]) 

def Inconsistentcy_Instantiation(xlsx,target, model,   times = 3,   epsilon=0.05, alpha = 0.02  ,  cfg =None):
    loss_fn = nn.CrossEntropyLoss().cuda()   
    contrast_criterion_adv = SupConLoss_adv(temperature = cfg.temperature,contrast_mode='one')
    model.eval()
    adv_image = {} 
    k = times
    alpha = epsilon / (k - 1) 
    j = 1 
    eps =    epsilon
    input, target = xlsx["orig_image"], target.cuda() 
    orig_images = xlsx["orig_image"].cuda().data 
    for name,para in model.named_parameters(): 
        para.requires_grad = False
    
    output_clean, feature_vector_clean = model( Normalize_layer(xlsx["orig_image"].clone().cuda(),cfg.mean,cfg.std) , adv = cfg.adv) 
 
    for j in range(2):
        img_x = xlsx["orig_image"].clone().cuda()   
        for i in range( k ): 
            img_x.requires_grad = True  
            output, feature_vector = model( Normalize_layer(img_x,cfg.mean,cfg.std) , adv = cfg.adv)   
            loss1 = loss_fn(output, target)   
             
            loss2 = contrast_criterion_adv(feature_vector.unsqueeze(1),feature_vector_clean.detach().data.unsqueeze(1),target) 
            if j == 0:
                loss = loss1* cfg.lamda[0] - loss2 * cfg.lamda[1]
            else:
                loss = loss2 * cfg.lamda[3] - loss1 * cfg.lamda[2]   
            loss.backward() 
            img_x.data = img_x.data + alpha * img_x.grad.data.sign() 
            img_x.data = torch.where(img_x.data > (orig_images + eps), orig_images + eps, img_x.data)
            img_x.data = torch.where(img_x.data < (orig_images - eps), orig_images - eps, img_x.data)
            
            img_x.data = torch.clamp(img_x.data, min=0, max=1)
            img_x.grad.data = torch.zeros(img_x.shape).cuda()
            
            img_x = img_x.detach() 
        adv_image["adv"+str(j+1)] =  Normalize_layer(img_x.cpu(),cfg.mean,cfg.std)
    
    for name,para in model.named_parameters(): 
        para.requires_grad = True  
    model.train() 
    return adv_image 
   


def load_config(args ):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup codebase default cfg.
    cfg = get_cfg()
     
    # Load config from cfg_file (load the configs that vary accross datasets).
    
    # if args.cfg_file is not None:
    #     cfg.merge_from_file(args.cfg_file)
    
    if args.opts is not None:
        cfg.merge_from_list(args.opts) 
    # 读取 YAML 文件 
    import yaml
    with open(cfg.yaml_path, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    
    for key, value in yaml_cfg.items():
        
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                cfg[key][subkey] = subvalue
        else:
            cfg[key] = value
    # Load config from command line, overwrite config from opts (for the convenience of experiemnts).
    if args.opts is not None:
         
        cfg.merge_from_list(args.opts) 
    # pdb.set_trace()
    cfg.DATA.multi_classification = cfg.multi_classification
    cfg.MODEL.multi_classification = cfg.multi_classification
    return cfg


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser( description="PyTorch implementation of XNN") 
    # CUSTOMIZED  
    parser.add_argument(
        "opts",
        help="See config/default_config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,#
    )
    args = parser.parse_args()
    cfg = load_config(args) 
    set_seed(cfg.manual_seed)  
    main(cfg ) 
