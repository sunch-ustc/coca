#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-21 16:50
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-26 16:57
Description        : Default Configurations
-------- 
Copyright (c) 2021 Multimedia Group USTC. 
'''
from yacs.config import CfgNode
from typing import Optional, Union, Callable, Dict, List, Tuple


def _assert_in(opt, opts):
    assert (opt in opts), '{} not in options {}'.format(opt, opts)
    return opt


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode() 
_C.method = 'coca'  # baseline or coca
_C.Test = ''
_C.yaml_path = '/code/tumor/config/default.yaml'
_C.multi_classification = []     
_C.ROOT_LOG = 'log'  # log dir name
_C.ROOT_MODEL = 'checkpoint'  # model dir name
_C.MODEL_SUFFIX = ''  # model suffix mark
_C.PRINT_FREQ = 20  # print every $PRINT_FREQ$ steps.
_C.EVAL_FREQ = 1  # print every $EVAL_FREQ$ epochs.
_C.NUM_GPUS = 1
_C.DIST = False  # Use DistributedDataParallel (DDP) 
_C.SHARD_ID = 0
_C.NUM_SHARDS = 2
_C.INIT_METHOD = 'tcp://localhost:9997'
_C.DEVICE = 'cuda'  
_C.EVAL_CHECKPOINT = ''
_C.EVAL_METRIC = 'acc' # Optional: 'auroc'  'acc'  'f1-macro' 'loss'
_C.EXP_NAME = 'radio-40-2' #'DEBUG'
_C.OUTPUT_DIR = './output/'
_C.OUTPUT_RESULT = './result.txt'
_C.CHECKPOINT_HISTORY = 1  #the number of file storing .pth.tar
_C.RECOVERY_FREQ = 0 #10000
_C.manual_seed = 0
_C.mark = ''
_C.attention_tumor = 0.0
_C.lamda = [1.0,1.0,1.0,1.0]
_C.adv = "contrast"
_C.times = 3
_C.epsilon = 0.025
_C.temperature = 1.0
_C.mean = [34.66, 36.28, 26.8]
_C.std = [73.44, 75.56, 57.0]
_C.beta12 = 1.0
_C.beta21 = 1.0
# -----------------------------------------------------------------------------
# DATA options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.DIR = './data/med_epe-Ax-T1_T1_E_T2_nobox.hdf5'
_C.DATA.multi_classification = [] 
_C.DATA.nii_file_path = ''
_C.DATA.interval_use= False
_C.DATA.interval= [0, 0, 0, 0, 4, 12]#[1, 100, 181, 211, 0, 60]
_C.DATA.BATCH_SIZE = 64
_C.DATA.TEST_BATCH_SIZE = 64
_C.DATA.VAL_BATCH_SIZE = 32
_C.DATA.NUM_WORKERS = 0
# _C.DATA.MEAN = [0.485, 0.456, 0.406]
# _C.DATA.STD = [0.229, 0.224, 0.225]
_C.DATA.TEST_RATIO = 0.2
_C.DATA.NUM_FRAMES = 6#  6
_C.DATA.SAMPLE_STEP = 1#5
_C.DATA.SEED = 1
_C.DATA.TEST_SEED=1
_C.DATA.MODALITY = ['T1_Ax','T1_E_Ax','T2_Ax'] 
_C.DATA.TUMOR =['choroid','ependymoma','glioma','mb']
_C.DATA.TRAIN_BALANCE = True
_C.DATA.test_balance=False
_C.DATA.mb_sample = 100
_C.DATA.fast_test=True
_C.DATA.fast_test_start_index=1
_C.DATA.num_frames_val=6
_C.DATA.sample_step_val=1
_C.DATA.binary_classification_suimu=False
_C.DATA.binary_four_classification=False
_C.DATA.patient_num=2000
_C.DATA.binary_classification_suimu=False
_C.DATA.mb_num  = 450 
_C.DATA.fine_grained=False
_C.DATA.just_test=False
_C.DATA.trans = 0
_C.DATA.mask = 0
_C.DATA.molecular = False
_C.DATA.mark = ""
# -----------------------------------------------------------------------------
# TRAIN options 
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()
# optimizer
_C.TRAIN.OPTIM = 'adam'# 'adam'
_C.TRAIN.BASE_LR = 0.0001
_C.TRAIN.PRE_LR = 1.0
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.CLIP_GRADIENT = 20.0
_C.TRAIN.WORKERS = 0
_C.TRAIN.TUNING = False
_C.TRAIN.FREEZE_BN = False
_C.TRAIN.SYNC_BN = True
_C.TRAIN.TUNE_FROM = ''

# save & load
_C.TRAIN.RESUME = ''
_C.TRAIN.RESUME_OPTIM = True

# scheduler
_C.TRAIN.EPOCHS = 200
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.SCHED = 'multistep' #cosine multistep
_C.TRAIN.MIN_LR = 1e-4
_C.TRAIN.DECAY_RATE = 0.3
_C.TRAIN.DECAY_EPOCHS = 300
_C.TRAIN.DECAY_EPOCHS_LIST = [ ]
_C.TRAIN.WARMUP_LR = 0.0001
_C.TRAIN.WARMUP_EPOCHS = 0

# -----------------------------------------------------------------------------
# MODEL options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'resnet18'
_C.MODEL.subnet = ''
_C.MODEL.multi_classification = [] 
_C.MODEL.NUM_BLOCKS = [3, 3, 3]
_C.MODEL.NUM_CLASSES = 4
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.GROUPS      = 0
_C.MODEL.rgb_trans = False
_C.MODEL.rgb_trans_lr = 0.0001
_C.MODEL.cut_region = 1
_C.MODEL.region_skip = 0
_C.MODEL.region_attention_heads = 2
_C.MODEL.FREEZE      = []  # "layer3.1.conv1.weight"
_C.MODEL.outpooling = "avg"
_C.MODEL.mark = '' 
# -----------------------------------------------------------------------------
# DEBUG options
# -----------------------------------------------------------------------------
_C.DEBUG = CfgNode()
_C.DEBUG.VIS_USE_TB = True


def _assert_and_infer_cfg(cfg):
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())