#!/bin/bash


python /code/KBS/coca.py  OUTPUT_DIR            "/output/coca"    \
                        OUTPUT_RESULT           "/output/coca/result.txt"  \
                        method                  'coca'  \
                        yaml_path               '/code/tumor/config/default.yaml' \
                        EVAL_CHECKPOINT         "/data/sunch/output/Experiment/adv_training_b_0_1_repeat2_e0025-1e-4/seed1/seed-1-fold-5/model_best.pth.tar"\
                        DATA.DIR                '/data/sunch/output/dataset/Internal.hdf5' \
                        DATA.BATCH_SIZE         64             \
                        DATA.TEST_BATCH_SIZE    64             \
                        TRAIN.BASE_LR           1e-4            \
                        TRAIN.PRE_LR            1.0             \
                        TRAIN.EPOCHS            200               \
                        beta12  0.1     beta21      0.1             \
                        times   3       epsilon     0.025    lamda [1.0,1.0,1.0,1.0]   
        
 