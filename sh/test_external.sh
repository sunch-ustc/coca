#!/bin/bash

python /code/KBS/coca.py  OUTPUT_DIR            "/output/coca"    \
                        method                  'coca'            \
                        yaml_path               '/code/tumor/config/default.yaml' \
                        EVAL_CHECKPOINT         "/data/sunch/output/Experiment/adv_training_b_0_1_repeat2_e0025-1e-4/seed1/seed-1-fold-5/model_best.pth.tar"\
                        DATA.DIR                '/data/sunch/output/dataset/external.hdf5' \
                        DATA.BATCH_SIZE         10             \
                        DATA.TEST_BATCH_SIZE    10             \
                        Test                    "external"
 
        
 