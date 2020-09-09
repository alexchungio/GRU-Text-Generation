#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cfgs.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/9 下午5:05
# @ Software   : PyCharm
#-------------------------------------------------------

from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

# ------------------------------------------------
# VERSION = 'FPN_Res101_20181201'
VERSION = 'GRU_Text_Generation_20200909'
NET_NAME = 'gru_text_generation'


# ---------------------------------------- System_config----------------------------
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "4"
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 10000

SUMMARY_PATH = ROOT_PATH + '/outputs/summary'
INFERENCE_SAVE_PATH = ROOT_PATH + '/outputs/inference_results'
TEST_SAVE_PATH = ROOT_PATH + '/outputs/test_results'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/outputs/inference_image'
# INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

TRAINED_CKPT = os.path.join(ROOT_PATH, 'outputs/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/outputs/evaluate_result'

#----------------------Data---------------------
DATASET_PATH = os.path.join(ROOT_PATH, 'data', 'shakespeare.txt')


#------------------------network config--------------------------------
BATCH_SIZE = 32

MAX_LENGTH = 500 # the number in singe time dimension of a single sequence of input data
FEATURE_SIZE = 10000
EMBEDDING_SIZE = 100

# NUM_UNITS = [128, 64, 32]
NUM_UNITS = [32, 16]
NUM_LAYERS = 2

#-------------------------train config-------------------------------
EMBEDDING_TRANSFER = False
LEARNING_RATE = 0.01
NUM_EPOCH = 10
KEEP_PROB = 0.8

# data
SPLIT_RATIO = 0.2