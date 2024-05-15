#!/usr/bin/env python

# -*- encoding: utf-8

'''
@file: config.py
@time: @Time : 4/26/21 2:56 PM 
@desc： 
               
'''


class Config(object):
    apr_dir = '/home/btr/bpmn/NER/model/'
    data_dir = '/home/btr/bpmn/NER/data/李文鑫data/'
    model_name = 'model_3.pt'
    epoch = 5
    bert_model = '/home/btr/bpmn/NER/model/bert-base-uncased'
    lr = 5e-5
    eps = 1e-8
    batch_size = 16
    mode = 'prediction'  # for prediction mode = "prediction"
    training_data = 'train.csv'
    val_data = 'dev.csv'
    test_data = 'prediction.csv'
    use_data = 'prediction.csv'
    test_out = 'test_prediction.csv'
    use_out = 'use_prediction.csv'
    raw_prediction_output = 'raw_prediction.csv'
