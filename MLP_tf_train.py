#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-10 13:26
@desc: 
"""
from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf

from numpy.random import RandomState
import numpy as np

from MLP_tf_model import Config,CitationRecNet 

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
    parser.add_argument('--path', default='./data/', help='data path')
    parser.add_argument('--layer1-dim', type=int, default=100, help='layer1 dimension')
    parser.add_argument('--layer2-dim', type=int, default=10, help='layer2 dimension')
    parser.add_argument('--learning-rate', type=float, default=0.01, help=' ')
    parser.add_argument('--epoch', type=int, default=50, help=' ')
    parser.add_argument('--batch-size', type=int, default=8, help=' ')
    parser.add_argument('--regularization-rate', type=float, default=0.01, help=' ')
    parser.add_argument('--log', default="./log/", help=' ')
    parser.add_argument('--saved-model', default="./saved_model/", help=' ')  
    parser.add_argument('--max-grad-norm', type=float, default=5.0, help=' ')
    parser.add_argument('--grad-clip', dest='grad_clip', action='store_true') 
    parser.add_argument('--no-grad-clip', dest='grad_clip', action='store_false') 
    parser.set_defaults(grad_clip=True)
    parser.add_argument('--batch-norm', dest='batch_norm', action='store_true') 
    parser.add_argument('--no-batch-norm', dest='batch-norm', action='store_false') 
    parser.set_defaults(batch_norm=True)

    args = parser.parse_args()
    return args

def run(args):
    config = Config(args)  # get all configurations

   

    #load data
    X_train = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/mlp-train-ci-vec')
    X_test = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/mlp-test-ci-vec')

    Y_train = []
    with open('/Volumes/SaveMe/data/2019/mlp/train-label', 'r') as f:
        for line in f:
            Y_train.append([int(line.strip())])
    Y_test = []
    with open('/Volumes/SaveMe/data/2019/mlp/test-label', 'r') as f:
        for line in f:
            Y_test.append([int(line.strip())])

    BATCH_SIZE = args.batch_size
    DATA_NUM = len(Y_train)
    STEPS = DATA_NUM // BATCH_SIZE + 1  # STEPS = number of batches
    x_dim = len(X_train[0])
    y_dim = len(Y_train[0])

    with tf.Graph().as_default(),tf.Session() as sess:
         # 问题，MLP_tf_model里的config是不是这里就没用了？
        model = MLP_tf_model(config.LAYER1_DIM,
                            config.LAYER2_DIM,
                            x_dim,
                            y_dim,
                            GRAD_CLIP,
                            config.LEARNING_RATE,
                            config.IS_BATCH_NORM)

        #If you use batch nomalization, you have to save model in this way
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        saver = tf.train.Saver(var_list=var_list,max_to_keep=10)#saver for checkpoints, add var_list because of batching training        
        #saver = tf.train.Saver(max_to_keep=10)#saver for checkpoints #for no batch nomalization

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(config.log, graph=tf.get_default_graph()) 

        for i in range(config.epoch):
            for j in range(STEPS):
                start = (i*BATCH_SIZE) % DATA_NUM
                end = ((i+1)*BATCH_SIZE) % DATA_NUM
                # 问题 为什么这里也有keep_pro这个参数？
                # 问题 这里的model.y_指的是训练集对应的label呗
                if end > start:
                    train_X = X_train[start:end]
                    train_Y = Y_train[start:end]
                else:
                    #need to concatenate the last part and the beginning part of the training data  
                    train_X = np.concatenate((X_train[start:],X_train[:end]),axis=0) 
                    train_Y = np.concatenate((Y_train[start:],Y_train[:end]),axis=0) 
                sess.run(model.train_step, feed_dict={model.x:train_X, model.y:train_Y, 
                                                    model.regularization_rate: config.REGULARIZATION_RATE,
                                                    model.max_grad_norm: config.MAZ_GRAD_NORM, model.dropout_keep:config.DROPOUT_KEEP})
                
                # write to tensorboard
                summary_writer.add_summary(merged_summary_op) 
                
                if j % 100 == 0:
                    #save model to local
                    model_path = args.saved_model+"/epoch_%s-batch_%s.ckpt" % (i,j)
                    saver.save(sess, model_path)

                    y_pred, total_cross_entropy = sess.run((model.y, model.loss), feed_dict={model.x: train_X, model.y: train_Y})
                    print("After %d training step(s), cross entropy on all data is %g" % (j, total_cross_entropy))
                    print("training y and real y difference:", train_Y[0:2], y_pred[0:2])
            # pre_Y = sess.run(y, feed_dict={x: X_test, keep_prob: 1.0})
            # for pred, real in zip(pre_Y, Y_test):
            #     print(pred, real)

if __name__ == '__main__':
    args = parse_args()
    run(args)
    print("Run the command line:\n" \
    "--> tensorboard --logdir=/Users/zhenggao/Desktop/alibaba/cross_domain_recommendation/code/deep_model/log/ " \
    "\nThen open http://0.0.0.0:6006/ into your web browser")