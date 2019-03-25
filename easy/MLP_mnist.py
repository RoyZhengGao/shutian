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
from tensorflow.examples.tutorials.mnist import input_data

from numpy.random import RandomState
import numpy as np

from MLP_tf_model import Config,CitationRecNet 

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
    parser.add_argument('--path', default='/tmp/data/', help='data path')
    parser.add_argument('--layer1-dim', type=int, default=100, help='layer1 dimension')
    parser.add_argument('--layer2-dim', type=int, default=10, help='layer2 dimension')
    parser.add_argument('--learning-rate', type=float, default=0.01, help=' ')
    parser.add_argument('--epoch', type=int, default=50, help=' ')
    parser.add_argument('--batch-size', type=int, default=8, help=' ') 

    args = parser.parse_args()
    return args

def run(args):
    config = Config(args)  # get all configurations

    mnist = input_data.read_data_sets(args.path, one_hot=False)
    # print(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels)

    #load data
    X_train = mnist.train.images
    X_test = mnist.test.images

    Y_train = mnist.train.labels
    Y_test = mnist.test.labels
    
    print("check data:",type(X_train),type(Y_train),len(X_train),len(Y_train),len(X_train[0]),len(Y_train[0]))

    BATCH_SIZE = args.batch_size
    DATA_NUM = len(Y_train)
    STEPS = DATA_NUM // BATCH_SIZE + 1  # STEPS = number of batches
    x_dim = len(X_train[0])
    y_dim = len(Y_train[0])

    with tf.Graph().as_default(),tf.Session() as sess:
         # 问题，MLP_tf_model里的config是不是这里就没用了？
        model = CitationRecNet(config.LAYER1_DIM,
                            config.LAYER2_DIM,
                            x_dim,
                            y_dim, 
                            config.LEARNING_RATE)

        init_op = tf.global_variables_initializer()
        sess.run(init_op) 

        for i in range(config.EPOCH):
            for j in range(STEPS):
                start = (j*BATCH_SIZE) % DATA_NUM
                end = ((j+1)*BATCH_SIZE) % DATA_NUM
                # 问题 为什么这里也有keep_pro这个参数？
                # 问题 这里的model.y_指的是训练集对应的label呗

                #need to re-orgnize the training data, especially for the last batch data. 
                if end > start:
                    train_X = X_train[start:end]
                    train_Y = Y_train[start:end]
                else:
                    #need to concatenate the last part and the beginning part of the training data  
                    train_X = np.concatenate((X_train[start:],X_train[:end]),axis=0) 
                    train_Y = np.concatenate((Y_train[start:],Y_train[:end]),axis=0) 
                sess.run(model.train_op, feed_dict={model.x:train_X, model.y:train_Y})
                
                if j % 100 == 0:
                    
                    y_pred, total_cross_entropy = sess.run((model.y_pred_softmax, model.loss), feed_dict={model.x: train_X, model.y: train_Y})
                    print("After %d training step(s), cross entropy on all data is %g" % (j, total_cross_entropy))
                    print("training y and real y difference:", train_Y[0:1], y_pred[0:1])
            # pre_Y = sess.run(y, feed_dict={x: X_test, keep_prob: 1.0})
            # for pred, real in zip(pre_Y, Y_test):
            #     print(pred, real)

if __name__ == '__main__':
    args = parse_args()
    run(args)
    print("Run the command line:\n" \
    "--> tensorboard --logdir=/Users/zhenggao/Desktop/alibaba/cross_domain_recommendation/code/deep_model/log/ " \
    "\nThen open http://0.0.0.0:6006/ into your web browser")