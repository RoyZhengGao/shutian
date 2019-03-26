#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-10 13:25
@desc: 
"""
from __future__ import print_function
# import sys
# sys.path.append("..") #if you want to import python module from other folders, 
                        #you need to append the system path
import tensorflow as tf
from numpy.random import RandomState
import numpy as np


class Config(object):
    def __init__(self,args):
        # 这里定义的参数我用了大写
        self.LAYER1_DIM = args.layer1_dim
        self.LAYER2_DIM = args.layer2_dim
        self.LEARNING_RATE = args.learning_rate
        self.EPOCH = args.epoch
        self.BATCH_SIZE = args.batch_size

class CitationRecNet(object):
    def __init__(self, layer1_dim, layer2_dim, x_dim, 
                y_dim,learning_rate):
        
        #in order to generate same random sequences 
        tf.set_random_seed(1)

        """
        input parameter
        """ 
        # 问题： 等式右边的是不是上面那行init后面括号里的参数
        self.layer1_dim = layer1_dim #否则传不到MLP() 里面，要不然MLP() function 得写成MLP(layer1_dim) 的形式 
        self.layer2_dim = layer2_dim
        self.x_dim = x_dim
        self.y_dim = y_dim 
        self.learning_rate = learning_rate

        """
        input data
        """
        
        # training data: record and label
        self.x = tf.placeholder(tf.float32, shape=(None, self.x_dim), name='x-input')
        self.y = tf.placeholder(tf.float32, shape=(None, self.y_dim), name='y-input')


        """
        graph structure
        """
        # predict data: label
        self.y_pred = self.MLP()
        self.y_pred_softmax = tf.nn.softmax(self.y_pred)
        """
        model training 
        """
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y))
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,name='optimizer') 
        self.train_op = self.optimizer.minimize(self.loss,name='train_op')

    def MLP(self):
        # network parameter
        # 问题：这里需要加self吗？比如下面这行
        # [x_dim, self.layer1_dim]这两个
        with tf.variable_scope("layer1"):
            self.W1 = tf.get_variable("w1", initializer=tf.random_normal([self.x_dim, self.layer1_dim], stddev=0.1), dtype=tf.float32)
            self.b1 = tf.get_variable("b1", initializer=tf.zeros([self.layer1_dim]), dtype=tf.float32)
        with tf.variable_scope("layer2"):
            self.W2 = tf.get_variable("w2", initializer=tf.random_normal([self.layer1_dim, self.layer2_dim], stddev=0.1), dtype=tf.float32)
            self.b2 = tf.get_variable("b2", initializer= tf.zeros([self.layer2_dim]), dtype=tf.float32)
        with tf.variable_scope("output"):
            self.W3 = tf.get_variable("w_output", initializer=tf.random_normal([self.layer2_dim, self.y_dim], stddev=0.1), dtype=tf.float32)
            self.b3 = tf.get_variable("b3", initializer= tf.zeros([self.y_dim]), dtype=tf.float32)
        hidden1 = tf.nn.sigmoid(tf.matmul(self.x, self.W1) + self.b1)  
        hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, self.W2) + self.b2)  
        y_pred = tf.matmul(hidden2, self.W3 + self.b3)
        return y_pred

    def CNN(self):
        pass




