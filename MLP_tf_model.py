#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-10 13:25
@desc: 
batch normalization references:
    1. https://blog.csdn.net/lhanchao/article/details/70308092
    2. https://zhuanlan.zhihu.com/p/24810318
    3. https://blog.csdn.net/wfei101/article/details/78587046
    4. https://www.zhihu.com/question/53133249
    5. https://www.cnblogs.com/hrlnw/p/7227447.html
    6. https://www.jianshu.com/p/cb8ebcee1b15
    7. https://blog.csdn.net/qq_38906523/article/details/80070012

gradient clip references:
    1. 
    2. 

L1/L2 regularization references:
    1. 

save/load model references:
    1. 

tensorboard usage:
    1. 

dropout references:
    1. 
"""
from __future__ import print_function
# import sys
# sys.path.append("..") #if you want to import python module from other folders, 
                        #you need to append the system path
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm #for batch normalization

from numpy.random import RandomState
import numpy as np


class Config(object):
    def __init__(self,args):
        # 这里定义的参数我用了大写
        self.DROPOUT_KEEP = args.dropout_keep
        self.LAYER1_DIM = args.layer1_dim
        self.LAYER2_DIM = args.layer2_dim
        self.LEARNING_RATE = args.learning_rate
        self.EPOCH = args.epoch
        self.BATCH_SIZE = args.batch_size
        self.MAZ_GRAD_NORM = args.max_grad_norm
        self.GRAD_CLIP = args.grad_clip 
        self.REGULARIZATION_RATE = args.learning_rate
        self.IS_BATCH_NORM = args.is_batch_norm

class CitationRecNet(object):
    def __init__(self, layer1_dim, layer2_dim, x_dim, 
                y_dim,grad_clip,learning_rate,is_batch_norm):
        
        #in order to generate same random sequences 
        tf.set_random_seed(1)

        """
        input parameter
        """ 
        # 问题： 等式右边的是不是上面那行init后面括号里的参数
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.is_batch_norm = is_batch_norm

        #whether gradient clip
        self.grad_clip = grad_clip 
        #learning rate decay
        self.global_step = tf.Variable(0, trainable=False) 
        self.learning_rate = tf.train.exponential_decay(learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=100,decay_rate=0.99)
        

        """
        input data
        """
        #regularization 
        self.regularization_rate = tf.placeholder(dtype=tf.float32, name='regularization_rate')
        # L2 regularization, you can choose to use L1/L2 separately or use their combination by choosing:
        # l2_regularizer, l1_regularizer or l1_l2_regularizer
        self.regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate) 

        #gradient clip 
        self.max_grad_norm = tf.placeholder(dtype=tf.float32, name='max_grad_norm')  
        
        # dropout keep probability
        self.dropout_keep = tf.placeholder(dtype=tf.float32, name='dropout_keep')

        # training data: record and label
        self.x = tf.placeholder(tf.float32, shape=(None, self.x_dim), name='x-input')
        self.y = tf.placeholder(tf.float32, shape=(None, self.y_dim), name='y-input')


        """
        graph structure
        """
        # predict data: label
        self.y_pred = self.MLP()
        
        """
        model training 
        """
        
        #batch normalization
        if self.is_batch_norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):#for batch normalization
                # 另外一种写法AAA
                # self.loss = tf.nn.softmax_cross_entropy_with_logits(y,y_)
                self.loss = -tf.reduce_mean(self.y * tf.log(tf.clip_by_value(self.y_pred, 1e-10, 1.0)))
                self.loss = self.loss + tf.add_n(tf.get_collection('loss')) #L2 regularization  
        else:
            self.loss = -tf.reduce_mean(self.y * tf.log(tf.clip_by_value(self.y_pred, 1e-10, 1.0)))
            self.loss = self.loss + tf.add_n(tf.get_collection('loss')) #L2 regularization

        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,name='optimizer')

        #gradient clip
        if grad_clip: 
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)
            # grads, tvars = zip(*self.optimizer.compute_gradients(self.loss))
            # grads, global_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars),global_step=self.global_step)
        else:
            self.train_op = self.optimizer.minimize(self.loss,self.global_step,name='train_op')


        #for tensorboard visualization
        tf.summary.scalar("loss", self.loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        self.merged_summary_op = tf.summary.merge_all()

    def MLP(self):
        # network parameter
        # 问题：这里需要加self吗？比如下面这行
        # [x_dim, self.layer1_dim]这两个
        with tf.variable_scope("layer1"):
            self.W1 = tf.get_variable("w1", initializer=tf.random_normal([x_dim, self.layer1_dim], stddev=0.1), dtype=tf.float32)
            self.b1 = tf.get_variable("b1", initializer=tf.zeros([self.layer1_dim]), dtype=tf.float32)
        with tf.variable_scope("layer2"):
            self.W2 = tf.get_variable("w2", initializer=tf.random_normal([self.layer1_dim, self.layer2_dim], stddev=0.1), dtype=tf.float32)
            self.b2 = tf.get_variable("b2", initializer= tf.zeros([self.layer2_dim]), dtype=tf.float32)
        with tf.variable_scope("output"):
        self.W3 = tf.get_variable("w_output", initializer=tf.random_normal([layer2_dim, y_dim], stddev=0.1), dtype=tf.float32)
        
        tf.add_to_collection('loss',self.regularizer(self.W1))
        tf.add_to_collection('loss',self.regularizer(self.W2))
        tf.add_to_collection('loss',self.regularizer(self.W3))
        
        hidden1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)
        hidden1 = tf.layers.batch_normalization(hidden1, training=self.is_batch_norm)
        hidden1_drop = tf.nn.dropout(hidden1, self.dropout_keep)
        hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, self.W2) + self.b2)
        hidden1 = tf.layers.batch_normalization(hidden1, training=self.is_batch_norm)
        hidden2_drop = tf.nn.dropout(hidden2, self.dropout_keep)
        y_pred= tf.nn.softmax(tf.matmul(hidden2_drop, self.W3))
        # 针对另外一种写法AAA
        # 这边是不是改成
        # y_ = tf.matmul(hidden2_drop, self.W3)
        return y_pred

    def CNN(self):
        pass




