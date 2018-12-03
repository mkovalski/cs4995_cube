#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import pdb

class ADINetwork(object):
    def __init__(self, output_dir, use_gpu = True):
        self.activation = tf.nn.elu
        self.sess = None
        self.save_dir = output_dir
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        self.use_gpu = use_gpu

        if self.save_dir[-1] != '/':
            self.save_dir = self.save_dir + '/'

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
        self.global_step = 0
        self.local_step = 0

    def inference(self, X):
        
        d0 = tf.layers.dense(X, 4096, activation = self.activation)
        d1 = tf.layers.dense(d0, 2048, activation = self.activation)

        p1 = tf.layers.dense(d1, 512, activation = self.activation)
        p_out = tf.layers.dense(p1, 12, activation = None)

        v1 = tf.layers.dense(d1, 512, activation = self.activation)

        # Added a sigmoid activation, although the paper doesn't
        # use an activation function here. Since sigmoid spans 0 to 1
        # and the true reward also does, I do this.
        v_out = tf.layers.dense(v1, 1, activation = None)

        return (v_out, p_out)
    
    def setup(self, cube_size):
        '''
        To use weighted values, provide a fixed batch size since the 
        weight vector will also be fixed
        
        '''

        lr = 1e-6

        self.x = tf.placeholder(shape = (None, cube_size), dtype = tf.float32)
        self.y_value = tf.placeholder(shape = (None, 1), dtype = tf.float32)
        self.y_policy = tf.placeholder(shape = (None, 12), dtype = tf.float32)
        
        self.weight = tf.placeholder(shape = (None,), dtype = tf.float32)

        self.v_out, self.p_out = self.inference(self.x)
        
        self.policy = tf.nn.softmax(self.p_out)

        # Weighted cross entropy based on which batch it is
        # Later batches are searched further down the tree, should have less weight
        # associated with it
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.p_out, labels=self.y_policy)
        self.weighted_cross_entropy = tf.math.multiply(self.cross_entropy, self.weight)

        self.p_cost = tf.reduce_mean(self.weighted_cross_entropy)
        
        self.v_cost = tf.losses.mean_squared_error(self.y_value, self.v_out, weights = tf.reshape(self.weight, (-1, 1)))

        self.cost = tf.reduce_sum(self.v_cost + self.p_cost)

        self.loss = tf.train.RMSPropOptimizer(lr).minimize(self.cost)
        
        if not self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]=""

        self.sess = tf.Session()
         
        self.saver = tf.train.Saver()
        
        if os.path.isdir(self.save_dir):
            try:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
            except Exception as e:
                print("Could not load checkpoint, sorry!")
                print(e)
                print("Starting from scratch")
                self.sess.run(tf.initializers.global_variables())
        else:
            self.sess.run(tf.initializers.global_variables())
            # TODO: Find a way to get the global variable state

    def evaluate(self, state):
        value, policy = self.sess.run([self.v_out, self.policy], feed_dict = {self.x: state})
        return value, policy
    
    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir, global_step = self.global_step)
    
    def log(self):
        pass

    def train(self, states, actions, values, weight):
        
        _, cost = self.sess.run([self.loss, self.cost], feed_dict = {self.x: states,
                                             self.y_value: values,
                                             self.y_policy: actions,
                                             self.weight: weight})
        '''        
        if self.local_step % 1000 == 0 and self.local_step != 0:

            ce = self.sess.run(self.cross_entropy, feed_dict = {self.x: states,
                                                 self.y_value: values,
                                                 self.y_policy: actions,
                                                 self.weight : weight})
            wce = self.sess.run(self.weighted_cross_entropy, feed_dict = {self.x: states,
                                                 self.y_value: values,
                                                 self.y_policy: actions,
                                                 self.weight: weight})


            v_cost = self.sess.run(self.v_cost, feed_dict = {self.x: states,
                                                 self.y_value: values,
                                                 self.y_policy: actions,
                                                 self.weight: weight})
            
            pdb.set_trace()
        '''

        self.local_step += 1
        self.global_step += 1
        return cost
