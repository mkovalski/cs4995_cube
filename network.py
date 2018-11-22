#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import pdb

class PolicyNetwork(object):
    def __init__(self, output_dir):
        self.activation = tf.nn.elu
        self.sess = None
        self.save_dir = output_dir

        if self.save_dir[-1] != '/':
            self.save_dir = self.save_dir + '/'

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
    
        self.global_step = 0
        self.local_step = 0

    def inference(self, X):
        
        d0 = tf.layers.dense(X, 4096, activation = self.activation)
        d1 = tf.layers.dense(d0, 2048, activation = self.activation)

        p1 = tf.layers.dense(d1, 512, activation = self.activation)
        p2 = tf.layers.dense(p1, 12, activation = None)

        p_out = tf.nn.softmax(p2)

        return p_out
    
    def setup(self, batch_size = 1):
        '''
        To use weighted values, provide a fixed batch size since the 
        weight vector will also be fixed
        
        '''

        lr = 1e-5

        self.x = tf.placeholder(shape = (None, 20 * 24), dtype = tf.float32)
        self.y_policy = tf.placeholder(shape = (None, 12), dtype = tf.float32)
        
        self.weight = tf.placeholder(shape = (None,), dtype = tf.float32)

        self.p_out = self.inference(self.x)
        
        # Weighted cross entropy based on which batch it is
        # Later batches are searched further down the tree, should have less weight
        # associated with it
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.p_out, labels=self.y_policy)
        self.weighted_cross_entropy = tf.math.multiply(self.cross_entropy, self.weight)

        self.cost = tf.reduce_mean(self.weighted_cross_entropy)

        self.loss = tf.train.AdamOptimizer(lr).minimize(self.cost)

        self.sess = tf.Session()
        
        self.saver = tf.train.Saver()
        
        checkpoint_path = os.path.join(self.save_dir, 'checkpoints')
        if os.path.isdir(checkpoint_path):
            try:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_path))
            except Exception as e:
                print("Could not load checkpoint, sorry!")
                print(e)
                print("Starting from scratch")
                self.sess.run(tf.initializers.global_variables())
        else:
            self.sess.run(tf.initializers.global_variables())
            # TODO: Find a way to get the global variable state

    def evaluate(self, state):
        policy = self.sess.run(self.p_out, feed_dict = {self.x: state})
        return policy
    
    def save(self):
        self.saver.save(self.sess, self.save_dir, global_step = self.global_step)
    
    def log(self):
        pass

    def train(self, states, actions, weight):
        
        _, cost = self.sess.run([self.loss, self.cost], feed_dict = {self.x: states,
                                             self.y_policy: actions,
                                             self.weight: weight})
        
        self.local_step += 1
        self.global_step += 1
        return cost

class ValueNetwork(object):
    def __init__(self, output_dir):
        self.activation = tf.nn.elu
        self.sess = None
        self.save_dir = output_dir

        if self.save_dir[-1] != '/':
            self.save_dir = self.save_dir + '/'

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
    
        self.global_step = 0
        self.local_step = 0

    def inference(self, X):
        
        d0 = tf.layers.dense(X, 4096, activation = self.activation)
        d1 = tf.layers.dense(d0, 2048, activation = self.activation)

        v1 = tf.layers.dense(d1, 512, activation = self.activation)
        v_out = tf.layers.dense(v1, 1,  activation = tf.sigmoid)

        return v_out
    
    def setup(self, batch_size = 1):
        '''
        To use weighted values, provide a fixed batch size since the 
        weight vector will also be fixed
        
        '''

        lr = 1e-5

        self.x = tf.placeholder(shape = (None, 20 * 24), dtype = tf.float32)
        self.y_value = tf.placeholder(shape = (None, 1), dtype = tf.float32)
        
        self.weight = tf.placeholder(shape = (None,), dtype = tf.float32)

        self.v_out = self.inference(self.x)
        
        # Weighted cross entropy based on which batch it is
        # Later batches are searched further down the tree, should have less weight
        # associated with it
        
        self.cost = tf.losses.mean_squared_error(self.y_value, self.v_out, weights = tf.reshape(self.weight, (batch_size, 1)))

        self.loss = tf.train.AdamOptimizer(lr).minimize(self.cost)

        self.sess = tf.Session()
        
        self.saver = tf.train.Saver()
        
        checkpoint_path = os.path.join(self.save_dir, 'checkpoints')
        if os.path.isdir(checkpoint_path):
            try:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_path))
            except Exception as e:
                print("Could not load checkpoint, sorry!")
                print(e)
                print("Starting from scratch")
                self.sess.run(tf.initializers.global_variables())
        else:
            self.sess.run(tf.initializers.global_variables())
            # TODO: Find a way to get the global variable state

    def evaluate(self, state):
        value = self.sess.run(self.v_out, feed_dict = {self.x: state})
        return value
    
    def save(self):
        self.saver.save(self.sess, self.save_dir, global_step = self.global_step)
    
    def log(self):
        pass

    def train(self, states, values, weight):
        
        _, cost = self.sess.run([self.loss, self.cost], feed_dict = {self.x: states,
                                             self.y_value: values,
                                             self.weight: weight})
        self.local_step += 1
        self.global_step += 1
        return cost
