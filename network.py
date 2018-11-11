#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os

class ADINetwork(object):
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
        p2 = tf.layers.dense(p1, 12, activation = self.activation)

        p_out = tf.nn.softmax(p2)

        v1 = tf.layers.dense(d1, 512, activation = self.activation)
        v_out = tf.layers.dense(v1, 1, activation = self.activation)

        return (v_out, p_out)
    
    def setup(self):
        self.x = tf.placeholder(shape = (None, 20 * 24), dtype = tf.float32)
        self.y_value = tf.placeholder(shape = (None, 1), dtype = tf.float32)
        self.y_policy = tf.placeholder(shape = (None, 12), dtype = tf.float32)
        
        self.v_out, self.p_out = self.inference(self.x)

        v_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.p_out, labels=self.y_policy))
        p_cost = tf.losses.mean_squared_error(self.y_value, self.v_out)
        self.cost = tf.reduce_mean(v_cost + p_cost)

        self.loss = tf.train.RMSPropOptimizer(1e-4).minimize(self.cost)

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
        value, policy = self.sess.run([self.v_out, self.p_out], feed_dict = {self.x: state})
        return value, policy
    
    def save(self):
        self.saver.save(self.sess, self.save_dir, global_step = self.global_step)
    
    def log(self):
        pass

    def train(self, states, actions, values, weight = 1):

        _, cost = self.sess.run([self.loss, self.cost], feed_dict = {self.x: states,
                                             self.y_value: values,
                                             self.y_policy: actions})
        self.local_step += 1
        self.global_step += 1
        return cost
