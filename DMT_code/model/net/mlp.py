from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
file_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(file_path)
sys.path.append('../../util')
from util import *
from base import base

import tensorflow as tf

class mlp(base):
    def __init__(self, wnd_conf):
        # call base constructor fuction
        base.__init__(self, wnd_conf)

        # data member: self.wnd_conf
        self.wnd_conf = wnd_conf
        # data member: self.output_units
        self.output_units   = wnd_conf[MODEL][OUTPUT_UNITS]
        # data member: self.hidden_units
        self.hidden_units = wnd_conf[MODEL][HIDDEN_UNITS]
        # data member: self.dropout_keep_prob_list
        self.dropout_keep_prob_list     = wnd_conf[MODEL][DROPOUT]

    # multi-layered perceptron: fully connected neural networks
    def mlp(self, inputs, is_train=True):
        y = inputs
        # hidden layers: idx-th hidden layer
        for idx, size in enumerate(self.hidden_units):
            y = self.dense_layer("layer%d" % idx, \
                                y, y.get_shape()[1].value, \
                                size, \
                                tf.nn.relu, \
                                bias_init=0.1, \
                                keep_prob=self.dropout_keep_prob_list[idx], \
                                is_train=is_train)

        # output layer, sigmoid activation with init=0, relu activation with init=0.1
        y = self.dense_layer("layer%d" % len(self.hidden_units), \
                            y, y.get_shape()[1].value, \
                            self.output_units, \
                            tf.identity, \
                            bias_init=0.0, \
                            is_train=is_train)
        return y

    def l2_norm(self, inputs):
        reg_loss = tf.losses.get_regularization_losses()
        total_reg_loss = tf.reduce_sum(reg_loss)

        return total_reg_loss

    # inference determines which model to choose
    # inputs =>  {featureName: featureVal}
    def inference(self, inputs, is_train=True):
        return self.mlp(inputs['features'], is_train=is_train)
