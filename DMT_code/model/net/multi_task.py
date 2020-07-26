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


class multi_task(base):
    def __init__(self, wnd_conf):
        # call base constructor fuction
        base.__init__(self, wnd_conf)

        # data member: self.wnd_conf
        self.wnd_conf = wnd_conf
        # data member: self.output_units
        self.output_units = wnd_conf[MODEL][OUTPUT_UNITS]
        # data member: self.hidden_units
        self.hidden_units_bottom = wnd_conf[MODEL][hidden_units_bottom]
        self.hidden_units_task = wnd_conf[MODEL][hidden_units_task]
        # data member: self.dropout_keep_prob_list
        self.dropout_keep_prob_list_bottom = wnd_conf[MODEL][DROPOUT_BOTTOM]
        self.dropout_keep_prob_list_task = wnd_conf[MODEL][DROPOUT_TASK]


    def embedding_shared_bottom(self, inputs, name, is_train=True):
        y = inputs
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            #shared bottom
            # hidden layers: idx-th hidden layer
            for idx, size in enumerate(self.hidden_units_bottom):
                y = self.dense_layer("layer%d" % idx,
                                     y, y.get_shape()[1].value,
                                     size,
                                     tf.nn.relu,
                                     bias_init=0.1,
                                     keep_prob=self.dropout_keep_prob_list_bottom[idx],
                                     is_train=is_train)

            return y

    def l2_norm(self, inputs):
        regularization_losses = []
        for emb in self.wnd_conf.embedding_list:
            embedding_name = emb[0]
            id_size = emb[1]
            embedding_dim = emb[2]
            feature_name = emb[3]

            embeding = self.embedding(embedding_name, id_size, embedding_dim, tf.AUTO_REUSE)

            uniq_ids, _ = tf.unique(inputs[feature_name].values)
            batch_embedding = tf.gather(embeding, uniq_ids)

            regularization_loss = tf.nn.l2_loss(batch_embedding)
            regularization_losses.append(regularization_loss)

        reg_loss = tf.losses.get_regularization_losses()
        total_reg_loss = tf.reduce_sum(reg_loss) + \
                         tf.reduce_sum(regularization_losses) * self.wnd_conf[MODEL][L2_EMB_LAMBDA] / \
                         self.wnd_conf[MODEL][BATCH_SIZE]
        return total_reg_loss

    def get_click_task(self, sharedOutput, is_train):
        y = sharedOutput
        with tf.variable_scope("click"):
            for idx, size in enumerate(self.hidden_units_task):
                y = self.dense_layer("click_fc_layer%d" % idx,
                                     y, y.get_shape()[1].value,
                                     size,
                                     tf.nn.relu,
                                     bias_init=0.1,
                                     keep_prob=self.dropout_keep_prob_list_task[idx],
                                     is_train=is_train)

            # head output
            logits = self.dense_layer("click_out_layer",
                                      y, y.get_shape()[1].value,
                                      self.output_units,
                                      tf.identity,
                                      bias_init=0.0,
                                      is_train=is_train)
        return logits


    def get_order_task(self, sharedOutput, is_train):
        y = sharedOutput
        with tf.variable_scope("order"):
            for idx, size in enumerate(self.hidden_units_task):
                y = self.dense_layer("ord_fc_layer%d" % idx,
                                     y, y.get_shape()[1].value,
                                     size,
                                     tf.nn.relu,
                                     bias_init=0.1,
                                     keep_prob=self.dropout_keep_prob_list_task[idx],
                                     is_train=is_train)
            # head output
            logits = self.dense_layer("order_out_layer",
                                      y, y.get_shape()[1].value,
                                      self.output_units,
                                      tf.identity,
                                      bias_init=0.0,
                                      is_train=is_train)
        return logits

    # inference determines which model to choose
    # inputs =>  {featureName: featureVal}
    def inference(self, inputs, is_train=True):
        inputs = self.embedding_combiner(inputs, is_train=is_train)
        shared_embeding = self.embedding_shared_bottom(inputs, 'shared_embeding', is_train=is_train)

        click_logit = self.get_click_task(shared_embeding, is_train)
        order_logit = self.get_order_task(shared_embeding, is_train)

        if self.wnd_conf[PARAMETER][LOSS_WEIGHT_METHOD] == 'uncertainty':
            value = [0.0]
            init = tf.constant_initializer(value)
            self.click_weight = tf.get_variable('uncertainty_click_weight', shape=[1], initializer=init)
            self.order_weight = tf.get_variable('uncertainty_order_weight', shape=[1], initializer=init)


        return click_logit, order_logit
