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

class embed_mlp(base):
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

    def embedding_mlp(self, inputs, is_train=True):
        features = self.embedding_combiner(inputs, is_train=is_train)

        y = features
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
			tf.reduce_sum(regularization_losses) * self.wnd_conf[MODEL][L2_EMB_LAMBDA] / self.wnd_conf[MODEL][BATCH_SIZE]
        return total_reg_loss

    # inference determines which model to choose
    # inputs =>  {featureName: featureVal}
    def inference(self, inputs, is_train=True):
        return self.embedding_mlp(inputs, is_train=is_train)
