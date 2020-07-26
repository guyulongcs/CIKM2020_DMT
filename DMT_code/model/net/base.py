from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
sys.path.append('../../util')
from util import *

import tensorflow as tf

class base(object):
    def __init__(self, wnd_conf):
        # data member: self.wnd_conf
        self.wnd_conf = wnd_conf
        # data member: self.wnd_wd
        self.wnd_wd   = wnd_conf[MODEL][WND_WD]
        # data member: self.bn_decay
        self.bn_decay  = wnd_conf[MODEL][BN_DECAY]
        # data member: self.is_bn
        self.is_bn     = wnd_conf[MODEL][IS_BN]
        # data member: self.is_dropout
        self.is_dropout = wnd_conf[MODEL][IS_DROPOUT]
        self.is_use_feature = wnd_conf[MODEL][IS_USE_FEATURE]
        print("is_use_feature:", self.is_use_feature)

    # function to create weights and biases
    def weight_bias(self, input_size, layer_size, bias_init):
        with tf.device('/cpu:0'):
            W = tf.get_variable("weights",
                                [input_size, layer_size], # W_shape
                                initializer=tf.truncated_normal_initializer(stddev=0.1),# wnd_wd is used as the regularization hyper-parameter
                                regularizer=(tf.contrib.layers.l2_regularizer(self.wnd_wd) if self.wnd_wd != 0.0 else None))
            b = tf.get_variable("biases",
                                [layer_size], # b_shape,
                                initializer=tf.constant_initializer(value=bias_init))
        return W, b

    def dense_layer(self, layer_name, inputs, input_size, layer_size, activation, bias_init=0.1, keep_prob=1.0, is_train=True):
        # layer_name variable_scope
        with tf.variable_scope(layer_name):
            W, b = self.weight_bias(input_size, layer_size, bias_init)
            layer = tf.matmul(inputs, W) + b
            if self.is_bn:
                with tf.device('/cpu:0'):
                    scale = tf.get_variable("scale", layer_size,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                    shift = tf.get_variable("shift", layer_size,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                    moving_mean = tf.get_variable("moving_mean", layer_size, initializer=tf.zeros_initializer(),
                                                  trainable=False)
                    moving_var = tf.get_variable("moving_var", layer_size,
                                                 initializer=tf.zeros_initializer(), trainable=False)
                    if is_train:
                        mean, var = tf.nn.moments(layer, axes=[0])
                        train_mean = tf.assign(moving_mean, moving_mean * self.bn_decay + mean * (1 - self.bn_decay))
                        train_var = tf.assign(moving_var, moving_var * self.bn_decay + var * (1 - self.bn_decay))
                        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_mean)
                        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, train_var)
                        with tf.control_dependencies([train_mean, train_var]):
                            layer = tf.nn.batch_normalization(layer, mean, var, shift, scale, 0.0001)
                    else:
                        layer = tf.nn.batch_normalization(layer, moving_mean, moving_var, shift, scale, 0.0001)
            layer = activation(layer)
            if self.is_dropout and is_train:
                layer = tf.nn.dropout(layer, keep_prob=keep_prob)

        return layer

    def embedding_old(self, id_name, id_size, emb_dim, reuse = None, zero_pad = False):
        with tf.variable_scope(id_name, reuse=reuse):
            with tf.device('/cpu:0'):
                emb = tf.get_variable("embedding",
                                [id_size, emb_dim], # W_shape
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
                if zero_pad:
                    emb = tf.concat((tf.zeros(shape=[1, emb_dim]), emb), 0)  # [1:, :]

                return emb

    def embedding(self, id_name, id_size, emb_dim, reuse = None, zero_pad = False):
        with tf.variable_scope(id_name, reuse=reuse):
            with tf.device('/cpu:0'):
                emb = tf.get_variable("embedding",
                                [id_size, emb_dim], # W_shape
                                initializer=tf.contrib.layers.xavier_initializer())
                if zero_pad:
                    #emb = tf.concat((tf.zeros(shape=[1, emb_dim]), emb[1:,:]), 0)  # [1:, :]
                    emb = tf.concat((tf.zeros(shape=[1, emb_dim]), emb), 0)  # [1:, :]

                return emb

    def embedding_combiner(self, inputs, is_train=True, combiner_type="mean"):
        features = None
        if(self.is_use_feature):
            features = inputs['features']

        sim_features = {}
        sim_ids = {}
        for sim_embed_1, sim_embed_2 in self.wnd_conf.sim_embed:
            sim_ids[sim_embed_1] = 1
            sim_ids[sim_embed_2] = 1

        for emb in self.wnd_conf.embedding_list:
            embedding_name = emb[0]
            id_size = emb[1]
            embedding_dim = emb[2]
            feature_name = emb[3]
            emb_wts_name = feature_name + "Wts"

            Wts = None
            if emb_wts_name in inputs:
                Wts = inputs[emb_wts_name]

            batch_embedding = self.embedding(embedding_name, id_size, embedding_dim, tf.AUTO_REUSE)
            avg_embedding = tf.nn.embedding_lookup_sparse(batch_embedding, inputs[feature_name], Wts, combiner=combiner_type)

            if feature_name in sim_ids:
                sim_features[feature_name] = avg_embedding

            if(features is None):
                features = avg_embedding
            else:
                features = tf.concat(values = [features, avg_embedding], axis=1)

        for sim_embed_1, sim_embed_2 in self.wnd_conf.sim_embed:
            inner = tf.reduce_sum(sim_features[sim_embed_1] * sim_features[sim_embed_2], 1, keep_dims=True)
            cosin = inner / tf.expand_dims((tf.norm(sim_features[sim_embed_1],ord=2,axis=1) * \
				tf.norm(sim_features[sim_embed_2],ord=2,axis=1)),-1)
            diff = tf.abs(sim_features[sim_embed_1] - sim_features[sim_embed_2])

            features = tf.concat(values = [features, inner, cosin, diff, diff*diff], axis=1)

        return features

    def embedding_combiner_ids(self, inputs, is_train=True, combiner_type="mean"):
        features = None

        sim_features = {}
        sim_ids = {}
        for sim_embed_1, sim_embed_2 in self.wnd_conf.sim_embed:
            sim_ids[sim_embed_1] = 1
            sim_ids[sim_embed_2] = 1

        for emb in self.wnd_conf.embedding_list:
            embedding_name = emb[0]
            id_size = emb[1]
            embedding_dim = emb[2]
            feature_name = emb[3]
            emb_wts_name = feature_name + "Wts"

            Wts = None
            if emb_wts_name in inputs:
                Wts = inputs[emb_wts_name]

            batch_embedding = self.embedding(embedding_name, id_size, embedding_dim, tf.AUTO_REUSE)
            avg_embedding = tf.nn.embedding_lookup_sparse(batch_embedding, inputs[feature_name], Wts,
                                                          combiner=combiner_type)

            if feature_name in sim_ids:
                sim_features[feature_name] = avg_embedding

            if (features is None):
                features = avg_embedding
            else:
                features = tf.concat(values=[features, avg_embedding], axis=1)

        for sim_embed_1, sim_embed_2 in self.wnd_conf.sim_embed:
            inner = tf.reduce_sum(sim_features[sim_embed_1] * sim_features[sim_embed_2], 1, keep_dims=True)
            cosin = inner / tf.expand_dims((tf.norm(sim_features[sim_embed_1], ord=2, axis=1) * \
                                            tf.norm(sim_features[sim_embed_2], ord=2, axis=1)), -1)
            diff = tf.abs(sim_features[sim_embed_1] - sim_features[sim_embed_2])

            features = tf.concat(values=[features, inner, cosin, diff, diff * diff], axis=1)

        return features

    def embedding_update(self, sess):
        for id_name,embedding_path in self.wnd_conf.embedding_init_info.items():
            for embedding_info in self.wnd_conf.embedding_list:
                embedding_name = embedding_info[0]
                if id_name != embedding_name:
                    continue

                id_size = embedding_info[1]
                embedding_dim = embedding_info[2]
                with tf.variable_scope("DnnModel", reuse=True):
                    embedding_var = self.embedding(embedding_name, id_size, embedding_dim, tf.AUTO_REUSE)

                embeddings = np.load(embedding_path + '.' + 'pickle')

                sku_embedding_placeholder = tf.placeholder(tf.float32, [id_size, embedding_dim])
                embedding_init = embedding_var.assign(sku_embedding_placeholder)

                sess.run(embedding_init, feed_dict={sku_embedding_placeholder: embeddings})
                break
