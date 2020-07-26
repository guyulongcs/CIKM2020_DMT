# -*- coding: utf-8 -*-
# @Time    : 2020-01-01 13:24
# @Author  : guyulong@jd.com
# @File    : transformer.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
sys.path.append('../../util')
from util import *
from base import base
from TransformerModel import *

class transformer(base):
    def __init__(self, wnd_conf):
        # call base constructor fuction
        base.__init__(self, wnd_conf)

        # data member: self.wnd_conf
        self.wnd_conf = wnd_conf
        # data member: self.wnd_wd
        self.wnd_wd = wnd_conf[MODEL][WND_WD]
        # data member: self.output_units
        self.output_units = wnd_conf[MODEL][OUTPUT_UNITS]
        # data member: self.hidden_units
        self.hidden_units = wnd_conf[MODEL][HIDDEN_UNITS]
        # data member: self.dropout_keep_prob_list
        self.dropout_keep_prob_list = wnd_conf[MODEL][DROPOUT]
        self.is_decoder_add_pos_emb = wnd_conf.is_decoder_add_pos_emb
        self.zero_pad = wnd_conf.zero_pad




    def generate_data(self, inputs):
        with tf.device('/cpu:0'):
            seq_data = []
            for index in range(len(self.wnd_conf.attention_embed_pairs)):
                attention_embed_pair = self.wnd_conf.attention_embed_pairs[index]
                seq_features = []
                tar_sku_features = []
                seq_ts_emb = None

                for attention_pair in attention_embed_pair:
                    user_feature, item_feature = attention_pair
                    mask_v = tf.ones(tf.size(inputs[user_feature].values), tf.int32)
                    mask_sp = tf.SparseTensor(indices=inputs[user_feature].indices, \
                                              values=mask_v,
                                              dense_shape=inputs[user_feature].dense_shape)
                    mask = tf.sparse.to_dense(mask_sp)
                    lens = tf.reduce_sum(mask, 1)

                    for emb in self.wnd_conf.embedding_list:
                        embedding_name = emb[0]
                        id_size = emb[1]
                        embedding_dim = emb[2]
                        feat_name = emb[3]
                        if feat_name == user_feature:
                            embed = self.embedding(embedding_name, id_size, embedding_dim, tf.AUTO_REUSE, zero_pad = self.zero_pad)
                            seq_features.append(tf.nn.embedding_lookup(embed, tf.sparse.to_dense(inputs[feat_name])))
                        elif feat_name == item_feature:
                            embed = self.embedding(embedding_name, id_size, embedding_dim, tf.AUTO_REUSE, zero_pad =self.zero_pad)
                            tar_sku_features.append(tf.nn.embedding_lookup(embed, inputs[feat_name].values))
                        else:
                            continue

                if(self.wnd_conf.is_use_seq_ts):
                    ts_feature = self.wnd_conf.attention_embed_seq_ts[index]
                    for emb in self.wnd_conf.embedding_list:
                        embedding_name = emb[0]
                        id_size = emb[1]
                        embedding_dim = emb[2]
                        feat_name = emb[3]
                        if feat_name == ts_feature:
                            print("Find_Seq_ts_{0}_success!".format(ts_feature))
                            embed = self.embedding(embedding_name, id_size, embedding_dim, tf.AUTO_REUSE, zero_pad =self.zero_pad)
                            ts_input = tf.sparse.to_dense(inputs[feat_name])
                            #parse time
                            ts_input = tf.add(tf.cast(tf.log(tf.cast(ts_input, tf.float32)) / tf.log(2.0), tf.int32), tf.constant(1, tf.int32))
                            ts_input = tf.clip_by_value(ts_input, clip_value_min=0, clip_value_max=23)
                            seq_ts_emb = tf.nn.embedding_lookup(embed, ts_input)
                            break
                        else:
                            continue

                seq_emb = tf.concat(seq_features, -1)
                tar_sku_emb = tf.concat(tar_sku_features, -1)
                single_seq = [mask, lens, seq_emb, tar_sku_emb, seq_ts_emb]
                seq_data.append(single_seq)

            return seq_data


    def trans_core(self, seq_data, is_train=True):
        interest_state_lst = []
        for i,sequence in enumerate(seq_data):
            stag = 'sequence_' + str(i)
            seq_mask, seq_lens, seq_emb, tar_sku_emb, seq_ts_emb = sequence
            with tf.variable_scope('trans_' + stag, reuse=tf.AUTO_REUSE):
                m = TransformerModel(self.wnd_conf)

                if(self.wnd_conf.is_trans_input_by_mlp):
                    seq_emb = tf.layers.dense(seq_emb, self.wnd_conf.d_model, name='dense_trans_seq_' + stag)
                    tar_sku_emb = tf.layers.dense(tar_sku_emb, self.wnd_conf.d_model, name='dense_trans_sku_' + stag)

                seq_key = seq_emb
                seq_key_lens=seq_lens
                seq_key_ts_emb = seq_ts_emb
                seq_q = tf.expand_dims(tar_sku_emb, axis=1)   #[N, 1, d]
                seq_q_lens = tf.ones_like(seq_q[:, 0, 0], dtype=tf.int32) #[N]
                input = (seq_q, seq_q_lens, seq_key, seq_key_lens, seq_key_ts_emb)
                user_stat = m.encode_decode(input, name="encode_decode_" + stag, training=is_train)

                print("is_trans_out_concat_item:", self.wnd_conf.is_trans_out_concat_item)
                print("is_trans_out_by_mlp:", self.wnd_conf.is_trans_out_by_mlp)

                #method 1: [user_stat, tar_sku_emb]
                if(self.wnd_conf.is_trans_out_concat_item):
                    final_state = tf.concat([user_stat, tar_sku_emb], axis=-1)
                    if(self.wnd_conf.is_trans_out_by_mlp):
                        final_state = tf.layers.dense(final_state, self.wnd_conf.d_model, name='dense_trans_concat_' + stag)
                #method 2: user_stat
                else:
                    final_state = user_stat
            interest_state_lst.append(final_state)
        interest_state = tf.concat(interest_state_lst, -1)
        print("interest_state:", interest_state.get_shape().as_list())
        return interest_state


    def embedding_trans(self, inputs, is_train=True):
        with tf.variable_scope('embedding_trans', reuse=tf.AUTO_REUSE):
            #get embeddings of inputs
            self.seq_data = self.generate_data(inputs)
            self.interest_state = self.trans_core(self.seq_data, is_train=is_train)

            features = self.embedding_combiner(inputs)
            y = tf.concat([features, self.interest_state], -1)

            #mlp
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


    def embedding_combiner(self, inputs, combiner_type="mean"):
        features = None
        if (self.is_use_feature):
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

            seq_features = []
            for attention_embed_pair in self.wnd_conf.attention_embed_pairs:
                for user_feature, item_feature in attention_embed_pair:
                    seq_features.append(user_feature)
            if feature_name in seq_features:
                continue

            batch_embedding = self.embedding(embedding_name, id_size, embedding_dim, tf.AUTO_REUSE)
            avg_embedding = tf.nn.embedding_lookup_sparse(batch_embedding, inputs[feature_name], Wts,
                                                          combiner=combiner_type)

            if feature_name in sim_ids:
                sim_features[feature_name] = avg_embedding

            if(features is None):
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
        return self.embedding_trans(inputs, is_train=is_train)
