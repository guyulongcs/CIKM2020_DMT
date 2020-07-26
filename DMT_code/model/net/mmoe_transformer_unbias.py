from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
sys.path.append('../../util')

import  tensorflow as tf
from util import *
from base import base
from mmoe import mmoe
from TransformerModel import *

class mmoe_transformer_unbias(mmoe):
    def __init__(self, wnd_conf):
        # call base constructor fuction
        base.__init__(self, wnd_conf)

        # data member: self.wnd_conf
        self.wnd_conf = wnd_conf
        # data member: self.output_units
        self.output_units   = wnd_conf[MODEL][OUTPUT_UNITS]
        # data member: self.hidden_units
        self.hidden_units_bottom = wnd_conf[MODEL][hidden_units_bottom]
        self.hidden_units_task = wnd_conf[MODEL][hidden_units_task]
        # data member: self.dropout_keep_prob_list
        self.dropout_keep_prob_list_bottom = wnd_conf[MODEL][DROPOUT_BOTTOM]
        self.dropout_keep_prob_list_task = wnd_conf[MODEL][DROPOUT_TASK]
        self.num_experts = wnd_conf[MODEL][num_experts]

        self.is_decoder_add_pos_emb = wnd_conf.is_decoder_add_pos_emb
        self.zero_pad = wnd_conf.zero_pad

        #unbias
        self.hidden_units_bias = wnd_conf[MODEL][HIDDEN_UNITS_BIAS]
        self.dropout_rate_bias = wnd_conf[MODEL][dropout_rate_bias]


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

    def expert_gate(self,features,units, dropout_keep_prob_list,num_experts=4,num_tasks=3, is_train=True):
        expert_outputs = []

        for idx in range(num_experts):
            with tf.variable_scope("expert-%d" % idx):
                y = features
                for idx, size in enumerate(units):
                    #keep hidden size same as shared bottom
                    y = self.dense_layer("expert-layer-%d" % idx,
                                         y, y.get_shape()[1].value,
                                         size,
                                         tf.nn.relu,
                                         bias_init=0.1,
                                         keep_prob=dropout_keep_prob_list[idx],
                                         is_train=is_train)
                expert_outputs.append(y)

        #gates
        gates_output =[]

        for idx in range(num_tasks):
            with tf.variable_scope("gates-%d" % idx):
                y = features
                y = self.dense_layer("gates-layer-0",
                                     y, y.get_shape()[1].value,
                                     num_experts,
                                     tf.nn.softmax,
                                     bias_init=0.1,
                                     keep_prob=1,
                                     is_train=is_train)
                gates_output.append(y)

        final_outputs =[]


        experts = tf.stack(expert_outputs, axis=-1)

        #weigth task output
        for gate_output in gates_output:
            weights = tf.tile(tf.expand_dims(gate_output, axis=1), [1, experts.get_shape()[1].value, 1])
            weighted_output = tf.reduce_sum(experts * weights, axis=2)
            final_outputs.append(weighted_output)
        return final_outputs

    def build_tower(self,task_layer,units,dropout_keep_prob_list,name,is_train=True):
        with tf.variable_scope(name):
            y = task_layer
            for idx, size in enumerate(units):
                y = self.dense_layer(name + "-fc-%d" % idx,
                                     y, y.get_shape()[1].value,
                                     size,
                                     tf.nn.relu,
                                     bias_init=0.1,
                                     keep_prob=dropout_keep_prob_list[idx],
                                     is_train=is_train)
            output_layer = self.dense_layer(name+"-output",
                                           y, y.get_shape()[1].value,
                                           1,
                                           tf.identity,
                                           bias_init=0.1,
                                           keep_prob=1,
                                           is_train=is_train)

        return  output_layer



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
            self.seq_data = self.generate_data(inputs)
            self.interest_state = self.trans_core(self.seq_data, is_train=is_train)

            features = self.embedding_combiner(inputs)
            y = tf.concat([features, self.interest_state], -1)
        return y

    def embedding_combiner_bias(self, inputs, is_train=True, combiner_type="mean"):
        features = None
        for emb in self.wnd_conf.embedding_list_bias:
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

            if (features is None):
                features = avg_embedding
            else:
                features = tf.concat(values=[features, avg_embedding], axis=1)

        return features

    def embedding_mlp_bias(self, inputs, is_train=True):
        features = self.embedding_combiner_bias(inputs, is_train=is_train)

        y = features

        # hidden layers: idx-th hidden layer
        for idx, size in enumerate(self.hidden_units_bias):
            y = tf.layers.dense(inputs=y,
                                units=size,
                                activation=tf.nn.relu,
                                use_bias=True,
                                bias_initializer=tf.zeros_initializer(),
                                name="layer_bias%d" % idx
            )

            y = tf.layers.dropout(inputs=y,
                                  rate=self.dropout_rate_bias[idx],
                                  training=is_train,
                                  name="dropout_%d" % idx
            )

        # output layer, sigmoid activation with init=0, relu activation with init=0.1
        y = tf.layers.dense(inputs=y,
                            units=self.output_units,
                            activation=tf.identity,
                            use_bias=True,
                            bias_initializer=tf.zeros_initializer(),
                            name="layer_bias%d" % len(self.hidden_units_bias)
                            )

        return y

    # inference determines which model to choose
    # inputs =>  {featureName: featureVal}
    def inference(self, inputs, is_train=True, is_predict=False):
        features = self.embedding_trans(inputs, is_train=is_train)

        # input_layer = Input(shape=(num_features,))
        with tf.variable_scope("mmoe_layers"):
            mmoe_layers =self.expert_gate(features, self.hidden_units_bottom, self.dropout_keep_prob_list_bottom, num_experts=self.num_experts, num_tasks=2, is_train=is_train)

        output_layers = []

        output_info = ['click', 'order']

        # Build tower layer from MMoE layer
        for index, task_layer in enumerate(mmoe_layers):
            output_layer = self.build_tower(task_layer, self.hidden_units_task, self.dropout_keep_prob_list_task, output_info[index], is_train=is_train)
            output_layers.append(output_layer)

        #output
        y_rel = (output_layers[0], output_layers[1])

        if(not is_predict):
            y_bias = self.embedding_mlp_bias(inputs, is_train=is_train)
            return (y_rel, y_bias)
        else:
            return y_rel