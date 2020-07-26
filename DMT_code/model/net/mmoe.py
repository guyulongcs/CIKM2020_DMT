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

class mmoe(base):
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

    def expert_gate(self,features, units, dropout_keep_prob_list, num_experts=4,num_tasks=3, is_train=True):
        expert_outputs = []

        for idx in range(num_experts):
            with tf.variable_scope("expert-%d" % idx):
                y = features
                for idx, size in enumerate(units):
                    y = self.dense_layer("expert-layer-%d" % idx,
                                         y, y.get_shape()[1].value,
                                         size,
                                         tf.nn.relu,
                                         bias_init=0.1,
                                         keep_prob=dropout_keep_prob_list[idx],
                                         is_train=is_train)
                expert_outputs.append(y)

        expert_output_hidden_size = self.hidden_units_bottom[-1]

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
            weights = tf.tile(tf.expand_dims(gate_output, axis=1), [1, expert_output_hidden_size, 1])
            weighted_output = tf.reduce_sum(experts * weights, axis=2)
            final_outputs.append(weighted_output)
        return final_outputs

    def build_tower(self,task_layer,units, dropout_keep_prob_list, name,is_train=True):
        with tf.variable_scope(name):
            y = task_layer
            for idx, size in enumerate(units):
                y = self.dense_layer(name+"-fc-%d" % idx,
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
    # inference determines which model to choose
    # inputs =>  {featureName: featureVal}
    def inference(self, inputs, is_train=True):
        features = self.embedding_combiner(inputs, is_train=is_train)

        # input_layer = Input(shape=(num_features,))
        with tf.variable_scope("mmoe_layers"):
            mmoe_layers =self.expert_gate(features, self.hidden_units_bottom, self.dropout_keep_prob_list_bottom, num_experts=self.num_experts,num_tasks=2, is_train=is_train)

        output_layers = []

        output_info = ['click', 'order']

        # Build tower layer from MMoE layer
        for index, task_layer in enumerate(mmoe_layers):
            output_layer = self.build_tower(task_layer, self.hidden_units_task, self.dropout_keep_prob_list_task, output_info[index], is_train=is_train)
            output_layers.append(output_layer)

        return output_layers[0], output_layers[1]