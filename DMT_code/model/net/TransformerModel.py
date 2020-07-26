# -*- coding: utf-8 -*-
# @Time    : 2020-01-01 14:24
# @Author  : guyulong@jd.com
# @File    : TransformerModel.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf


from TransformerModel_util import  ff, positional_encoding, positional_encoding_learn, multihead_attention, ln

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
sys.path.append('../../util')
from util import *


class TransformerModel():
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        #d_model
        #maxlen_k
        #maxlen_q
        #dropout_rate
        #num_heads
        #d_ff: d_model*4
        #num_blocks_encode
        #num_blocks_decode
        


    def encode_decode(self, input, name="encode_decode", training=True):
        with tf.variable_scope(name):
            (seq_q, seq_q_lens, seq_k, seq_k_lens, seq_k_ts) = input


            state_encode, state_lens =self.encode((seq_k, seq_k_lens, seq_k_ts), name, training=training)
            state_decode=self.decode((seq_q, seq_q_lens, state_encode, seq_k_lens), name, training=training)
            state_decode=tf.squeeze(state_decode, axis=1)
            return state_decode

    def position_encode(self, seq_k, seq_k_ts, seq_max_len):
        # position_encoding_method: ["position_sin_cos", "position_learn", "time_add", "time_concat"]
        if (self.hp.position_encoding_method == "position_sin_cos"):
            print("Match_position_encode_method {0}".format(self.hp.position_encoding_method))
            seq_k += positional_encoding(seq_k, seq_max_len, masking=False, scope="positional_encoding_k_position_sin_cos")

        if (self.hp.position_encoding_method == "position_learn"):
            print("Match_position_encode_method {0}".format(self.hp.position_encoding_method))
            seq_k += positional_encoding_learn(seq_k, seq_max_len, masking=False, scope="positional_encoding_k_position_learn")

        if (self.hp.is_use_seq_ts and (seq_k_ts is not None) and (self.hp.position_encoding_method == "time_add")):
            print("Match_position_encode_method {0}".format(self.hp.position_encoding_method))
            seq_k_ts = tf.layers.dense(seq_k_ts, self.hp.d_model, name='dense_trans_seq_time_add')
            seq_k += seq_k_ts

        if (self.hp.is_use_seq_ts and (seq_k_ts is not None) and (self.hp.position_encoding_method == "time_concat")):
            print("Match_position_encode_method {0}".format(self.hp.position_encoding_method))
            seq_k = tf.concat([seq_k, seq_k_ts], axis=-1)
            seq_k = tf.layers.dense(seq_k, self.hp.d_model, name='dense_trans_seq_time_concat')

        print("position_encode seq_k:", seq_k)
        return seq_k

    def encode(self, xs, name="encoder", training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            seq_emb, seqlens, seq_k_ts = xs

            # src_masks
            #src_masks = seqmask

            # embedding
            enc = seq_emb
            enc *= self.hp.d_model**0.5 # scale

            #enc += positional_encoding(enc, self.hp.maxlen_k)
            enc = self.position_encode(enc, seq_k_ts, seq_max_len=self.hp.maxlen_k)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            enc_lens = seqlens

            ## Blocks
            for i in range(self.hp.num_blocks_encode):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              queries_length=enc_lens,
                                              keys_length=enc_lens,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="self-attention"
                                              )
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, seqlens

    def decode(self, ys, name="decoder", training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            #seq_q, state_encode, state_mask
            query_emb, query_length, key_emb, key_length = ys
            #query_length = tf.ones_like(query_emb[:, 0, 0], dtype=tf.int32)
            #query_mask = tf.ones_like(query_emb[:, 0, 0], dtype=tf.int32)
            #print("query_mask:", query_mask)
            #query_mask = tf.expand_dims(query_mask, axis=1)
            #print("query_mask:", query_mask)

            # embedding
            dec = query_emb
            dec *= self.hp.d_model ** 0.5  # scale
            if(self.hp.is_decoder_add_pos_emb):
                dec += positional_encoding(dec, self.hp.maxlen_q, masking=False, scope="positional_encoding")

            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks_decode):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=key_emb,
                                              values=key_emb,
                                              queries_length=query_length,
                                              keys_length=key_length,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        y_dec = dec
        return y_dec
