

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.saved_model import (
            signature_constants, signature_def_utils, tag_constants)

import sys
import os
import subprocess

file_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(file_path + '/../parse')
sys.path.append(file_path + '/../conf')
sys.path.append(file_path + '/../util')
from util import *
from data_feed import index_tables as lookup
from .preprocess import vec_constant
from model import inference_mlp as inference

def export_model(wnd_conf, ckpt_name=None):
    # Build inference model.
    # dense feat with sparse
    with tf.Graph().as_default() as graph:
        tables = lookup.LookupTables(wnd_conf)

        feature_values = tf.placeholder(tf.float32, [None], name="dense_values_placeholder")
        feature_indices = tf.placeholder(tf.int16, [None], name="dense_indices_placeholder")
        dense_shape = tf.to_int64(feature_indices[0:2])
        dense_real_indice = tf.reshape(tf.to_int64(feature_indices[2:]), (tf.shape(feature_values)[0], 2))
        batch_size = tf.placeholder(tf.int32, shape=(), name="batch_size_placeholder")
        feature = tf.SparseTensor(indices=dense_real_indice, values=feature_values, dense_shape=dense_shape)
        feature = tf.sparse.to_dense(feature)
        feature = tf.reshape(feature, (batch_size, wnd_conf[MODEL][FEAT_DIM]))
        input_feature = {'BatchSize': batch_size}
        save_model_input = {"dense_values_placeholder": feature_values, 'batch_size_placeholder': batch_size,
                            "dense_indices_placeholder": feature_indices}

        iside_index = 0
        uside_index = 0
        for emb in wnd_conf.embedding_list:
            id_feat_name = emb[3]
            id_type = emb[4]
            id_feat_wts_name = id_feat_name + 'Wts'
            if id_type == 'i':
                id_values_placeholder_name = "emb_values_placeholder_" + str(iside_index)
                id_wts_placeholder_name = "emb_wts_placeholder_" + str(iside_index)
                id_indices_placeholder_name = "emb_indices_placeholder_" + str(iside_index)
                id_values_placeholder = tf.placeholder(tf.string, [None], name=id_values_placeholder_name)
                id_indices_placeholder = tf.placeholder(tf.int16, [None], name=id_indices_placeholder_name)
                id_wts_placeholder = tf.placeholder(tf.float32, [None], name=id_wts_placeholder_name)
                id_shape = tf.to_int64(id_indices_placeholder[0:2])
                id_real_indices = tf.reshape(tf.to_int64(id_indices_placeholder[2:]),
                                             (tf.shape(id_values_placeholder)[0], 2))
                input_feature[id_feat_name] = tf.SparseTensor(indices=id_real_indices,
                                                              values=tables.inf_transform(id_feat_name,
                                                                                          id_values_placeholder),
                                                              dense_shape=id_shape)
                input_feature[id_feat_wts_name] = tf.SparseTensor(indices=id_real_indices, values=id_wts_placeholder,
                                                                  dense_shape=id_shape)
                save_model_input[id_values_placeholder_name] = id_values_placeholder
                save_model_input[id_wts_placeholder_name] = id_wts_placeholder
                save_model_input[id_indices_placeholder_name] = id_indices_placeholder
                iside_index += 1
            elif id_type == 'u':
                id_values_placeholder_name = "emb_common_values_placeholder_" + str(uside_index)
                id_wts_placeholder_name = "emb_common_wts_placeholder_" + str(uside_index)
                id_values_placeholder = tf.placeholder(tf.string, [None], name=id_values_placeholder_name)

                model_type = wnd_conf[MODEL][MODEL_TYPE]
                if (model_type == "din_v2"):
                    print("model_type:din_v2")
                    id_values_splited = tf.string_split(id_values_placeholder, '&').values
                    # id_wts_placeholder = tf.placeholder(tf.float32, [None], name=id_wts_placeholder_name)
                    # id_wts_placeholder = tf.identity(tf.Print(id_wts_placeholder ,[batch_size, id_feat_name, id_values_placeholder , id_values_splited,id_wts_placeholder],summarize=600000))
                    input_feature[id_feat_name] = tables.inf_transform(id_feat_name, id_values_splited)
                else:
                    input_feature[id_feat_name] = tables.inf_transform(id_feat_name, id_values_placeholder)

                id_wts_placeholder = tf.placeholder(tf.float32, [None], name=id_wts_placeholder_name)
                input_feature[id_feat_wts_name] = id_wts_placeholder
                save_model_input[id_values_placeholder_name] = id_values_placeholder
                save_model_input[id_wts_placeholder_name] = id_wts_placeholder
                uside_index += 1

        norm_vec_const, std_list = vec_constant(wnd_conf)
        inference_const_vec = tf.constant(value=norm_vec_const, dtype=tf.float32)
        std = tf.constant(value=std_list, dtype=tf.float32)
        epsilon = tf.constant(0.0000001, dtype=tf.float32, shape=[len(std_list)])

        clip_feature = tf.clip_by_value(feature, clip_value_min=0.0, clip_value_max=sys.float_info.max)
        normalize_feature = tf.subtract(tf.div(tf.multiply(clip_feature, std),
                                               tf.square(tf.add_n([std, epsilon])) * 3.0), inference_const_vec)

        clip_normalize_feature = tf.clip_by_value(normalize_feature, clip_value_min=-0.99, clip_value_max=0.99)

        input_feature["features"] = clip_normalize_feature
        # Run inference.
        # Run inference.
        inf = inference.Inference(wnd_conf)
        # tf.Print(clip_normalize_feature,[clip_normalize_feature],summarize=600000)
        with tf.variable_scope("DnnModel"):
            #logits = inf.online_inference(input_feature)
            click_logit, order_logit = inf.online_inference(input_feature)
        #scores = tf.reshape(tf.sigmoid(logits), [-1], name="Scores")

        click_scores = tf.reshape(tf.sigmoid(click_logit), [-1], name="click_Scores")
        order_scores = tf.reshape(tf.sigmoid(order_logit), [-1], name="order_Scores")
        weights = wnd_conf[EXPORT_MODEL][EXPORT_WEIGHT]
        print("*"*100)
        print(weights)
        print("*" * 100)
        scores = tf.divide(tf.add(weights[0]*click_scores, weights[1]*order_scores), sum(weights), name='Scores')

        ## gpu settings
        os.environ['CUDA_VISIBLE_DEVICES'] = wnd_conf[MODEL][GPU_VISIBLE]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        output_path = wnd_conf[PATH][MODEL_FROZEN_PATH]
        del_path(output_path)
        exporter = tf.saved_model.Builder(output_path)
        with tf.Session(graph=graph) as sess:
            # sess.run(legacy_init_op)
            ckpt_path = wnd_conf[PATH][MODEL_PATH] + ckpt_name
            print("Restore from ckpt_path: ", ckpt_path)
            saver.restore(sess, ckpt_path)
            signature_def_map = {
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def_utils.predict_signature_def(
                    save_model_input, {"Scores": scores})}
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            exporter.add_meta_graph_and_variables(sess, tags=[tag_constants.SERVING],
                                                  signature_def_map=signature_def_map, main_op=legacy_init_op,
                                                  clear_devices=True)
            exporter.save()
        print('Successfully exported model to %s' % output_path)
