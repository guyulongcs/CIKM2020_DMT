from __future__ import print_function

from tensorflow.python.tools import freeze_graph

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
            signature_constants, signature_def_utils, tag_constants, utils)

import sys
import os

file_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(file_path + '/../util')
from util import *

def vec_constant(wnd_conf):
    #(feature, mean, std)
    norm_vec_const = []
    mean_list = get_const_data(wnd_conf[PATH][TRAIN_DATA_MEAN_PATH])
    std_list = get_const_data(wnd_conf[PATH][TRAIN_DATA_STD_PATH])
    mean = tf.constant(value=mean_list,dtype=tf.float64)
    std =  tf.constant(value=std_list,dtype=tf.float64)
    epsilon = tf.constant(0.0000001,dtype=tf.float64,shape=[len(mean_list)])

    mean_std_element_wise_mul = tf.multiply(mean,std)
    std_add_epsilon = tf.add_n([std,epsilon])
    std_add_epsilon_square = tf.square(std_add_epsilon)

    div1 = tf.div(mean_std_element_wise_mul,std_add_epsilon_square * 3)
    div2 = tf.div(mean_std_element_wise_mul,std_add_epsilon)
    add_div1_div2 = tf.add_n([div1,div2])
    inference_constant_vec = tf.subtract(add_div1_div2,mean)

    sess = tf.Session()
    for i in range(1):
      x = sess.run([inference_constant_vec])

      norm_vec_const.extend(x[0])
    return norm_vec_const, std_list
