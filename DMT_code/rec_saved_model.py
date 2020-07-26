#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-
#

from __future__ import print_function

from tensorflow.python.tools import freeze_graph

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
            signature_constants, signature_def_utils, tag_constants, utils)

import sys
import os

file_path = os.path.dirname(os.path.abspath(__file__))

from parse import parse
sys.path.append(file_path + '/conf')
import recsys_conf as conf
sys.path.append(file_path + '/util')
from util import *
from saved_model.export_embed_mlp_recall import export_embed_mlp_recall

from saved_model.export_model import export_model
from saved_model.export_model_imp import export_model_imp

def main(unused_argv=None):
  # args is a dict
  args = parse.argument_parse()
  wnd_conf = conf.Conf(
          conf_path=args['conf_path'], 
          conf_file=args['conf_file'])
  if wnd_conf[MODEL][MODEL_TYPE] == 'embed_mlp_recall':
      export_embed_mlp_recall(wnd_conf, args['model_ckpt'])
  else:
      #export_model_imp(wnd_conf, args['model_ckpt'])
      export_model(wnd_conf, args['model_ckpt'])


if __name__ == '__main__':
  tf.app.run()
