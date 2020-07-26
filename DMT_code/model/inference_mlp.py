from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
import importlib

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
sys.path.append('../util')
from util import *

class Inference(object):
	"""docstring for Inference"""
	def __init__(self, wnd_conf):
		super(Inference, self).__init__()
		# data member: self.wnd_conf
		self.wnd_conf       = wnd_conf
		# data member: self.model_type
		self.model_type     = wnd_conf[MODEL][MODEL_TYPE]

		self.module = importlib.import_module('.%s' % self.model_type, 'net')

		if self.model_type == "mlp":
			self.model = self.module.mlp(wnd_conf)
		elif self.model_type == "id_mlp":
			self.model = self.module.id_mlp(wnd_conf)
		elif self.model_type == "embed_mlp":
			self.model = self.module.embed_mlp(wnd_conf)
		elif self.model_type == "embed_mlp_mulnet":
			self.model = self.module.embed_mlp_mulnet(wnd_conf)
		elif self.model_type == "embed_mlp_unbias":
			self.model = self.module.embed_mlp_unbias(wnd_conf)
		elif self.model_type == 'din':
			self.model = self.module.din(wnd_conf)
		elif self.model_type == 'din_id':
			self.model = self.module.din_id(wnd_conf)
		elif self.model_type == 'din_v2':
			self.model = self.module.din_v2(wnd_conf)
		elif self.model_type == 'dien':
			self.model = self.module.dien(wnd_conf)
		elif self.model_type == 'transformer':
			self.model = self.module.transformer(wnd_conf)
		elif self.model_type == 'dien_v2':
			self.model = self.module.dien_v2(wnd_conf)
		elif self.model_type == "wnd":
			self.model = self.module.wnd(wnd_conf)
		elif self.model_type == "dcn":
			self.model = self.module.dcn(wnd_conf)
		elif self.model_type == "lr":
			self.model = self.module.lr(wnd_conf)
		elif self.model_type == "multi_task":
			self.model = self.module.multi_task(wnd_conf)
		elif self.model_type == "mmoe":
			self.model = self.module.mmoe(wnd_conf)
		elif self.model_type == "multi_task_transformer":
			self.model = self.module.multi_task_transformer(wnd_conf)
		elif self.model_type == "mmoe_transformer":
			self.model = self.module.mmoe_transformer(wnd_conf)
		elif self.model_type == "mmoe_transformer_unbias":
			self.model = self.module.mmoe_transformer_unbias(wnd_conf)

		else:
			print("Unknown model, exit now")
			exit(1)

	def embedding_update(self, sess):
		self.model.embedding_update(sess)

	def online_build_sparsetensor(self, inputs, is_train = False):
		merged_user_sp = {}
		new_inputs = {}
		for emb in self.wnd_conf.embedding_list:
			feature_name = emb[3]
			id_type = emb[4]
			feature_wts_name = feature_name + "Wts"

			if id_type == 'u':
				sku_index = tf.expand_dims(tf.range(inputs['BatchSize']),1)
				index_0col = tf.expand_dims(tf.reshape(tf.tile(sku_index,[1,tf.size(inputs[feature_name])]),[-1]) ,1)
				index_1col = tf.expand_dims(tf.tile(tf.range(tf.size(inputs[feature_name])),[inputs['BatchSize']]),1)
				indices = tf.to_int64(tf.concat(values = [index_0col,index_1col], axis=1))
				values = tf.tile(inputs[feature_name],[inputs['BatchSize']])
				wts_values = tf.tile(inputs[feature_wts_name],[inputs['BatchSize']])
				dense_shape = [inputs['BatchSize'],tf.size(inputs[feature_name])]

				user_sp = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

				if (self.model_type == "din_v2"):
					wts_index_0col = tf.expand_dims(
						tf.reshape(tf.tile(sku_index, [1, tf.size(inputs[feature_wts_name])]), [-1]), 1)
					wts_index_1col = tf.expand_dims(
						tf.tile(tf.range(tf.size(inputs[feature_wts_name])), [inputs['BatchSize']]), 1)
					wts_indices = tf.to_int64(tf.concat(values=[wts_index_0col, wts_index_1col], axis=1))
					wts_dense_shape = [inputs['BatchSize'], tf.size(inputs[feature_wts_name])]

					user_sp_wts = tf.SparseTensor(indices=wts_indices, values=wts_values, dense_shape=wts_dense_shape)
				else:
					user_sp_wts = tf.SparseTensor(indices=indices, values=wts_values, dense_shape=dense_shape)

				merged_user_sp[feature_name] = user_sp
				merged_user_sp[feature_wts_name] = user_sp_wts

		for k,v in inputs.items():
			if k in  merged_user_sp:
				new_inputs[k] = merged_user_sp[k]
			else:
				new_inputs[k] = v

		return self.inference(new_inputs, is_train=is_train, is_predict=True)

	# inference determines which model to choose
	# inputs =>  {featureName: featureVal}
	def inference(self, inputs, is_train=True, is_predict=False):
		return self.model.inference(inputs,is_train,is_predict)

	# inference determines which model to choose
	# inputs =>  {featureName: featureVal}
	def online_inference(self, inputs, is_train = False):
		if self.model_type == "dcn" or \
				self.model_type == "embed_mlp" or \
				self.model_type == "embed_mlp_mulnet" or \
				self.model_type == "mmoe" or \
				self.model_type == "mmoe_transformer" or \
				self.model_type == "mmoe_transformer_unbias" or \
				self.model_type == "multi_task" or \
				self.model_type == "multi_task-transformer" or \
				self.model_type == "din" or \
				self.model_type == "din_v2" or \
				self.model_type == "dien" or \
				self.model_type == "dien_v2" or \
				self.model_type == "transformer":
			return self.online_build_sparsetensor(inputs, is_train=is_train)
		elif self.model_type == "embed_mlp_recall":
			return self.embedding_mlp_recall(inputs, is_train=is_train)
		elif self.model_type == "mlp":
			return self.inference(inputs, is_train=is_train)
		else:
			print("Unknown model, exit now")
			exit(1)

	# loss function
	def loss(self, logits, labels, mask, is_train=True):
		xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(logits, [-1]), labels=labels)

		with tf.name_scope("loss"):
			if is_train:
				mask_weight = mask * self.wnd_conf[CLASS_WEIGHT][TRAIN_WEIGHT]
				entropy_mat = tf.transpose(mask_weight) * xentropy
				loss = tf.reduce_sum(tf.reduce_mean(entropy_mat, axis=1))
				#tf.Print(entropy_mat,[xentropy,mask,mask_weight,entropy_mat],summarize=600000)
			else:
				mask_weight = mask * self.wnd_conf[CLASS_WEIGHT][VALID_WEIGHT]
				entropy_mat = tf.transpose(mask_weight) * xentropy
				loss = tf.reduce_sum(tf.reduce_mean(entropy_mat, axis=1))

		return loss

	def cal_cross_entropy(self, output, labels):
		# output:[B]   probability from sigmoid
		# labels: [B]  ground truth
		p = tf.reshape(output, [-1, 1])
		p = tf.concat([1 - p, p], axis=-1)
		r = tf.keras.backend.sparse_categorical_crossentropy(output=p, target=labels, from_logits=False)
		return r

	def loss_multi_task_unbias(self, logits, labels, mask, is_train=True, loss_unbias_method="two_head_add", loss_ctr_rel_method="ctr"):
		return self.logit_loss_unbias(logits, labels, mask, is_train, loss_unbias_method, loss_ctr_rel_method)

	def logit_loss_unbias(self, logits, labels, mask, is_train, loss_unbias_method, loss_ctr_rel_method):
		((click_logit, order_logit), y_bias) = logits

		if (loss_unbias_method == "two_head_multiply"):
			p_ctr = tf.sigmoid(click_logit) * tf.sigmoid(y_bias)
			p_cvr = tf.sigmoid(order_logit) * tf.sigmoid(y_bias)

		if (loss_unbias_method == "two_head_add"):
			p_ctr = tf.sigmoid(click_logit + y_bias)
			p_cvr = tf.sigmoid(order_logit + y_bias)

		p_rel_ctr = tf.sigmoid(click_logit)
		p_rel_cvr = tf.sigmoid(order_logit)

		print("logit_loss...")
		print("wnd_conf[PARAMETER][LOSS_WEIGHT]:", self.wnd_conf[PARAMETER][LOSS_WEIGHT])
		print("wnd_conf[CLASS_WEIGHT][WEIGHT_CTR]:", self.wnd_conf[CLASS_WEIGHT][WEIGHT_CTR])
		print("wnd_conf[CLASS_WEIGHT][WEIGHT_ECVR]:", self.wnd_conf[CLASS_WEIGHT][WEIGHT_ECVR])

		labels_clk = tf.reduce_sum(mask[:, 1:5], axis=-1)
		labels_order = tf.add(mask[:, 3], mask[:, 4])

		with tf.name_scope("click_loss"):
			#xentropy_clk = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(click_logit, [-1]), labels=tf.reshape(labels_clk, [-1]))
			xentropy_clk = self.cal_cross_entropy(output=p_ctr, labels=tf.reshape(labels_clk, [-1]))
			xentropy_clk_rel = self.cal_cross_entropy(output=p_rel_ctr, labels=tf.reshape(labels_clk, [-1]))
			if (loss_ctr_rel_method == "ctr_rel"):
				xentropy_clk += xentropy_clk_rel
			mask_weight = mask * self.wnd_conf[CLASS_WEIGHT][WEIGHT_CTR]
			entropy_mat_clk = tf.transpose(mask_weight) * xentropy_clk
			loss_clk = tf.reduce_sum(tf.reduce_mean(entropy_mat_clk, axis=1))


		with tf.name_scope("order_loss"):
			#xentropy_ord = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(order_logit, [-1]), labels=tf.reshape(labels_order, [-1]))
			xentropy_ord = self.cal_cross_entropy(output=p_cvr, labels=tf.reshape(labels_order, [-1]))
			xentropy_ord_rel = self.cal_cross_entropy(output=p_rel_cvr, labels=tf.reshape(labels_order, [-1]))
			if (loss_ctr_rel_method == "ctr_rel"):
				xentropy_ord += xentropy_ord_rel
			mask_weight = mask * self.wnd_conf[CLASS_WEIGHT][WEIGHT_ECVR]
			entropy_mat_ord = tf.transpose(mask_weight) * xentropy_ord
			loss_order = tf.reduce_sum(tf.reduce_mean(entropy_mat_ord, axis=1))

		if self.wnd_conf[PARAMETER][LOSS_WEIGHT_METHOD] == 'uncertainty':
			print("=========================using uncertainty weights======================")
			return tf.exp(-self.model.click_weight) * loss_clk + 0.5 * self.model.click_weight + \
			       tf.exp(-self.model.order_weight) * loss_order + 0.5 * self.model.order_weight
		else:
			print("=========================using fixed weights======================")
			return self.wnd_conf[PARAMETER][LOSS_WEIGHT][0] * loss_clk + \
			       self.wnd_conf[PARAMETER][LOSS_WEIGHT][1] * loss_order

	def loss_multi_task(self, logits, labels, mask, is_train=True):
		return self.logit_loss(logits, labels, mask, is_train)

	def logit_loss(self, logits, labels, mask, is_train=True):
		(click_logit, order_logit) = logits
		print("logit_loss...")
		print("wnd_conf[PARAMETER][LOSS_WEIGHT]:", self.wnd_conf[PARAMETER][LOSS_WEIGHT])
		print("wnd_conf[CLASS_WEIGHT][WEIGHT_CTR]:", self.wnd_conf[CLASS_WEIGHT][WEIGHT_CTR])
		print("wnd_conf[CLASS_WEIGHT][WEIGHT_ECVR]:", self.wnd_conf[CLASS_WEIGHT][WEIGHT_ECVR])

		labels_clk = tf.reduce_sum(mask[:, 1:5], axis=-1)
		labels_order = tf.add(mask[:, 3], mask[:, 4])

		with tf.name_scope("click_loss"):
			xentropy_clk = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(click_logit, [-1]), labels=tf.reshape(labels_clk, [-1]))
			mask_weight = mask * self.wnd_conf[CLASS_WEIGHT][WEIGHT_CTR]
			entropy_mat_clk = tf.transpose(mask_weight) * xentropy_clk
			loss_clk = tf.reduce_sum(tf.reduce_mean(entropy_mat_clk, axis=1))


		with tf.name_scope("order_loss"):
			xentropy_ord = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(order_logit, [-1]), labels=tf.reshape(labels_order, [-1]))
			mask_weight = mask * self.wnd_conf[CLASS_WEIGHT][WEIGHT_ECVR]
			entropy_mat_ord = tf.transpose(mask_weight) * xentropy_ord
			loss_order = tf.reduce_sum(tf.reduce_mean(entropy_mat_ord, axis=1))

		if self.wnd_conf[PARAMETER][LOSS_WEIGHT_METHOD] == 'uncertainty':
			print("=========================using uncertainty weights======================")
			return tf.exp(-self.model.click_weight) * loss_clk + 0.5 * self.model.click_weight + \
			       tf.exp(-self.model.order_weight) * loss_order + 0.5 * self.model.order_weight
		else:
			print("=========================using fixed weights======================")
			return self.wnd_conf[PARAMETER][LOSS_WEIGHT][0] * loss_clk + \
			       self.wnd_conf[PARAMETER][LOSS_WEIGHT][1] * loss_order

	def l2_norm(self, inputs):
		return self.model.l2_norm(inputs)

	# optimizer
	def get_optimizer(self, optimizer, learning_rate):
		print("Use the optimizer: {}".format(optimizer))
		if optimizer == "sgd":
			return tf.train.GradientDescentOptimizer(learning_rate)
		elif optimizer == "adadelta":
			return tf.train.AdadeltaOptimizer(learning_rate)
		elif optimizer == "adagrad":
			return tf.train.AdagradOptimizer(learning_rate)
		elif optimizer == "adam":
			return tf.train.AdamOptimizer(learning_rate)
		elif optimizer == "ftrl":
			return tf.train.FtrlOptimizer(learning_rate)
		elif optimizer == "rmsprop":
			return tf.train.RMSPropOptimizer(learning_rate)
		else:
			print("Unknow optimizer, exit now")
			exit(1)

