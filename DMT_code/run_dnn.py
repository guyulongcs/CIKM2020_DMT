from __future__ import print_function, absolute_import, division

import datetime
import time
import os
import tensorflow as tf
import sys
import subprocess
import pandas as pd

file_path = os.path.dirname(os.path.abspath(__file__))

from parse import parse

sys.path.append(file_path + '/conf')
import recsys_conf as conf

from data_feed import tfrecord_mask as tfrecord
from data_feed import index_tables as lookup
# from metrics import metrics, metrics3, metrics2
from metrics import metrics
from model import inference_mlp as inference

sys.path.append(file_path + '/util')
from util import *


def log_to_file(info_str, file_name):
    if file_name.startswith("hdfs") or file_name.startswith("/user"):
        subprocess.call(["echo '%s' | hadoop fs -appendToFile - %s" % (
            info_str, file_name)], shell=True)
    else:
        subprocess.call(["echo '%s' >> %s" % (info_str, file_name)], shell=True)


def restore_session_from_checkpoint(sess, saver, checkpoint):
    if checkpoint:
        print("Restore session from checkpoint: {}".format(checkpoint))
        saver.restore(sess, checkpoint)
        return True
    else:
        return False


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def average_losses(tower_losses):
    # Average over the 'tower' dimension.
    loss = tf.concat(axis=0, values=tower_losses)
    loss = tf.reduce_mean(loss, 0)
    return loss


def cal_ctr_cvr_unibas(y_rel, y_bias, loss_unbias_method="two_head_add"):
    click_logit, order_logit = y_rel
    if (loss_unbias_method == "two_head_multiply"):
        p_ctr = tf.sigmoid(click_logit) * tf.sigmoid(y_bias)
        p_cvr = tf.sigmoid(order_logit) * tf.sigmoid(y_bias)

    if (loss_unbias_method == "two_head_add"):
        p_ctr = tf.sigmoid(click_logit + y_bias)
        p_cvr = tf.sigmoid(order_logit + y_bias)

    return (p_ctr, p_cvr)
    pass


def cal_ctr_cvr(y_rel):
    click_logit, order_logit = y_rel
    p_ctr = tf.sigmoid(click_logit)
    p_cvr = tf.sigmoid(order_logit)
    return (p_ctr, p_cvr)


def train(wnd_conf, ckpt_name=None):
    print("If this is the training process:", wnd_conf[INFO][TYPE] == 'train')
    tables = lookup.LookupTables(wnd_conf)

    ## read train data
    per_tower_data = tfrecord.get_multi_towers_batch(wnd_conf, lookup_tables=tables)

    ## step count and learning rate
    step = 0
    if "current" not in ckpt_name:
        step = int(ckpt_name.split("-")[1])
    global_step = tf.Variable(step, name="global_step", trainable=False)

    # decay learning rate
    learning_rate = tf.train.piecewise_constant(global_step,
                                                wnd_conf[MODEL][STEP_BOUNDARY], wnd_conf[MODEL][LEARNING_RATE])

    ## inference and get an optimizer that performs gradient descent.
    inf = inference.Inference(wnd_conf)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = inf.get_optimizer(wnd_conf[MODEL][OPTIMIZER], learning_rate)

    # Calculate the predict for each model tower.
    tower_eval_predict = []
    tower_eval_predict_score = []
    tower_labels = []

    tower_click_predict_score = []
    tower_order_predict_score = []

    tower_click_label_predict = []
    tower_order_label_predict = []

    # Calculate the gradients for each model tower.
    tower_grads = []
    tower_losses = []
    for gpu in wnd_conf[MODEL][GPU_VISIBLE].split(','):
        with tf.device('/gpu:%s' % gpu):
            with tf.variable_scope("DnnModel", reuse=(int(gpu) > 0)):
                # get one batch for the GPU
                tower_batch_labels, tower_batch_mask, tower_batch_features = per_tower_data[gpu]

                tower_train_logits = inf.inference(tower_batch_features, is_train=True)
                if (wnd_conf.is_unbias_model):
                    tower_train_loss = inf.loss_multi_task_unbias(tower_train_logits, tower_batch_labels,
                                                                  tower_batch_mask, is_train=True,
                                                                  loss_unbias_method=wnd_conf[MODEL][loss_unbias_method],
                                                                  loss_ctr_rel_method=wnd_conf[MODEL][LOSS_CTR_REL_METHOD])
                else:
                    tower_train_loss = inf.loss_multi_task(tower_train_logits, tower_batch_labels, tower_batch_mask,
                                                           is_train=True, propensity_weight_mul=tower_batch_features[
                            "propensity_weight_mul"])
                print("tower_train_loss:", tower_train_loss)

                if (wnd_conf.is_unbias_model):
                    y_rel, y_bias = tower_train_logits
                    click_logit, order_logit = y_rel
                    (p_ctr, p_cvr) = cal_ctr_cvr_unibas(y_rel, y_bias,
                                                        loss_unbias_method=wnd_conf[MODEL][loss_unbias_method])
                else:
                    p_ctr, p_cvr = cal_ctr_cvr(tower_train_logits)

                if (wnd_conf[MODEL][WND_WD] > 0.00001):
                    tower_train_loss = tower_train_loss + inf.l2_norm(tower_batch_features)

                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                # Calculate the gradients for the batch of data on this tower.
                grads = opt.compute_gradients(tower_train_loss)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)

                # Keep track of the loss across all towers
                tower_losses.append(tf.expand_dims(tower_train_loss, 0))

                tower_train_eval_click_sigmoid = p_ctr
                tower_click_predict = tf.cast(tf.greater(tower_train_eval_click_sigmoid, tf.constant(0.5)), tf.float32)
                tower_click_predict_score.append(tower_train_eval_click_sigmoid)
                tower_click_label_predict.append(tower_click_predict)

                tower_train_eval_order_sigmoid = p_cvr
                tower_order_predict = tf.cast(tf.greater(tower_train_eval_order_sigmoid, tf.constant(0.5)), tf.float32)
                tower_order_predict_score.append(tower_train_eval_order_sigmoid)
                tower_order_label_predict.append(tower_order_predict)

                tower_labels.append(tower_batch_mask)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    batch_train_loss = average_losses(tower_losses)

    # Apply the gradients to adjust the shared variables.
    train_op = opt.apply_gradients(grads, global_step=global_step)

    train_click_predict = tf.concat(axis=0, values=tower_click_label_predict)
    train_click_predict_score = tf.concat(axis=0, values=tower_click_predict_score)

    train_order_predict = tf.concat(axis=0, values=tower_order_label_predict)
    train_order_predict_score = tf.concat(axis=0, values=tower_order_predict_score)

    batch_labels = tf.concat(axis=0, values=tower_labels)

    # train metrics
    train_metrics_var_scope = "train_metrics"
    train_mean_loss, train_mean_loss_op = tf.metrics.mean(values=batch_train_loss, name=train_metrics_var_scope)

    train_click_precision, train_click_precision_op = tf.metrics.precision(
        labels=tf.reduce_sum(batch_labels[:, 1:5], axis=-1),
        predictions=train_click_predict[:, 0],
        name=train_metrics_var_scope)
    train_click_recall, train_click_recall_op = tf.metrics.recall(labels=tf.reduce_sum(batch_labels[:, 1:5], axis=-1),
                                                                  predictions=train_click_predict[:, 0],
                                                                  name=train_metrics_var_scope)
    train_click_auc, train_click_auc_op = tf.metrics.auc(labels=tf.reduce_sum(batch_labels[:, 1:5], axis=-1),
                                                         predictions=train_click_predict_score[:, 0],
                                                         name=train_metrics_var_scope)

    train_order_precision, train_order_precision_op = tf.metrics.precision(
        labels=tf.add(batch_labels[:, 3], batch_labels[:, 4]),
        predictions=train_order_predict[:, 0],
        name=train_metrics_var_scope)
    train_order_recall, train_order_recall_op = tf.metrics.recall(labels=tf.add(batch_labels[:, 3], batch_labels[:, 4]),
                                                                  predictions=train_order_predict[:, 0],
                                                                  name=train_metrics_var_scope)
    train_order_auc, train_order_auc_op = tf.metrics.auc(labels=tf.add(batch_labels[:, 3], batch_labels[:, 4]),
                                                         predictions=train_order_predict_score[:, 0],
                                                         name=train_metrics_var_scope)

    tf.summary.scalar('train_click_precision', train_click_precision)
    tf.summary.scalar('train_click_recall', train_click_recall)
    tf.summary.scalar('train_click_auc', train_click_auc)

    tf.summary.scalar('train_order_precision', train_order_precision)
    tf.summary.scalar('train_order_recall', train_order_recall)
    tf.summary.scalar('train_order_auc', train_order_auc)

    # train metrics init op
    train_metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=train_metrics_var_scope)
    train_metrics_init_op = tf.variables_initializer(var_list=train_metrics_vars, name='train_metrics_init')

    ## merge all summaries
    summary_op = tf.summary.merge_all()

    var_list = [var for var in tf.global_variables() if "moving" in var.name]
    var_list += tf.trainable_variables()
    # max_to_keep=0 means to save all checkpoint files
    saver = tf.train.Saver(var_list=var_list, max_to_keep=0)

    ## gpu settings
    os.environ['CUDA_VISIBLE_DEVICES'] = wnd_conf[MODEL][GPU_VISIBLE]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ## gpu number
    gpu_num = len(wnd_conf[MODEL][GPU_VISIBLE].split(','))

    # set the total_data_num to calculate current_epoch_num
    total_data_num = wnd_conf[MODEL][TOTAL_EXAMPLE_NUM]
    current_epoch_num = -1

    if wnd_conf[PARAMETER][LOSS_WEIGHT_METHOD] == 'uncertainty':
        click_weight = tf.get_default_graph().get_tensor_by_name('DnnModel/uncertainty_click_weight:0')
        order_weight = tf.get_default_graph().get_tensor_by_name('DnnModel/uncertainty_order_weight:0')

    with tf.Session(config=config) as sess:
        train_tensorboard_path = wnd_conf[PATH][SUMMARY_PATH] + "/train/"
        # if not os.path.exists(train_tensorboard_path):
        #    os.makedirs(train_tensorboard_path)
        writer = tf.summary.FileWriter(train_tensorboard_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        # reset all train metrics to be zero

        # print model variables
        print("print model variables:")
        tvars = tf.trainable_variables()
        tvars_vals = sess.run(tvars)

        for var, val in zip(tvars, tvars_vals):
            print(var.name, val)  # Prints the name of the variable alongside its value.

        sess.run(train_metrics_init_op)
        if wnd_conf[MODEL][MODEL_TYPE] == "embed_mlp" or wnd_conf[MODEL][MODEL_TYPE] == "embed_mlp_recall":
            inf.embedding_update(sess)

        if ckpt_name != 'model.ckpt-0':
            cur_model_ckpt = wnd_conf[PATH][MODEL_PATH] + ckpt_name
            saver.restore(sess, cur_model_ckpt)
            print("load ckpt for train : %s" % cur_model_ckpt)
        else:
            del_path(wnd_conf[PATH][MODEL_PATH])

        sys.stdout.write("[%s] start training\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        try:
            while step < wnd_conf[MODEL][MAX_ITER_STEP]:
                start_time = time.time()
                # reset all train metrics to be zero
                temp_epoch_num = int((wnd_conf[MODEL][BATCH_SIZE] * gpu_num * step) / total_data_num) + 1
                if temp_epoch_num != current_epoch_num:
                    current_epoch_num = temp_epoch_num
                    print('*' * 210)
                    print('>> Current epoch num:', current_epoch_num)
                    print('*' * 210)

                batch_loss_value, _, step, _, _, _, _, _, _, _ = sess.run(
                    [batch_train_loss, train_op, global_step,
                     train_mean_loss_op, train_click_precision_op,
                     train_click_recall_op, train_click_auc_op,
                     train_order_precision_op,
                     train_order_recall_op, train_order_auc_op])
                if wnd_conf[PARAMETER][LOSS_WEIGHT_METHOD] == 'uncertainty':
                    train_click_precision_value, train_click_recall_value, train_click_auc_value, \
                    train_order_precision_value, train_order_recall_value, train_order_auc_value, train_mean_loss_value, \
                    click_weight_value, order_weight_value = \
                        sess.run([train_click_precision, train_click_recall, train_click_auc,
                                  train_order_precision, train_order_recall, train_order_auc, train_mean_loss,
                                  click_weight, order_weight])
                    sys.stdout.write(
                        "[%s] spent time: %f | batch_train_loss: %f | mean_train_loss: %f | train_click_precision: %f "
                        "| train_click_recall: %f | train_click_auc: %f | "
                        "train_order_precision: %f | "
                        "train_order_recall: %f | train_order_auc: %f | click_weight: %f |"
                        "order_weight: %f  |--- iter: %d | \n" % \
                        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time,
                         batch_loss_value, train_mean_loss_value,
                         train_click_precision_value, train_click_recall_value, train_click_auc_value,
                         train_order_precision_value, train_order_recall_value, train_order_auc_value,
                         click_weight_value, order_weight_value, step))
                else:
                    train_click_precision_value, train_click_recall_value, train_click_auc_value, \
                    train_order_precision_value, train_order_recall_value, train_order_auc_value, train_mean_loss_value, \
                        = \
                        sess.run([train_click_precision, train_click_recall, train_click_auc,
                                  train_order_precision, train_order_recall, train_order_auc, train_mean_loss
                                  ])

                    sys.stdout.write(
                        "[%s] spent time: %f | batch_train_loss: %f | mean_train_loss: %f | train_click_precision: %f | train_click_recall: %f | train_click_auc: %f |train_order_precision: %f | train_order_recall: %f | train_order_auc: %f | iter: %d\n" % \
                        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time,
                         batch_loss_value, train_mean_loss_value,
                         train_click_precision_value, train_click_recall_value, train_click_auc_value,
                         train_order_precision_value, train_order_recall_value, train_order_auc_value,
                         step))

                sys.stdout.write("=" * 100)

                if step == 1 or step % wnd_conf[MODEL][VALIDATE_STEP] == 0:
                    summary = sess.run(summary_op)
                    writer.add_summary(summary, step)

                    metrics_str = "%s" % ('*' * 70) + "\n" + \
                                  ">> iter_steps:" + str(step) + "\n" + \
                                  "batch_train_loss:" + str(batch_loss_value) + "\n" + \
                                  "mean_train_loss:" + str(train_mean_loss_value) + "\n" + \
                                  "train_click_precision:" + str(train_click_precision_value) + "\n" + \
                                  "train_click_recall:" + str(train_click_recall_value) + "\n" + \
                                  "train_click_auc:" + str(train_click_auc_value) + "\n" + \
                                  "train_order_precision:" + str(train_order_precision_value) + "\n" + \
                                  "train_order_recall:" + str(train_order_recall_value) + "\n" + \
                                  "train_order_auc:" + str(train_order_auc_value) + "\n"

                    log_to_file(metrics_str, wnd_conf[PATH][TRAIN_RESULT])

                    print("\n[%s] model saving..." % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    saver.save(sess, wnd_conf[PATH][MODEL_PATH] + 'model.ckpt', global_step=step)
                    create_file(wnd_conf[PATH][MODEL_PATH], 'step-%d.model.DONE' % step)

        except tf.errors.OutOfRangeError:
            print("-----------------------------OutOfRangeError---------------------------------")
            print("TrainEnd!")
        writer.close()
        saver.save(sess, wnd_conf[PATH][MODEL_PATH] + 'model.ckpt', global_step=step)
        create_file(wnd_conf[PATH][MODEL_PATH], 'step-%d.model.DONE' % step)


def get_validation_newest_step(validation_path):
    if validation_path.startswith("hdfs") or validation_path.startswith("/user"):
        p1 = subprocess.Popen("hadoop fs -cat %s | grep iter_steps " % validation_path,
                              shell=True, stdout=subprocess.PIPE)
        steps = p1.stdout.read().decode('utf-8').strip().split("\n")

    else:
        p1 = subprocess.Popen("cat %s | grep iter_steps " % validation_path, shell=True, stdout=subprocess.PIPE)
        (result, error) = p1.communicate()
        steps = result.decode('utf-8').strip().split("\n")
    ## if less than 3, then reset to 0
    if len(steps) < 3:
        return 0
    steps = [int(step.split(":")[1]) for step in steps]
    steps = sorted(steps)
    return steps[-1]


def get_ckpt_from_fs(lower_bound, ckpt_path):
    newest_ckpt = ""
    newest_step = lower_bound
    if ckpt_path.startswith("hdfs") or ckpt_path.startswith("/user"):
        p1 = subprocess.Popen('hdfs dfs -ls ' + ckpt_path, shell=True, stdout=subprocess.PIPE)
        p1 = subprocess.Popen('sort -k6,7', shell=True, stdin=p1.stdout, stdout=subprocess.PIPE)
    else:
        p1 = subprocess.Popen('ls -ltr ' + ckpt_path, shell=True, stdout=subprocess.PIPE)

    files = p1.stdout.read().decode('utf-8').split('\n')
    filenames = [file.split(" ")[-1] for file in files[1:]]
    filenames = [name for name in filenames if "model.ckpt" in name]

    if len(filenames) > 0:
        for fname in filenames:
            step = int(fname.split('model.ckpt-')[-1].split('.')[0])
            if step > lower_bound:
                newest_ckpt = "model.ckpt-%d" % step
                newest_step = step
                break
    return newest_ckpt, newest_step


def validation(wnd_conf):
    step = get_validation_newest_step(wnd_conf[PATH][VALIDATION_RESULT])
    print("get_validation_newest_step: %s" % str(step))

    # if validation data is in hdfs, then get it to local path.
    validation_path = wnd_conf[PATH][VALIDATION_DATA_PATH]
    # if validation_path.startswith("hdfs") or validation_path.startswith("/user"):
    #    validation_path = hdfs_files_to_local(validation_path)

    val_tensorboard_path = wnd_conf[PATH][SUMMARY_PATH] + "/val/"
    # if not os.path.exists(val_tensorboard_path):
    #    os.makedirs(val_tensorboard_path)
    summary_write = tf.summary.FileWriter(val_tensorboard_path)
    while step < wnd_conf[MODEL][MAX_ITER_STEP]:
        newest_ckpt, newest_step = get_ckpt_from_fs(step, wnd_conf[PATH][MODEL_PATH])
        if step == newest_step or not file_exists(wnd_conf[PATH][MODEL_PATH], 'step-%d.model.DONE' % newest_step):
            time.sleep(5)
            continue
        step = newest_step

        tables = lookup.LookupTables(wnd_conf)
        ## read validation data
        validation_labels, validation_header, validation_mask, validation_features = \
            tfrecord.get_val_test_batch(file_path=validation_path,
                                        EPOCH_NUM=1, batch_size=wnd_conf[MODEL][VALIDATION_BATCH_SIZE],
                                        wnd_conf=wnd_conf, lookup_tables=tables)
        inf = inference.Inference(wnd_conf)
        ## do validation
        with tf.variable_scope("DnnModel"):
            validation_logits = inf.inference(validation_features, is_train=False)
            # validation_loss = inf.loss(validation_logits, validation_labels, validation_mask, is_train=False)

            if (wnd_conf.is_unbias_model):
                tower_train_loss = inf.loss_multi_task_unbias(validation_logits, validation_labels, validation_mask,
                                                              is_train=True,
                                                              loss_unbias_method=wnd_conf[MODEL][loss_unbias_method],
                                                              loss_ctr_rel_method=wnd_conf[MODEL][LOSS_CTR_REL_METHOD])
            else:
                tower_train_loss = inf.loss_multi_task(validation_logits, validation_labels, validation_mask,
                                                       is_train=True, propensity_weight_mul=tower_batch_features[
                        "propensity_weight_mul"])

            if (wnd_conf.is_unbias_model):
                y_rel, y_bias = validation_logits
                click_logit, order_logit = y_rel
                (p_ctr, p_cvr) = cal_ctr_cvr_unibas(y_rel, y_bias,
                                                    loss_unbias_method=wnd_conf[MODEL][loss_unbias_method])
            else:
                p_ctr, p_cvr = cal_ctr_cvr(validation_logits)

        validation_click_logits_sigmoid = p_ctr
        validation_click_predict = tf.cast(tf.greater(validation_click_logits_sigmoid, tf.constant(0.5)), tf.float32)

        validation_order_logits_sigmoid = p_cvr
        validation_order_predict = tf.cast(tf.greater(validation_order_logits_sigmoid, tf.constant(0.5)), tf.float32)

        # validation metrics
        validation_metrics_var_scope = "validation_metrics"

        validation_click_precision, validation_click_precision_op = tf.metrics.precision(
            labels=tf.reduce_sum(validation_mask[:, 1:5], axis=-1), predictions=validation_click_predict[:, 0],
            name=validation_metrics_var_scope)
        validation_click_recall, validation_click_recall_op = tf.metrics.recall(
            labels=tf.reduce_sum(validation_mask[:, 1:5], axis=-1), predictions=validation_click_predict[:, 0],
            name=validation_metrics_var_scope)
        validation_click_auc, validation_click_auc_op = tf.metrics.auc(
            labels=tf.reduce_sum(validation_mask[:, 1:5], axis=-1), predictions=validation_click_logits_sigmoid[:, 0],
            name=validation_metrics_var_scope)

        validation_order_precision, validation_order_precision_op = tf.metrics.precision(
            labels=tf.add(validation_mask[:, 3], validation_mask[:, 4]), predictions=validation_order_predict[:, 0],
            name=validation_metrics_var_scope)
        validation_order_recall, validation_order_recall_op = tf.metrics.recall(
            labels=tf.add(validation_mask[:, 3], validation_mask[:, 4]), predictions=validation_order_predict[:, 0],
            name=validation_metrics_var_scope)
        validation_order_auc, validation_order_auc_op = tf.metrics.auc(
            labels=tf.add(validation_mask[:, 3], validation_mask[:, 4]),
            predictions=validation_order_logits_sigmoid[:, 0],
            name=validation_metrics_var_scope)

        validation_mean_loss, validation_mean_loss_op = tf.metrics.mean(
            values=tower_train_loss, name=validation_metrics_var_scope)
        tf.summary.scalar('validation_click_precision', validation_click_precision)
        tf.summary.scalar('validation_click_recall', validation_click_recall)
        tf.summary.scalar('validation_click_auc', validation_click_auc)

        tf.summary.scalar('validation_order_precision', validation_order_precision)
        tf.summary.scalar('validation_order_recall', validation_order_recall)
        tf.summary.scalar('validation_order_auc', validation_order_auc)

        tf.summary.scalar('validation_mean_loss', validation_mean_loss)
        sum_ops = tf.summary.merge_all()

        # validation metric init op
        validation_metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=validation_metrics_var_scope)
        validation_metrics_init_op = tf.variables_initializer(var_list=validation_metrics_vars,
                                                              name='validation_metrics_init')
        ## saver
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        config.gpu_options.allow_growth = True
        # config.allow_soft_placement=True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            restore_session_from_checkpoint(sess, saver, os.path.join(wnd_conf[PATH][MODEL_PATH], newest_ckpt))
            sess.run(validation_metrics_init_op)

            validation_logits_sigmoid_values_list = []

            validation_click_logits_sigmoid_values_list = []
            validation_detail_logits_sigmoid_values_list = []
            validation_order_logits_sigmoid_values_list = []

            header_list = []

            start_time = time.time()
            try:
                while True:
                    validation_click_logits_sigmoid_values, \
                    validation_detail_logits_sigmoid_values, \
                    validation_order_logits_sigmoid_values, \
                    headers_values, _, _, _, _, _, _ = sess.run(
                        [validation_click_logits_sigmoid,
                         validation_order_logits_sigmoid,
                         validation_header,
                         validation_click_precision_op,
                         validation_click_recall_op,
                         validation_click_auc_op,

                         validation_order_precision_op,
                         validation_order_recall_op,
                         validation_order_auc_op,
                         validation_mean_loss_op])
                    validation_click_logits_sigmoid_values_list.extend(validation_click_logits_sigmoid_values[:, 0])
                    validation_detail_logits_sigmoid_values_list.extend(validation_detail_logits_sigmoid_values[:, 0])
                    validation_order_logits_sigmoid_values_list.extend(validation_order_logits_sigmoid_values[:, 0])
                    header_list.extend(headers_values)
            except tf.errors.OutOfRangeError:
                print("-----------------------------OutOfRange---------------------------------")

            validation_click_precision_value, \
            validation_click_recall_value, \
            validation_click_auc_value, \
            validation_order_precision_value, \
            validation_order_recall_value, \
            validation_order_auc_value, \
            validation_mean_loss_value = sess.run([
                validation_click_precision,
                validation_click_recall,
                validation_click_auc,
                validation_order_precision,
                validation_order_recall,
                validation_order_auc,
                validation_mean_loss])

            summary_out = sess.run(sum_ops)
            summary_write.add_summary(summary_out, step)
            sys.stdout.write(
                "[%s] spent time: %f | validation_loss: %f | validation_click_precision: %f | validation_click_recall: %f | validation_click_auc: %f  |validation_order_precision: %f | validation_order_recall: %f | validation_order_auc: %f | iter: %d\n" % \
                (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time,
                 validation_mean_loss_value,
                 validation_click_precision_value,
                 validation_click_recall_value,
                 validation_click_auc_value,
                 validation_order_precision_value,
                 validation_order_recall_value,
                 validation_order_auc_value,
                 step))

            validation_metrics_str = ">> iter_steps:" + str(step) + "\n" + "validation_loss:" + str(
                validation_mean_loss_value) + "\n" + "validation_click_precision:" + str(
                validation_click_precision_value) + "\n" + "validation_click_recall:" + str(
                validation_click_recall_value) + "\n" + "validation_click_auc:" + str(
                validation_click_auc_value) + "\n" + "validation_order_precision:" + str(
                validation_order_precision_value) + "\n" + "validation_order_recall:" + str(
                validation_order_recall_value) + "\n" + "validation_order_auc:" + str(validation_order_auc_value) + "\n"

            log_to_file(validation_metrics_str, wnd_conf[PATH][VALIDATION_RESULT])

            total_logit = []
            for i, k in zip(validation_click_logits_sigmoid_values_list,
                            validation_order_logits_sigmoid_values_list):
                total_logit.append(i + k)
            metric_sets, at_list = metrics.get_offline_metrics(wnd_conf[SCHEMA][HEADER_SCHEMA],
                                                               header_list, total_logit)

            for action, metric in metric_sets.items():
                offline_metrics_str = ''
                metric_threshlod_pair = zip(at_list, metric)
                for tuple0, tuple1 in metric_threshlod_pair:
                    offline_metrics_str += "action_{a}_at_{n}: {m}\n".format(a=action, n=tuple0, m=tuple1)
                log_to_file(offline_metrics_str, wnd_conf[PATH][VALIDATION_RESULT])
        tf.reset_default_graph()
    summary_write.close()
    print("ValidationEnd!")


def predict(wnd_conf, ckpt_name=None, test_tag="", test_score_method=""):
    print("predict...")
    all_test_data_path = wnd_conf[PATH][TEST_DATA_PATH].split(',')
    if (test_tag == "ord"):
        all_test_data_path = wnd_conf[PATH][TEST_DATA_PATH_ORD].split(',')
    print("test_score_method:", test_score_method)
    out_file_test = wnd_conf[PATH][OUTPUT_PATH] + wnd_conf.tag + '.' + 'ckpt-' + \
                    ckpt_name.split('-')[-1] + '.test_result' + '_' + test_tag + "_" + args['test_score_method']

    header_score_file = out_file_test + '.detail'

    print("out_file_test:", out_file_test)
    print("header_score_file:", header_score_file)

    del_path(out_file_test)
    del_path(header_score_file)

    for test_data_path in all_test_data_path:
        tables = lookup.LookupTables(wnd_conf)
        ## read validation data
        test_labels, test_header, test_mask, test_features = tfrecord.get_val_test_batch(file_path=test_data_path,
                                                                                         EPOCH_NUM=1,
                                                                                         batch_size=wnd_conf[MODEL][
                                                                                             TEST_BATCH_SIZE],
                                                                                         wnd_conf=wnd_conf,
                                                                                         lookup_tables=tables)

        ## model
        inf = inference.Inference(wnd_conf)

        ## test
        with tf.variable_scope("DnnModel"):
            test_eval_logits = inf.inference(test_features, is_train=False)
            if (wnd_conf.is_unbias_model):
                test_loss = inf.loss_multi_task_unbias(test_eval_logits, test_labels, test_mask,
                                                       is_train=False,
                                                       loss_unbias_method=wnd_conf[MODEL][loss_unbias_method],
                                                       loss_ctr_rel_method=wnd_conf[MODEL][LOSS_CTR_REL_METHOD])
            else:
                test_loss = inf.loss_multi_task(test_eval_logits, test_labels, test_mask, is_train=False,
                                                propensity_weight_mul=test_features["propensity_weight_mul"])

            if (wnd_conf.is_unbias_model):
                y_rel, y_bias = test_eval_logits
                # click_logit, order_logit = y_rel
                # test_score_method: 'rel' or 'ctr'
                if (test_score_method == "rel"):
                    p_ctr, p_cvr = cal_ctr_cvr(y_rel)
                else:
                    (p_ctr, p_cvr) = cal_ctr_cvr_unibas(y_rel, y_bias,
                                                        loss_unbias_method=wnd_conf[MODEL][loss_unbias_method])
            else:
                p_ctr, p_cvr = cal_ctr_cvr(test_eval_logits)


        # test
        test_click_logits_sigmoid = p_ctr
        test_click_predict = tf.cast(tf.greater(test_click_logits_sigmoid, tf.constant(0.5)), tf.float32)

        test_order_logits_sigmoid = p_cvr
        test_order_predict = tf.cast(tf.greater(test_order_logits_sigmoid, tf.constant(0.5)), tf.float32)

        # test metrics
        test_metrics_var_scope = "test_metrics"

        test_click_precision, test_click_precision_op = tf.metrics.precision(
            labels=tf.reduce_sum(test_mask[:, 1:5], axis=-1),
            predictions=test_click_predict[:, 0],
            name=test_metrics_var_scope)
        test_click_recall, test_click_recall_op = tf.metrics.recall(labels=tf.reduce_sum(test_mask[:, 1:5], axis=-1),
                                                                    predictions=test_click_predict[:, 0],
                                                                    name=test_metrics_var_scope)
        test_click_auc, test_click_auc_op = tf.metrics.auc(labels=tf.reduce_sum(test_mask[:, 1:5], axis=-1),
                                                           predictions=test_click_logits_sigmoid[:, 0],
                                                           name=test_metrics_var_scope)

        test_order_precision, test_order_precision_op = tf.metrics.precision(
            labels=tf.add(test_mask[:, 3], test_mask[:, 4]),
            predictions=test_order_predict[:, 0],
            name=test_metrics_var_scope)
        test_order_recall, test_order_recall_op = tf.metrics.recall(labels=tf.add(test_mask[:, 3], test_mask[:, 4]),
                                                                    predictions=test_order_predict[:, 0],
                                                                    name=test_metrics_var_scope)
        test_order_auc, test_order_auc_op = tf.metrics.auc(labels=tf.add(test_mask[:, 3], test_mask[:, 4]),
                                                           predictions=test_order_logits_sigmoid[:, 0],
                                                           name=test_metrics_var_scope)
        if "mmoe" in wnd_conf[MODEL][MODEL_TYPE]:
            click_weight = tf.get_default_graph().get_tensor_by_name(
                'DnnModel/mmoe_layers/gates-0/gates-layer-0/Softmax:0')
            order_weight = tf.get_default_graph().get_tensor_by_name(
                'DnnModel/mmoe_layers/gates-1/gates-layer-0/Softmax:0')

        test_mean_loss, test_mean_loss_op = tf.metrics.mean( \
            values=test_loss, name=test_metrics_var_scope)

        # test metric init op
        test_metrics_vars = tf.get_collection( \
            tf.GraphKeys.LOCAL_VARIABLES, scope=test_metrics_var_scope)
        test_metrics_init_op = tf.variables_initializer( \
            var_list=test_metrics_vars, name='test_metrics_init')

        tf.summary.scalar('test_click_precision', test_click_precision)
        tf.summary.scalar('test_click_recall', test_click_recall)
        tf.summary.scalar('test_click_auc', test_click_auc)

        tf.summary.scalar('test_order_precision', test_order_precision)
        tf.summary.scalar('test_order_recall', test_order_recall)
        tf.summary.scalar('test_order_auc', test_order_auc)

        ## saver
        saver = tf.train.Saver()

        ## gpu settings
        os.environ['CUDA_VISIBLE_DEVICES'] = wnd_conf[MODEL][GPU_VISIBLE]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            model_ckpt = wnd_conf[PATH][MODEL_PATH] + ckpt_name

            restore_session_from_checkpoint(sess, saver, model_ckpt)
            sess.run(test_metrics_init_op)

            test_click_logits_sigmoid_list = []
            test_order_logits_sigmoid_list = []

            header_list = []
            feed_batch_num = 0
            if "mmoe" in wnd_conf[MODEL][MODEL_TYPE]:
                click_weight_value_list = []
                order_weight_value_list = []
            try:
                while True:
                    # check data feed
                    feed_batch_num = feed_batch_num + 1
                    print('*' * 70)
                    print("predicting data of feed_batch_num:", feed_batch_num)
                    if "mmoe" in wnd_conf[MODEL][MODEL_TYPE]:
                        test_click_logits_sigmoid_values, \
                        test_order_logits_sigmoid_values, \
                        test_header_values, _, _, _, _, _, _, _, click_weight_value, order_weight_value = sess.run(
                            [test_click_logits_sigmoid,
                             test_order_logits_sigmoid,
                             test_header,
                             test_click_precision_op,
                             test_click_recall_op,
                             test_click_auc_op,
                             test_order_precision_op,
                             test_order_recall_op,
                             test_order_auc_op,
                             test_mean_loss_op,
                             click_weight,
                             order_weight])
                    else:
                        test_click_logits_sigmoid_values, \
                        test_order_logits_sigmoid_values, \
                        test_header_values, _, _, _, _, _, _, _ = sess.run(
                            [test_click_logits_sigmoid,
                             test_order_logits_sigmoid,
                             test_header,
                             test_click_precision_op,
                             test_click_recall_op,
                             test_click_auc_op,
                             test_order_precision_op,
                             test_order_recall_op,
                             test_order_auc_op,
                             test_mean_loss_op])

                    # test_eval_logits_sigmoid_values is a ndarray of shape (batch_size, 1)
                    test_click_logits_sigmoid_list.extend(test_click_logits_sigmoid_values[:, 0])
                    test_order_logits_sigmoid_list.extend(test_order_logits_sigmoid_values[:, 0])
                    header_list.extend(test_header_values)
                    if "mmoe" in wnd_conf[MODEL][MODEL_TYPE]:
                        click_weight_value_list.extend(click_weight_value)
                        order_weight_value_list.extend(order_weight_value)

                    test_click_precision_value, test_click_recall_value, test_click_auc_value, \
                    test_order_precision_value, test_order_recall_value, test_order_auc_value, test_loss_value = \
                        sess.run([test_click_precision, test_click_recall, test_click_auc,
                                  test_order_precision, test_order_recall, test_order_auc,
                                  test_mean_loss])
                    print("test_click_precision   :", test_click_precision_value)
                    print("test_click_recall      :", test_click_recall_value)
                    print("test_click_auc         :", test_click_auc_value)
                    print("test_order_precision   :", test_order_precision_value)
                    print("test_order_recall      :", test_order_recall_value)
                    print("test_order_auc         :", test_order_auc_value)
                    print("test_loss        :", test_loss_value)

                # break
            except tf.errors.OutOfRangeError:
                print("-----------------------------OutOfRangeError---------------------------------")

            test_metrics_str = "test_data_path:" + str(test_data_path) + "\n" + "test_click_precision:" + str(
                test_click_precision_value) + "\n" + "test_click_recall:" + str(
                test_click_recall_value) + "\n" + "test_click_auc:" + str(
                test_click_auc_value) + "\n" + "test_order_precision:" + str(
                test_order_precision_value) + "\n" + "test_order_recall:" + str(
                test_order_recall_value) + "\n" + "test_order_auc:" + str(
                test_order_auc_value) + "\n" + "test_loss:" + str(
                test_loss_value) + "\n"

            log_to_file(test_metrics_str, out_file_test)

            offline_metrics_str = "add clk_score/ord_socre: 1/1..."
            log_to_file(offline_metrics_str, out_file_test)
            print(offline_metrics_str)
            total_logit = []
            for i, k in zip(test_click_logits_sigmoid_list, test_order_logits_sigmoid_list):
                total_logit.append(i + k)
            metric_sets, at_list = metrics.get_offline_metrics(wnd_conf[SCHEMA][HEADER_SCHEMA],
                                                               header_list, total_logit)
            for action, metric in metric_sets.items():
                metric_pre, metric_mrr = metric
                offline_metrics_str = ''

                metric_threshlod_pair = zip(at_list, metric_pre)
                for tuple0, tuple1 in metric_threshlod_pair:
                    offline_metrics_str += "action_{a}_pre_at_{n}: {m}\n".format(a=action, n=tuple0, m=tuple1)

                offline_metrics_str += "\n"

                metric_threshlod_pair = zip(at_list, metric_mrr)
                for tuple0, tuple1 in metric_threshlod_pair:
                    offline_metrics_str += "action_{a}_mrr_at_{n}: {m}\n".format(a=action, n=tuple0, m=tuple1)

                offline_metrics_str += "\n"

                log_to_file(offline_metrics_str, out_file_test)
                print(offline_metrics_str)

            metric_sets = metrics.get_offline_metrics_auc(wnd_conf[SCHEMA][HEADER_SCHEMA], header_list, total_logit)
            for action, metric in metric_sets.items():
                offline_metrics_str = ''
                offline_metrics_str += "action_{a}_auc: {m}\n".format(a=action, m=metric[0])
                log_to_file(offline_metrics_str, out_file_test)
                print(offline_metrics_str)

            print("================== process file==================")

            version = wnd_conf.tag + '_' + test_tag
            checkpoint = 'ckpt-' + ckpt_name.split('-')[-1]

            # metrics3.save_to_local(wnd_conf[SCHEMA][HEADER_SCHEMA], header_list, test_click_logits_sigmoid_list,
            #                          test_order_logits_sigmoid_list, out_file_test,
            #                          version, checkpoint)
            # if "mmoe" in wnd_conf[MODEL][MODEL_TYPE]:
            #     metrics3.save_weights_to_local(click_weight_value_list,
            #                                      order_weight_value_list)
            print("====================DONE========")
            df = pd.read_csv("./res/{0}_test_{1}.csv".format(version, checkpoint))
            # out_name = "./res/{0}_result_{1}.txt".format(version, checkpoint)
            out_name = out_file_test
            # metrics2.get_offline_metrics(df, out_name)
            print("finish", out_name)
            sys.exit(0)

        tf.reset_default_graph()


if __name__ == '__main__':
    # args is a dict
    args = parse.argument_parse()
    wnd_conf = conf.Conf(
        conf_path=args['conf_path'],
        conf_file=args['conf_file'])

    if args['is_test'] == 'false':
        job_name = eval(os.getenv('TF_CONFIG'))['task']['type']
        if job_name == "chief":
            print(">> Go to train function.")
            train(wnd_conf, args['model_ckpt'])
        elif job_name == "evaluator":
            print(">> Go to validation function.")
            validation(wnd_conf)

    else:
        print(">> Go to test function.")
        predict(wnd_conf, args['model_ckpt'], args['test_tag'], args['test_score_method'])
