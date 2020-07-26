import tensorflow as tf
import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path + '/../util')
sys.path.append(file_path + '/../conf')
import recsys_conf
sys.path.append(file_path + '/../parse')
import parse
sys.path.append(file_path + '/../util')
from util import *
from util_unbias import propensity_em_position, propensity_em_page

feature_dim = 0
weight_num = 0
gpu_num = 4
id_features = ['uSkus', 'iSku']
id_features_bias = []

propensity_em_type = 'position'

def parse_single_line(example_proto):
    fields = {
        "label": tf.FixedLenFeature([], tf.float32),
        "mask": tf.FixedLenFeature([weight_num], tf.float32),
        "features": tf.FixedLenFeature([feature_dim], tf.float32),
        "header": tf.FixedLenFeature([], tf.string)
    }

    for id_feature in id_features:
        id_feature_wts = id_feature + 'Wts'
        fields[id_feature_wts] = tf.VarLenFeature(tf.float32)
        fields[id_feature] = tf.VarLenFeature(tf.string)

    print("id_features_bias:", id_features_bias)
    for id_feature in id_features_bias:
        id_feature_wts = id_feature + 'Wts'
        fields[id_feature_wts] = tf.VarLenFeature(tf.float32)
        fields[id_feature] = tf.VarLenFeature(tf.string)

    # fields['utermsCnt'] = tf.VarLenFeature(tf.int64)

    parsed_features = tf.parse_single_example(example_proto, fields)

    label = parsed_features["label"]
    mask = parsed_features["mask"]
    features = parsed_features["features"]
    header = parsed_features["header"]

    feature_set = {'features': features}
    for id_feature in id_features:
        id_feature_wts = id_feature + 'Wts'
        feature_set[id_feature] = parsed_features[id_feature]
        feature_set[id_feature_wts] = parsed_features[id_feature_wts]
    # feature_set['utermsCnt'] = parsed_features['utermsCnt']

    for id_feature in id_features_bias:
        id_feature_wts = id_feature + 'Wts'
        feature_set[id_feature] = parsed_features[id_feature]
        feature_set[id_feature_wts] = parsed_features[id_feature_wts]

    #position
    feature_set['em_position'] = tf.strings.to_number(tf.string_split([header], '\t').values[4], tf.int32)
    feature_set['em_position'] = tf.math.minimum(feature_set['em_position'], tf.constant(400, tf.int32))
    feature_set['em_page'] = tf.strings.to_number(tf.string_split([header], '\t').values[11], tf.int32)
    feature_set['em_page'] = tf.math.minimum(feature_set['em_page'], tf.constant(100, tf.int32))


    if(propensity_em_type == "position"):
        propensity_look_up_table = tf.constant(propensity_em_position, dtype=tf.float32)
        feature_set['propensity'] = tf.gather(propensity_look_up_table, feature_set['em_position'])
    if (propensity_em_type == "page"):
        propensity_look_up_table = tf.constant(propensity_em_page, dtype = tf.float32)
        feature_set['propensity'] = tf.gather(propensity_look_up_table, feature_set['em_page'])


    feature_set['propensity_weight'] = tf.clip_by_value(tf.divide(1, feature_set['propensity']), tf.constant(1.0), tf.constant(10.0))
    feature_set['propensity_weight_positive'] = tf.where(tf.greater(label, tf.constant(0.5)), feature_set['propensity_weight'], tf.constant(1.0))
    #feature_set['propensity_weight_mul'] = feature_set['propensity_weight_positive']
    feature_set['propensity_weight_mul'] = feature_set['propensity_weight']


    return label, header, mask, feature_set


def get_batch(wnd_conf, lookup_tables=None):
    num_parallel_readers = 100
    num_parallel_batches = 1
    global feature_dim
    feature_dim = wnd_conf[MODEL][FEAT_DIM]
    global weight_num
    weight_num = len(wnd_conf[CLASS_WEIGHT][TRAIN_WEIGHT])
    global id_features
    id_features = wnd_conf.get_idschema()
    global id_features_bias
    id_features_bias = wnd_conf.get_idschema_bias()
    global propensity_em_type
    propensity_em_type = wnd_conf[MODEL][propensity_em_type]

    files = tf.data.Dataset.list_files(wnd_conf[PATH][TRAIN_DATA_PATH] + '*')

    dataset = files.apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=num_parallel_readers,
                                                 sloppy=True))
    # repeat() before shuffle provides better performance
    dataset = dataset.repeat(wnd_conf[MODEL][EPOCH_NUM])
    dataset = dataset.shuffle(wnd_conf[MODEL][SHUFFLE_SIZE])
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=parse_single_line, batch_size=wnd_conf[MODEL][BATCH_SIZE],
                                           num_parallel_batches=num_parallel_batches))
    dataset = dataset.prefetch(gpu_num)
    iterator = dataset.make_one_shot_iterator()
    batch_labels, _, batch_mask, batch_features = iterator.get_next()
    if lookup_tables != None:
        lookup_tables.transform_id2index(batch_features)
    return batch_labels, batch_mask, batch_features


def get_multi_towers_batch(wnd_conf, lookup_tables=None):
    available_cpu_num = get_available_cpu_num()
    num_parallel_readers = int(available_cpu_num / 2) + 1
    num_parallel_calls = int(available_cpu_num / 2) + 1
    global feature_dim
    feature_dim = wnd_conf[MODEL][FEAT_DIM]
    global weight_num
    weight_num = len(wnd_conf[CLASS_WEIGHT][TRAIN_WEIGHT])
    global id_features
    id_features = wnd_conf.get_idschema()
    global id_features_bias
    id_features_bias = wnd_conf.get_idschema_bias()
    global propensity_em_type
    propensity_em_type = wnd_conf.propensity_em_type

    files = tf.data.Dataset.list_files(wnd_conf[PATH][TRAIN_DATA_PATH] + '*', seed=131)

    # dataset = files.apply(
    #     tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, \
    #                                              cycle_length=num_parallel_readers, sloppy=True, buffer_output_elements = 100, prefetch_input_elements = 0))
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, \
                                                 cycle_length=num_parallel_readers, sloppy=True))
    # repeat() before shuffle provides better performance
    dataset = dataset.repeat(wnd_conf[MODEL][EPOCH_NUM])
    dataset = dataset.shuffle(wnd_conf[MODEL][SHUFFLE_SIZE])
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=parse_single_line, \
                                                               batch_size=wnd_conf[MODEL][BATCH_SIZE],
                                                               num_parallel_calls=num_parallel_calls))
    dataset = dataset.prefetch(gpu_num)
    iterator = dataset.make_one_shot_iterator()

    per_tower_data = {}
    for gpu in wnd_conf[MODEL][GPU_VISIBLE].split(','):
        batch_labels, _, batch_mask, batch_features = iterator.get_next()
        if lookup_tables != None:
            lookup_tables.transform_id2index(batch_features)
        per_tower_data[gpu] = [batch_labels, batch_mask, batch_features]
    return per_tower_data


def get_dist_batch(wnd_conf, num_workers, worker_index, lookup_tables=None):
    num_parallel_readers = 5
    shuffle_buffer_size = 100000
    num_parallel_calls = 5
    global feature_dim
    feature_dim = wnd_conf[MODEL][FEAT_DIM]
    global weight_num
    weight_num = len(wnd_conf[CLASS_WEIGHT][TRAIN_WEIGHT])
    global id_features
    id_features = wnd_conf.get_idschema()
    global id_features_bias
    id_features_bias = wnd_conf.get_idschema_bias()
    global propensity_em_type
    propensity_em_type = wnd_conf.propensity_em_type

    dataset = tf.data.Dataset.list_files(wnd_conf[PATH][TRAIN_DATA_PATH] + '*')
    dataset = dataset.shard(num_workers, worker_index)
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=num_parallel_readers,
                                                 sloppy=True))
    # repeat() before shuffle provides better performance
    dataset = dataset.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=shuffle_buffer_size, count=wnd_conf[MODEL][EPOCH_NUM]))
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=parse_single_line, batch_size=wnd_conf[MODEL][BATCH_SIZE],
                                           num_parallel_calls=num_parallel_calls))
    dataset = dataset.prefetch(None)
    iterator = dataset.make_one_shot_iterator()
    batch_labels, _, batch_mask, batch_features = iterator.get_next()
    if lookup_tables != None:
        lookup_tables.transform_id2index(batch_features)
    return batch_labels, batch_mask, batch_features


def get_val_test_batch(file_path, EPOCH_NUM=-1, batch_size=2000, wnd_conf=None, lookup_tables=None):
    available_cpu_num = get_available_cpu_num()
    num_parallel_readers = int(available_cpu_num / 2 ) + 1
    num_parallel_calls = int(available_cpu_num / 2) + 1
    global feature_dim
    feature_dim = wnd_conf[MODEL][FEAT_DIM]
    global weight_num
    weight_num = len(wnd_conf[CLASS_WEIGHT][VALID_WEIGHT])
    global id_features
    id_features = wnd_conf.get_idschema()
    global id_features_bias
    id_features_bias = wnd_conf.get_idschema_bias()
    global propensity_em_type
    propensity_em_type = wnd_conf.propensity_em_type

    files = tf.data.Dataset.list_files(file_path + '*')

    # dataset = files.apply(
    #     tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, \
    #                                              cycle_length=num_parallel_readers, sloppy=True, buffer_output_elements = 100, prefetch_input_elements = 0))
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset,
                                                 cycle_length=num_parallel_readers, sloppy=True))

    dataset = dataset.repeat(EPOCH_NUM)
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=parse_single_line,
                                           batch_size=batch_size, num_parallel_calls=num_parallel_calls))
    dataset = dataset.prefetch(gpu_num)
    iterator = dataset.make_one_shot_iterator()
    batch_labels, batch_headers, batch_mask, batch_features = iterator.get_next()
    if lookup_tables != None:
        lookup_tables.transform_id2index(batch_features)
    return batch_labels, batch_headers, batch_mask, batch_features


if __name__ == '__main__':
    args = parse.argument_parse()
    wnd_conf = recsys_conf.Conf(
        conf_path=args['conf_path'],
        conf_file=args['conf_file'])
    # tables = lookup.LookupTables(wnd_conf)

    #print("conf:", args['conf_file'])
    test_path = wnd_conf[PATH][VALIDATION_DATA_PATH]
    batch_size = 1

    ## read train data
    batch_labels, batch_headers, batch_masks, batch_features = \
        get_val_test_batch(test_path, \
                           batch_size=batch_size, wnd_conf=wnd_conf)

    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        try:
            while True:
                batch_labels_v, batch_headers_v, batch_masks_v, batch_masks_v_mul, batch_features_v = \
                    sess.run(
                        [batch_labels, batch_headers, batch_masks, batch_masks * wnd_conf[CLASS_WEIGHT][TRAIN_WEIGHT],
                         batch_features])

                print("batch_labels_v:\n", batch_labels_v)
                print("batch_headers_v:\n", batch_headers_v)
                print("batch_masks_v:\n", batch_masks_v)
                print("batch_masks_v_mul:\n", batch_masks_v_mul)
                print("batch_features_v:\n", batch_features_v)

                print("batch_features_v['position']:\n", batch_features_v['position'])
                print("batch_features_v['page']:\n", batch_features_v['page'])
                print("batch_features_v['propensity']:\n", batch_features_v['propensity'])
                print("batch_features_v['propensity_weight']:\n", batch_features_v['propensity_weight'])
                print("batch_features_v['propensity_weight_positive']:\n", batch_features_v['propensity_weight_positive'])
                print("batch_features_v['propensity_weight_mul']:\n", batch_features_v['propensity_weight_mul'])


                break
        except tf.errors.OutOfRangeError:
            print("-----------------------------OutOfRangeError---------------------------------")
