import os
import time
import subprocess

INFO = 'info'
TYPE = 'type'

PARAMETER='parameter'
LABEL_WEIGHT='label_weight'
LOSS_WEIGHT='loss_weight'
LOSS_WEIGHT_METHOD = 'loss_weight_method'

EXPORT_MODEL='export_model'
EXPORT_WEIGHT='export_weight'

MODEL = 'model'
FEAT_DIM = 'feature_dimension'
OUTPUT_UNITS = 'output_units'
HIDDEN_UNITS = 'hidden_units'
IS_USE_FEATURE = 'is_use_feature'
HIDDEN_UNITS_BIAS = 'hidden_units_bias'
loss_unbias_method = 'loss_unbias_method'
LOSS_CTR_REL_METHOD = 'loss_ctr_rel_method'
propensity_em = 'propensity_em'
propensity_em_type = 'propensity_em_type'
MODEL_TYPE = 'model_type'
ENABLE_SSP = 'enable_ssp'
LEARNING_RATE = 'learning_rate'
STEP_BOUNDARY = 'step_boundary'
OPTIMIZER = 'optimizer'
hidden_units_bottom = 'hidden_units_bottom'
hidden_units_task = 'hidden_units_task'
num_experts = 'num_experts'
DROPOUT = 'dropout'
dropout_rate_bias = 'dropout_rate_bias'
DROPOUT_BOTTOM = 'dropout_bottom'
DROPOUT_TASK = 'dropout_task'
EPOCH_NUM = 'epoch_num'
BATCH_SIZE = 'batch_size'
SHUFFLE_SIZE = 'shuffle_size'
TEST_BATCH_SIZE = 'test_batch_size'
VALIDATION_BATCH_SIZE = 'validation_batch_size'
DEVICE = 'device'
GPU_VISIBLE = 'gpu_visible'
VALIDATE_STEP = 'validate_step'
IS_BN = 'is_bn'
BN_DECAY = 'bn_decay'
IS_DROPOUT = 'is_dropout'
FILTER_SHAPE = 'filter_shape'
MAX_ITER_STEP = 'max_iter_step'
TOTAL_EXAMPLE_NUM = 'total_example_num'
SAVE_CKPT_NUMS = 'save_ckpt_nums'
WND_WD = 'wnd_wd'
L2_EMB_LAMBDA = 'l2_emb_lambda'
zero_pad = 'zero_pad'


transformer_d_model = 'transformer_d_model'
transformer_d_ff = 'transformer_d_ff'
transformer_num_heads = 'transformer_num_heads'
transformer_num_blocks_encode = 'transformer_num_blocks_encode'
transformer_num_blocks_decode = 'transformer_num_blocks_decode'
transformer_maxlen_k ='transformer_maxlen_k'
transformer_maxlen_q = 'transformer_maxlen_q'
transformer_dropout_rate = 'transformer_dropout_rate'
transformer_is_trans_input_by_mlp = 'transformer_is_trans_input_by_mlp'
transformer_position_encoding_method = 'transformer_position_encoding_method'
transformer_is_trans_out_concat_item = 'transformer_is_trans_out_concat_item'
transformer_is_trans_out_by_mlp = 'transformer_is_trans_out_by_mlp'
transformer_is_decoder_add_pos_emb = 'transformer_is_decoder_add_pos_emb'


PATH = 'path'
TRAIN_DATA_PATH = 'train_data_path'
TEST_DATA_PATH = 'test_data_path'
TEST_DATA_PATH_ORD = 'test_data_path_ord'
TRAIN_DATA_MEAN_PATH = 'train_data_mean_path'
TRAIN_DATA_STD_PATH = 'train_data_std_path'
TRAIN_DATA_STAT_PATH = 'train_data_stat_path'
TRAIN_RESULT = 'train_result'
TEST_RESULT = 'test_result'
VALIDATION_DATA_PATH = 'validation_data_path'
VALIDATION_RESULT = 'validation_result'
OUTPUT_PATH = 'output_path'
SUMMARY_PATH = 'summary_path'
MODEL_PATH = 'model_path'
MODEL_FROZEN_PATH = 'model_frozen_path'
MODEL_IMP_PATH = 'model_imp_path'

CLASS_WEIGHT = 'class_weight'
TRAIN_WEIGHT = 'train_weight'
VALID_WEIGHT = 'valid_weight'

WEIGHT_CTR = 'weight_ctr'
WEIGHT_ECVR = 'weight_ecvr'

SCHEMA = 'schema'
HEADER_SCHEMA = 'header_schema'

EMBEDDING = 'embedding'
EMB = 'emb'
EMB_BIAS = 'emb_bias'
ATTENTION_EMBED = 'attention_embed'
attention_embed_seq_ts = 'attention_embed_seq_ts'
SIM_EMBED = 'sim_embed'
UPDATE_EMB = 'update_emb'

ONLINE = 'online_learning'
MIN_TRAIN_EXA_NUMS = 'min_train_exa_num'
MAX_RATIO = 'data_max_ratio'
MIN_RATIO = 'data_min_ratio'
TO_ADDRS = 'email_to_addrs'
SUBJECT = 'email_subject'


def str_to_bool(s):
    return s in ['True', 'true', 'yes', 'TRUE', '1']


def csv_to_f_list(s, f):
    return [f(a) for a in s.strip().split(',')]


def csv_to_int_list(s):
    return csv_to_f_list(s, int)


def csv_to_float_list(s):
    return csv_to_f_list(s, float)


def parse_weight(s):
    weights_list = s.split(',')
    print(weights_list)
    label_weight_dict = {}
    for labei2weight_str in weights_list:
        label_weight_list = labei2weight_str.split(':')
        label = int(label_weight_list[0])
        weight = float(label_weight_list[1])
        label_weight_dict[label] = weight
    labels = label_weight_dict.keys()
    labels = sorted(labels)

    return [label_weight_dict[label] for label in labels]


def get_const_data(file_path):
    if not os.path.exists(file_path):
        return get_const_data_from_hdfs(file_path)
    else:
        return get_const_data_from_local(file_path)


def get_const_data_from_local(file_path):
    file = open(file_path, 'r+')
    line = file.readline()
    temp = line.split('\t')
    const_data = [float(s.strip()) for s in temp]
    return const_data


def hdfsFileExist(path, file):
    path = '/'.join(path.split('/')[0:-1])
    cmd = 'hadoop fs -test -e ' + path + '/' + file
    is_exist = subprocess.call(cmd, shell=True)
    if is_exist == 0:
        return True
    else:
        # logging.error(path + " no file " + file)
        return False


def hdfsToLocal(hdfsPath, localfile):
    if os.path.exists(localfile):
        os.remove(localfile)
    cmd = 'hadoop fs -getmerge ' + hdfsPath + ' ' + localfile
    invoke_times = 0
    stats = 1
    while stats != 0 and invoke_times < 5:
        stats = subprocess.call(cmd, shell=True)
        invoke_times += 1
        time.sleep(5)
        # logging.info("getmerge file times:" + str(invoke_times))
    return stats


def hdfs_files_to_local(hdfs_path):
    local_path = os.path.join(os.getcwd(), hdfs_path.rstrip('/').split('/')[-1] + "/")
    if os.path.exists(local_path):
        del_path(local_path)
    cmd = 'hadoop fs -get ' + hdfs_path
    invoke_times = 0
    stats = 1
    while stats != 0 and invoke_times < 5:
        stats = subprocess.call(cmd, shell=True)
        invoke_times += 1
        time.sleep(5)
    if invoke_times == 5 and stats == 1:
        return hdfs_path
    suc_file = os.path.join(local_path, '_SUCCESS')
    if os.path.exists(suc_file):
        print("remove success file")
        os.remove(suc_file)
    return local_path


def get_const_data_from_hdfs(file_path):
    if hdfsFileExist(file_path, '_SUCCESS'):
        stat_file = "temp.stat"
        if not hdfsToLocal(file_path, stat_file):
            data = get_const_data_from_local("./" + stat_file)
            # subprocess.call("\\rm ./temp.stat",shell=True)
            return data


def del_path(path):
    if path.startswith("hdfs") or path.startswith("/user"):
        hdfs_rm(path)
    else:
        local_rm(path)


def hdfs_rm(path):
    cmd = 'hadoop fs -rm  -r ' + path
    subprocess.call(cmd, shell=True)


def local_rm(path):
    subprocess.call(["/bin/rm", "-rf", path])


def file_exists(path, filename):
    if path.startswith("hdfs") or path.startswith("/user"):
        if hdfsFileExist(path, filename):
            return True
        else:
            return False
    else:
        if os.path.exists(os.path.join(path, filename)):
            return True
        else:
            return False


def create_file(path, filename):
    if path.startswith("hdfs") or path.startswith("/user"):
        cmd = 'hadoop fs -touch ' + path + "/" + filename
        subprocess.call(cmd, shell=True)
    else:
        file = open(os.path.join(path, filename), "w")
        file.close()

def get_available_cpu_num():
    if 'TF_CONFIG' in os.environ and os.path.exists("tfconf.py"):
        import tfconf
        job_name = eval(os.getenv('TF_CONFIG'))['task']['type']
        if job_name == "chief":
            return int(tfconf.chief['cpu'])
        elif job_name == "evaluator":
            return int(tfconf.evaluator['cpu'])
    else:
        return int(os.popen('cat /proc/cpuinfo |grep "process"|wc -l').read())


if __name__ == '__main__':
    print(csv_to_int_list('1,2,3'))
