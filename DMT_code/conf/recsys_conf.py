# -*- coding:utf-8 -*-
import os
import sys

if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path + '/../util')
from util import *

SEPERATOR = '\n' + '>' * 60


class Conf:
    def __init__(self, conf_path='./', conf_file='dnn.model.conf'):

        # self.conf_parser
        self.conf_parser = configparser.ConfigParser()
        self.conf_parser.read(conf_path + conf_file)

        # self.tag
        self.tag = self.get_tag(conf_file)
        self.label_cnt_lst = []
        self.yes_label_cnt_lst = []
        # self.conf_sections is a map, self.conf_sections[section] is also a map
        self.conf_sections = {}
        for section in self.conf_parser.sections():
            self.conf_sections[section] = self.get_section_conf(section)

        self.reset(PARAMETER, LABEL_WEIGHT, csv_to_int_list, None)
        self.reset(PARAMETER, LOSS_WEIGHT, csv_to_float_list, None)
        self.reset(EXPORT_MODEL, EXPORT_WEIGHT, csv_to_float_list, None)

        self.reset(MODEL, FEAT_DIM, int, None)
        self.reset(MODEL, OUTPUT_UNITS, int, None)
        self.reset(MODEL, HIDDEN_UNITS, csv_to_int_list, None)
        self.reset(MODEL, HIDDEN_UNITS_BIAS, csv_to_int_list, None)
        self.reset(MODEL, hidden_units_bottom, csv_to_int_list, None)
        self.reset(MODEL, hidden_units_task, csv_to_int_list, None)
        self.reset(MODEL, num_experts, int, None)

        self.reset(MODEL, IS_USE_FEATURE, str_to_bool, True)
        self.reset(MODEL, DROPOUT, csv_to_float_list, None)
        self.reset(MODEL, DROPOUT_BOTTOM, csv_to_float_list, None)
        self.reset(MODEL, DROPOUT_TASK, csv_to_float_list, None)
        self.reset(MODEL, EPOCH_NUM, int, None)
        self.reset(MODEL, BATCH_SIZE, int, None)
        self.reset(MODEL, SHUFFLE_SIZE, int, 100000)
        self.reset(MODEL, TEST_BATCH_SIZE, int, None)
        self.reset(MODEL, VALIDATION_BATCH_SIZE, int, None)
        self.reset(MODEL, VALIDATE_STEP, int, None)
        self.reset(MODEL, IS_BN, str_to_bool, None)
        self.reset(MODEL, BN_DECAY, float, 0.999)
        self.reset(MODEL, IS_DROPOUT, str_to_bool, None)
        self.reset(MODEL, FILTER_SHAPE, csv_to_int_list, None)
        self.reset(MODEL, MAX_ITER_STEP, int, None)
        self.reset(MODEL, TOTAL_EXAMPLE_NUM, int, None)
        self.reset(MODEL, LOSS_CTR_REL_METHOD, str, None)

        self.reset(MODEL, propensity_em, str_to_bool, False)
        self.reset(MODEL, propensity_em_type, str, None)
        print("propensity_em:", self[MODEL][propensity_em])

        #unbias
        self.reset(MODEL, propensity_em, str_to_bool, False)
        self.reset(MODEL, propensity_em_type, str, None)
        print("propensity_em:", self[MODEL][propensity_em])

        self.reset(EMBEDDING, attention_embed_seq_ts, str, '')

        self.labels = self.get_labels(self[CLASS_WEIGHT][TRAIN_WEIGHT])
        self.reset(CLASS_WEIGHT, TRAIN_WEIGHT, parse_weight, None)
        self.reset(CLASS_WEIGHT, VALID_WEIGHT, parse_weight, None)
        self.reset(CLASS_WEIGHT, WEIGHT_CTR, parse_weight, None)
        self.reset(CLASS_WEIGHT, WEIGHT_ECVR, parse_weight, None)
        self.reset(MODEL, WND_WD, float, None)
        self.reset(MODEL, L2_EMB_LAMBDA, float, None)
        self.reset(MODEL, ENABLE_SSP, str_to_bool, True)

        self.reset(ONLINE, MIN_TRAIN_EXA_NUMS, int, 400000)
        self.reset(ONLINE, MAX_RATIO, float, 1.2)
        self.reset(ONLINE, MIN_RATIO, float, 0.8)
        self.reset(ONLINE, TO_ADDRS, str, '')
        self.reset(ONLINE, SUBJECT, str, '模型更新实验')

        self.reset(EMBEDDING, attention_embed_seq_ts, str, '')

        self[ONLINE][TO_ADDRS] = [addr.strip() for addr in self[ONLINE][TO_ADDRS].split(',')]

        self[MODEL][STEP_BOUNDARY] = [int(step) for step in self[MODEL][STEP_BOUNDARY].split(",")]
        self[MODEL][LEARNING_RATE] = [float(rate) for rate in self[MODEL][LEARNING_RATE].split(",")]

        # generate the summary path and model path
        #self[PATH][SUMMARY_PATH] = self[PATH][OUTPUT_PATH] + self.tag + '.summary'
        default_summary_path = "/export/App/training_platform/PinoModel/models"
        if "summary_path" in self.conf_parser.options("path"):
            if self[PATH][SUMMARY_PATH] != "" and self[PATH][SUMMARY_PATH] != default_summary_path:
                self[PATH][SUMMARY_PATH] = self[PATH][SUMMARY_PATH]
            else:
                self[PATH][SUMMARY_PATH] = default_summary_path
        else:
            self[PATH][SUMMARY_PATH] = default_summary_path
        self[PATH][MODEL_PATH] = self[PATH][OUTPUT_PATH] + self.tag + '.model/'
        self[PATH][MODEL_FROZEN_PATH] = self[PATH][MODEL_PATH] + 'frozen/'
        self[PATH][MODEL_IMP_PATH] = self[PATH][MODEL_PATH] + 'imp/'
        self[PATH][VALIDATION_RESULT] = self[PATH][OUTPUT_PATH] + self.tag + '.validation.result'
        self[PATH][TRAIN_RESULT] = self[PATH][OUTPUT_PATH] + self.tag + '.train.result'

        if not os.path.exists(self[PATH][MODEL_PATH]):
            os.makedirs(self[PATH][MODEL_PATH])

        if self[PATH][TRAIN_DATA_PATH][-1] != '/':
            self[PATH][TRAIN_DATA_PATH] += '/'

        if self[PATH][TEST_DATA_PATH][-1] != '/':
            self[PATH][TEST_DATA_PATH] += '/'

        if self[PATH][VALIDATION_DATA_PATH][-1] != '/':
            self[PATH][VALIDATION_DATA_PATH] += '/'

        self.embedding_list = self.get_emb(self[EMBEDDING][EMB])
        self.embedding_list_bias = self.get_emb(self[EMBEDDING][EMB_BIAS])
        self.attention_embed_pairs = self.get_attention_embed_v2(self[EMBEDDING][ATTENTION_EMBED])
        self.attention_embed_seq_ts = self.get_attention_embed_ts(self[EMBEDDING][attention_embed_seq_ts])
        print("attention_embed_seq_ts:", self.attention_embed_seq_ts)
        self.sim_embed = self.get_attention_embed(self[EMBEDDING][SIM_EMBED])
        self.embedding_init_info = self.get_emb_init_info(self[EMBEDDING][UPDATE_EMB])

        self.weight_ctr = self[CLASS_WEIGHT][WEIGHT_CTR]
        self.weight_ecvr = self[CLASS_WEIGHT][WEIGHT_ECVR]
        print("weight_ctr", self.weight_ctr)
        print("weight_ecvr:", self.weight_ecvr)


        self[SCHEMA][HEADER_SCHEMA] = [schema.strip() for schema in self[SCHEMA][HEADER_SCHEMA].split(',')]
        if (self[PATH][TRAIN_DATA_STAT_PATH] != None and len(self[PATH][TRAIN_DATA_STAT_PATH]) > 0):
            self.get_label_cnt_lst(self[PATH][TRAIN_DATA_STAT_PATH])
            self[MODEL][TOTAL_EXAMPLE_NUM] = sum(self.label_cnt_lst)
            self.label_cnt_lst = [x / self.label_cnt_lst[-1] for x in self.label_cnt_lst]

            total_step = int((self[MODEL][EPOCH_NUM] * self[MODEL][TOTAL_EXAMPLE_NUM]) / (
                    self[MODEL][BATCH_SIZE] * len(self[MODEL][GPU_VISIBLE].split(','))))

            self[MODEL][SAVE_CKPT_NUMS] = int(self[MODEL][MAX_ITER_STEP] / self[MODEL][VALIDATE_STEP])
            if self[MODEL][MAX_ITER_STEP] > total_step:
                self[MODEL][MAX_ITER_STEP] = total_step
                self[MODEL][SAVE_CKPT_NUMS] = int(
                    self[MODEL][MAX_ITER_STEP] * len(self[MODEL][GPU_VISIBLE].split(',')) / self[MODEL][VALIDATE_STEP])


        self.model_type=self[MODEL][MODEL_TYPE]

        #base
        print("embedding_list:", self.embedding_list)
        self.zero_pad = self[MODEL][zero_pad]
        print("zero_pad:", self.zero_pad)


        #multi_task
        print("hidden_units_bottom:", self[MODEL][hidden_units_bottom])
        print("hidden_units_task:", self[MODEL][hidden_units_task])
        print("num_experts:", self[MODEL][num_experts])

        #unbias
        self.is_unbias_model = "unbias" in self.model_type
        print("is_unbias_model:", self.is_unbias_model)
        self.propensity_em = self[MODEL][propensity_em]
        self.propensity_em_type = self[MODEL][propensity_em_type]
        print("propensity_em:", self.propensity_em)
        print("propensity_em_type:", self.propensity_em_type)
        print("embedding_list_bias:", self.embedding_list_bias)

        #if (self.model_type == "embed_mlp_unbias"):
        if ("unbias" in self.model_type):
            self.loss_unbias_method = self[MODEL][loss_unbias_method]
            print("loss_unbias_method:", self.loss_unbias_method)
            self.reset(MODEL, dropout_rate_bias, csv_to_float_list, None)
            self.dropout_rate_bias = self[MODEL][dropout_rate_bias]
            print("dropout_rate_bias:", self.dropout_rate_bias)
            self.loss_ctr_rel_method = self[MODEL][LOSS_CTR_REL_METHOD]
            print("loss_ctr_rel_method:", self.loss_ctr_rel_method)

        #is_use_feature
        self.is_use_feature = self[MODEL][IS_USE_FEATURE]
        print("is_use_feature:", self.is_use_feature)


        #transformer
        if ("transformer" in self.model_type):
            self.reset(MODEL, transformer_d_model, int, None)
            self.reset(MODEL, transformer_d_ff, int, None)
            self.reset(MODEL, transformer_num_heads, int, None)
            self.reset(MODEL, transformer_num_blocks_encode, int, None)
            self.reset(MODEL, transformer_num_blocks_decode, int, None)
            self.reset(MODEL, transformer_maxlen_k, int, None)
            self.reset(MODEL, transformer_maxlen_q, int, None)
            self.reset(MODEL, transformer_dropout_rate, float, None)
            self.reset(MODEL, transformer_is_trans_input_by_mlp, str_to_bool, False)
            self.reset(MODEL, transformer_position_encoding_method, str, 'position_sin_cos')
            self.reset(MODEL, transformer_is_trans_out_concat_item, str_to_bool, True)
            self.reset(MODEL, transformer_is_trans_out_by_mlp, str_to_bool, False)
            self.reset(MODEL, transformer_is_decoder_add_pos_emb, str_to_bool, False)

            self.d_model = self[MODEL][transformer_d_model]
            self.d_ff = self[MODEL][transformer_d_ff]
            self.num_heads = self[MODEL][transformer_num_heads]
            self.num_blocks_encode = self[MODEL][transformer_num_blocks_encode]
            self.num_blocks_decode = self[MODEL][transformer_num_blocks_decode]
            self.maxlen_k = self[MODEL][transformer_maxlen_k]
            self.maxlen_q = self[MODEL][transformer_maxlen_q]
            self.dropout_rate = self[MODEL][transformer_dropout_rate]

            self.is_trans_input_by_mlp = self[MODEL][transformer_is_trans_input_by_mlp]
            self.position_encoding_method = self[MODEL][transformer_position_encoding_method]
            self.is_use_seq_ts = len(self[EMBEDDING][attention_embed_seq_ts]) >= 1
            self.is_trans_out_concat_item = self[MODEL][transformer_is_trans_out_concat_item]
            self.is_trans_out_by_mlp = self[MODEL][transformer_is_trans_out_by_mlp]
            self.is_decoder_add_pos_emb = self[MODEL][transformer_is_decoder_add_pos_emb]

            print("is_trans_input_by_mlp", self.is_trans_input_by_mlp)
            print("is_use_seq_ts:", self.is_use_seq_ts)
            print("position_encoding_method:", self.position_encoding_method)
            print("is_trans_out_concat_item:", self.is_trans_out_concat_item)
            print("is_trans_out_by_mlp:", self.is_trans_out_by_mlp)
            print("is_decoder_add_pos_emb:", self.is_decoder_add_pos_emb)


    # why introduce reset function.
    # because arguments parsed from "*.conf" file are all type of string,
    # we should convert them to the right types, e.g. int, float
    def reset(self, section, option, f, default):
        try:
            self.conf_sections[section][option] = f(self.conf_sections[section][option])
        except:
            if section not in self.conf_sections:
                self.conf_sections[section] = {}
            self.conf_sections[section][option] = default
        finally:
            pass

    def __getitem__(self, k):
        return self.conf_sections[k]

    def get_conf(self, section, option=None):
        try:
            if (option == None):
                return self.conf_sections[section]
            else:
                return self.conf_sections[section][option]
        except:
            return None

    # Function: put all the k-v (key-value) pairs into a map
    def get_section_conf(self, section):
        param = {}
        for (k, v) in self.conf_parser.items(section):
            param[k] = v
        return param

    # if input: dnn.254.embed.mask.v10.conf, then output: dnn.254.embed.mask.v10
    # this function remove the last 'conf' string
    def get_tag(self, conf_file):
        splitted = conf_file.split('.')
        if (splitted[-1] == 'conf'):
            return '.'.join(splitted[0:-1])
        else:
            return conf_file

    # pattern :  embedding-name:embedding-ids-size:embedding-dim:embedding-feature-name#....
    # for example: Cid3:Cid3_size:Cid3_emb_dim:iCid3
    def get_emb(self, embs):
        if len(embs) <= 2:
            return []
        emb_list = []
        splitted = embs.split('#')
        for emb_info in splitted:
            emb_fields_list = emb_info.split(':')
            emb_fields_list[1] = int(emb_fields_list[1])
            emb_fields_list[2] = int(emb_fields_list[2])
            emb_list.append(emb_fields_list)
        return emb_list

    def get_attention_embed(self, attention):
        if len(attention) <= 2:
            return []
        attention_embeds = []
        for attention_pair_str in attention.split('#'):
            attention_pair = attention_pair_str.split(':')
            attention_embeds.append((attention_pair[0], attention_pair[1]))
        return attention_embeds

    def get_attention_embed_v2(self, attention):
        if len(attention) <= 2:
            return []
        attention_embeds_all = []
        for attention_pair_str_all in attention.split('|'):
            attention_embeds = []
            for attention_pair_str in attention_pair_str_all.split('#'):
                attention_pair = attention_pair_str.split(':')
                attention_embeds.append((attention_pair[0], attention_pair[1]))
            attention_embeds_all.append(attention_embeds)
        return attention_embeds_all

    def get_attention_embed_ts(self, attention):
        if len(attention) <= 1:
            return []
        attention_embeds_all = []
        for attention_pair_str_all in attention.split('|'):
            attention_pair_str_all = attention_pair_str_all.strip()
            attention_embeds_all.append(attention_pair_str_all)
        return attention_embeds_all

    def get_idschema(self):
        id_schemas = []
        for emb in self.embedding_list:
            id_schemas.append(emb[3])
        return id_schemas

    def get_idschema_bias(self):
        print("get_idschema_bias...")
        id_schemas = []
        for emb in self.embedding_list_bias:
            id_schemas.append(emb[3])
        #print("id_schemas:", id_schemas)
        return id_schemas

    def get_emb_init_info(self, emb_infos):
        emb_info_list = {}
        splitted = emb_infos.split('#')
        for emb_info in splitted:
            fields = emb_info.split(':')
            if len(fields) != 2:
                continue
            emb_info_list[fields[0]] = fields[1]
        return emb_info_list

    def get_label_cnt_lst(self, file_path):
        stat_file = "today_label.stat"
        if hdfsFileExist(file_path, '_SUCCESS'):
            hdfsToLocal(file_path, stat_file)
        else:
            stat_file = file_path
        for line in open(stat_file, "r").readlines():
            self.label_cnt_lst.append(int(line.strip()))

    def get_labels(self, label_weight_str):
        weights_list = [weight.strip() for weight in label_weight_str.split(',')]
        labels = []
        for labei2weight_str in weights_list:
            label_weight_list = labei2weight_str.split(':')
            labels.append(int(label_weight_list[0]))
        labels = sorted(labels)
        return labels

    def get_yes_label_cnt(self):
        if not os.path.exists('yes_label.stat'):
            os.system('mv today_label.stat yes_label.stat')
        with open("yes_label.stat") as fp:
            for line in fp:
                self.yes_label_cnt_lst.append(int(line.strip()))
        print(self.yes_label_cnt_lst)
        self.yes_label_cnt_lst = [x / self.yes_label_cnt_lst[-1] for x in self.yes_label_cnt_lst]


if __name__ == '__main__':
    conf_path = './settings/'
    conf_file = 'dnn.model.conf'
    wnd_conf = Conf(conf_path, conf_file)
    print(wnd_conf.conf_sections)
    print(wnd_conf.conf_sections[ONLINE][SUBJECT])
