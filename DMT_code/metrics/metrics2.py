import pandas as pd
import multiprocessing as mp
import numpy as np
import os
import sys
import subprocess
from sklearn.metrics import roc_auc_score

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)


file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path + '../data_feed')
import gc
from ctypes import cdll, CDLL

cdll.LoadLibrary("libc.so.6")
libc = CDLL("libc.so.6")


CLICK = 2
ORDER = 5

#at_list = [2, 4, 10, 12, 20, 24, 40]
at_list = [2, 4, 6, 8, 10, 12, 14]

def log_to_file(info_str, file_name):
    if file_name.startswith("hdfs") or file_name.startswith("/user"):
        subprocess.call(["echo '%s' | hadoop fs -appendToFile - %s" % (
            info_str, file_name)], shell=True)
    else:
        subprocess.call(["echo '%s' >> %s" % (info_str, file_name)], shell=True)


# args is a tuple
# return a float
def get_pre_at_n(df, N, action):
    # descending by score, and if scores are same then ascending by label
    sorted_df = df.head(N)
    # length is N if df have at least N lines or
    # length is smaller than N if df have less than N lines
    (length, width) = sorted_df.shape
    if (length == 0):
        return 0
    check = sorted_df['label'] >= action
    # sys.stdout.write check

    # sys.stdout.write type(check)
    # <class 'pandas.core.series.Series'>
    return check.astype(int).sum() * 1.0 / length

def get_mrr_at_n(df, N, action):
    # descending by score, and if scores are same then ascending by label
    sorted_df = df.head(N)
    # length is N if df have at least N lines or
    # length is smaller than N if df have less than N lines
    (length, width) = sorted_df.shape
    if length == 0:
        return 0

    check = sorted_df['label'] >= action
    check = check.astype(int).to_list()

    res = 0.0
    #print("check:", check)
    for i in range(length):
        #print("check[i]:", check[i])
        if(check[i]):
            res = 1/float(i+1)
            break
    return res


def grid_search_metric(dfs, clk_output, ord_output, click_weight, order_weight):
    at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #sys.stdout.write('{} begin grid_search_metric\n'.format(os.getpid()))
    total_count = len(dfs)
    index = 0
    for df in dfs:
        index += 1
        #print("{}--{}_{}---{}/{} is completed".format(os.getpid(), click_weight, order_weight,
        #                                                 index, total_count))
        # df['score'] = click_weight * df['click_score'] + detail_weight * df['detail_score'] + order_weight * df[
        #     'order_score']
        final_df = df.sort_values(by=['score', 'label'], ascending=[False, True])
        for i in range(len(at_list)):
            at_n_list_clk[i] += get_pre_at_n(final_df, at_list[i], CLICK)
            at_n_list_ord[i] += get_pre_at_n(final_df, at_list[i], ORDER)

    clk_output.put(at_n_list_clk)
    ord_output.put(at_n_list_ord)
    #sys.stdout.write('{} end grid_search_metric\n'.format(os.getpid()))

def grid_search_metric_pre_mrr(dfs, clk_output_pre, ord_output_pre, clk_output_mrr, ord_output_mrr):
    pre_at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pre_at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mrr_at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mrr_at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    total_count = len(dfs)
    index = 0
    for df in dfs:
        index += 1
        # print("{}--{}_{}---{}/{} is completed".format(os.getpid(), click_weight, order_weight,
        #                                                 index, total_count))
        # df['score'] = click_weight * df['click_score'] + detail_weight * df['detail_score'] + order_weight * df[
        #     'order_score']
        sorted_df = df.sort_values(by=['score', 'label'], ascending=[False, True])
        for i in range(len(at_list)):
            pre_at_n_list_clk[i] += get_pre_at_n(sorted_df, at_list[i], CLICK)
            pre_at_n_list_ord[i] += get_pre_at_n(sorted_df, at_list[i], ORDER)
            mrr_at_n_list_clk[i] += get_mrr_at_n(sorted_df, at_list[i], CLICK)
            mrr_at_n_list_ord[i] += get_mrr_at_n(sorted_df, at_list[i], ORDER)
    clk_output_pre.put(pre_at_n_list_clk)
    ord_output_pre.put(pre_at_n_list_ord)
    clk_output_mrr.put(mrr_at_n_list_clk)
    ord_output_mrr.put(mrr_at_n_list_ord)

def handle_partition_metric(dfs, clk_output, ord_output):
    at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for df in dfs:
        sys.stdout.write(str(df.shape) + '\n')
        click_sorted_df = df.sort_values(by=['click_score', 'label'], ascending=[False, True])
        order_sorted_df = df.sort_values(by=['order_score', 'label'], ascending=[False, True])
        for i in range(len(at_list)):
            at_n_list_clk[i] += get_pre_at_n(click_sorted_df, at_list[i], CLICK)
            at_n_list_ord[i] += get_pre_at_n(order_sorted_df, at_list[i], ORDER)

    clk_output.put(at_n_list_clk)
    ord_output.put(at_n_list_ord)


def save_to_local(header_schema, headers, test_click_logits_sigmoid_values,
                  test_order_logits_sigmoid_values, file_name):
    if sys.version_info[0] < 3:
        df = pd.DataFrame([value.split('\t') for value in headers], columns=header_schema)
    else:
        df = pd.DataFrame([value.decode().strip().split('\t') for value in headers], columns=header_schema)
    df['label'] = df['label'].astype(int)

    df['click_score'] = pd.Series(test_click_logits_sigmoid_values)
    df['order_score'] = pd.Series(test_order_logits_sigmoid_values)
    df.to_csv("./output/test.csv", index=None)
    with open("./output/path.txt", 'w') as w:
        w.write(file_name)


def calculate_metrics(header_schema, headers, test_click_logits_sigmoid_values,
                      test_order_logits_sigmoid_values, out_file_test):
    if sys.version_info[0] < 3:
        df = pd.DataFrame([value.split('\t') for value in headers], columns=header_schema)
    else:
        df = pd.DataFrame([value.decode().strip().split('\t') for value in headers], columns=header_schema)
    df['label'] = df['label'].astype(int)

    df['click_score'] = pd.Series(test_click_logits_sigmoid_values)
    df['order_score'] = pd.Series(test_order_logits_sigmoid_values)
    sys.stdout.write("-" * 100)
    get_offline_metrics(df, out_file_test)

def transform_label(df, action):
    check = df['label'] >= action
    df['newlabel'] = check.astype(int)
    return df


def cal_auc(df, action):
    # print("df size:", df.shape)
    # print("df:")
    # print(df)
    df = transform_label(df, action)
    #print("df['newlabel']:", df['newlabel'])
    #result = roc_auc_score(df['newlabel'], df['score'])
    try:
        result = roc_auc_score(df['newlabel'], df['score'])
    except:
        #print("except auc!")
        return 1
    return result


def handle_partition_metric_auc(dfs, clk_output, ord_output):
    auc_clk = [0.0]
    auc_ord = [0.0]

    for df in dfs:
        auc_clk[0] += cal_auc(df, CLICK)
        auc_ord[0] += cal_auc(df, ORDER)

    clk_output.put(auc_clk)
    ord_output.put(auc_ord)

def handle_partition_metric_auc_group_weight(dfs, clk_output, ord_output, weight_output, group_weight_method):
    auc_clk = [0.0]
    auc_ord = [0.0]
    weight = [0.0]

    for df in dfs:
        #print("\ndf:", df)
        w = 1
        df['newlabel'] = (df['label'] >= 1).astype(int)
        if(group_weight_method == "impression"):
            w = len(df)
        elif(group_weight_method == "click"):
            w = sum(df['newlabel'])

        auc_clk[0] += cal_auc(df, CLICK) * w
        auc_ord[0] += cal_auc(df, ORDER) * w
        weight[0] += w

        #print("w:", w)

    clk_output.put(auc_clk)
    ord_output.put(auc_ord)
    weight_output.put(weight)


def get_offline_metrics_auc_mix(df):
    auc_clk = cal_auc(df, CLICK)
    auc_order = cal_auc(df, ORDER)
    return (auc_clk, auc_order)


def get_offline_metrics_auc_group_weight_df(df, group_method="uuid", group_weight_method="impression"):
    #print("get_offline_metrics_auc_df")

    grp = df.groupby(group_method)

    process_num = int(mp.cpu_count() * 0.7)
    process_grp_list = {}
    for pid in range(process_num):
        process_grp_list[pid] = []

    gid = 0
    cnt_group_len1=0
    for name, group in grp:
        if(len(group) == 1):
            cnt_group_len1 += 1
            continue
        process_grp_list[gid % process_num].append(group)
        gid = gid + 1
    #print("cnt_group_len1:", cnt_group_len1)
    #print("cnt_group_valid:", gid)

    clk_outputs = [mp.Queue() for pid in range(process_num)]
    ord_outputs = [mp.Queue() for pid in range(process_num)]
    weight_outputs = [mp.Queue() for pid in range(process_num)]

    # Setup a list of processes that we want to run
    processes = [mp.Process(target=handle_partition_metric_auc_group_weight, \
                            args=(process_grp_list[pid], clk_outputs[pid], ord_outputs[pid], weight_outputs[pid], group_weight_method)) for pid in
                 range(process_num)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    clk_results = [np.array(clk_outputs[pid].get()) for pid in range(process_num)]
    ord_results = [np.array(ord_outputs[pid].get()) for pid in range(process_num)]
    weight_results = [np.array(weight_outputs[pid].get()) for pid in range(process_num)]

    sum = np.zeros(1)
    sum_w = np.zeros(1)
    for i in range(len(ord_results)):
        sum += ord_results[i]
        sum_w += weight_results[i]
    avg = sum / sum_w
    ord_metric = avg

    sum = np.zeros(1)
    sum_w = np.zeros(1)
    for i in range(len(clk_results)):
        sum += clk_results[i]
        sum_w += weight_results[i]
    avg = sum / sum_w
    clk_metric = avg
    del df

    return {CLICK: clk_metric[0], ORDER: ord_metric[0]}  # return ord_metric, clk_metric


def get_offline_metrics_auc_df(df, group_method="uuid"):
    #print("get_offline_metrics_auc_df")

    grp = df.groupby(group_method)

    process_num = int(mp.cpu_count() * 0.7)
    process_grp_list = {}
    for pid in range(process_num):
        process_grp_list[pid] = []

    gid = 0
    cnt_group_len1=0
    for name, group in grp:
        if(len(group) == 1):
            cnt_group_len1 += 1
            continue
        process_grp_list[gid % process_num].append(group)
        gid = gid + 1
    #print("cnt_group_len1:", cnt_group_len1)
    #print("cnt_group_valid:", gid)

    clk_outputs = [mp.Queue() for pid in range(process_num)]
    ord_outputs = [mp.Queue() for pid in range(process_num)]

    # Setup a list of processes that we want to run
    processes = [mp.Process(target=handle_partition_metric_auc, \
                            args=(process_grp_list[pid], clk_outputs[pid], ord_outputs[pid])) for pid in
                 range(process_num)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    clk_results = [np.array(clk_outputs[pid].get()) for pid in range(process_num)]
    ord_results = [np.array(ord_outputs[pid].get()) for pid in range(process_num)]

    sum = np.zeros(1)
    for per_result in ord_results:
        sum += per_result
    avg = sum / gid
    ord_metric = avg

    sum = np.zeros(1)
    for per_result in clk_results:
        sum += per_result
    avg = sum / gid
    clk_metric = avg
    del df

    return {CLICK: clk_metric[0], ORDER: ord_metric[0]}  # return ord_metric, clk_metric


def get_offline_metrics(df, out_file_test):
    # df = df[['label', 'sid', 'score', 'uuid']]
    df = df[['label', 'click_score', 'order_score', 'uuid', 'sid']]

    gid, process_grp_list, process_num = split_group(df)
    print('begin separate_metric\n')

    # separate_metric
    #pre@k
    clk_metric, ord_metric = separate_mrr(gid, process_grp_list, process_num)
    res = {CLICK: clk_metric, ORDER: ord_metric}
    offline_metrics_str = "separate_metric as follows:"
    log_to_file('separate_metric', out_file_test)
    print(offline_metrics_str)
    for action, metric in res.items():
        offline_metrics_str = ''
        metric_threshlod_pair = zip(at_list, metric)
        for tuple0, tuple1 in metric_threshlod_pair:
            offline_metrics_str += "action_{a}_at_{n}: {m}\n".format(a=action, n=tuple0, m=tuple1)
        log_to_file(offline_metrics_str, out_file_test)
        print(offline_metrics_str)

    # #auc
    # metric_sets = get_offline_metrics_auc_df(df, group_method="uuid")
    # for action, metric in metric_sets.items():
    #     offline_metrics_str = ''
    #     offline_metrics_str += "action_{a}_auc: {m}\n".format(a=action, m=metric[0])
    #     log_to_file(offline_metrics_str, out_file_test)
    #     print(offline_metrics_str)


    max_value = 0.0
    max_key = ''

    #weight = [(1.0, 0.25), (1.0, 0.5), (1.0, 1.0), (1.0, 2), (1.0, 4), (1.0, 8), (1.0, 16), (1.0, 32), (1.0, 64), (1.0, 128), (1.0, 256)]
    weight = [(1.0, 0.05), (1.0, 0.1), (1.0, 0.25), (1.0, 0.5), (1.0, 1.0), (1.0, 2), (1.0, 4), (1.0, 8)]
    #weight = [(1.0, 0.25), (1.0, 0.5)]

    # grid_measure_max_action = '2'
    # grid_measure_max_k = '2'

    for click_weight, order_weight in weight:  # 20
        del gid, process_grp_list, process_num
        offline_metrics_str = "+" * 100
        log_to_file(offline_metrics_str, out_file_test)
        print(offline_metrics_str)

        print("begin search {}_{}\n".format(click_weight, order_weight))
        df['score'] = (click_weight * df['click_score'] + order_weight * df['order_score']) / (click_weight + order_weight)
        gid, process_grp_list, process_num = split_group(df)

        log_to_file(str(click_weight) + "_" + str(order_weight) + '\n', out_file_test)
        #pre@k, mrr@k
        metric_sets, at_list_ = calculate_pre_mrr(gid, process_grp_list, process_num)
        print(str(click_weight)  + "_" + str(order_weight) + '\n')
        for action, metric in metric_sets.items():
            metric_pre, metric_mrr = metric
            offline_metrics_str = ''

            metric_threshlod_pair = zip(at_list_, metric_pre)
            for tuple0, tuple1 in metric_threshlod_pair:
                offline_metrics_str += "action_{a}_pre_at_{n}: {m}\n".format(a=action, n=tuple0, m=tuple1)
                #choose best clk_pre@4
                if str(action) == '2' and str(tuple0) == '4' and float(tuple1) > max_value:
                    max_value = float(tuple1)
                    max_key = str(click_weight) + "_" + str(order_weight)

            offline_metrics_str += "\n"

            metric_threshlod_pair = zip(at_list_, metric_mrr)
            for tuple0, tuple1 in metric_threshlod_pair:
                offline_metrics_str += "action_{a}_mrr_at_{n}: {m}\n".format(a=action, n=tuple0, m=tuple1)

            offline_metrics_str += "\n"

            log_to_file(offline_metrics_str, out_file_test)
            print(offline_metrics_str)


        #auc: mix user data together
        offline_metrics_str = "Mix_user_AUC: \n"
        mix_auc_clk, mix_auc_ord = get_offline_metrics_auc_mix(df)
        offline_metrics_str += "mix_user_auc_clk: {m}\n".format(m=mix_auc_clk)
        offline_metrics_str += "mix_user_auc_ord: {m}\n".format(m=mix_auc_ord)
        log_to_file(offline_metrics_str, out_file_test)
        print(offline_metrics_str)

        #auc: group by user
        offline_metrics_str = "Group_user_AUC: \n"
        metric_sets = get_offline_metrics_auc_df(df, group_method="uuid")
        group_user_auc_clk = metric_sets[CLICK]
        group_user_auc_ord = metric_sets[ORDER]
        group_user_auc_clk_ord_f1 = 2 * group_user_auc_clk * group_user_auc_ord / (group_user_auc_clk + group_user_auc_ord)

        offline_metrics_str += "group_user_auc_clk: {m}\n".format(m=group_user_auc_clk)
        offline_metrics_str += "group_user_auc_ord: {m}\n".format(m=group_user_auc_ord)
        offline_metrics_str += "group_user_auc_f1_clk_ord: {m}\n".format(m=group_user_auc_clk_ord_f1)
        log_to_file(offline_metrics_str, out_file_test)
        print(offline_metrics_str)

        # auc: group by user, weightImpression
        offline_metrics_str = "Group_WeightImpression_user_AUC: \n"
        group_weight_method_list = ["impression", "click"]
        group_weight_method = group_weight_method_list[0]
        metric_sets = get_offline_metrics_auc_group_weight_df(df, group_method="uuid", group_weight_method=group_weight_method)

        group_user_auc_clk = metric_sets[CLICK]
        group_user_auc_ord = metric_sets[ORDER]
        offline_metrics_str += "group_weightImpression_user_auc_clk: {m}\n".format(m=group_user_auc_clk)
        offline_metrics_str += "group_weightImpression_user_auc_ord: {m}\n".format(m=group_user_auc_ord)
        log_to_file(offline_metrics_str, out_file_test)
        print(offline_metrics_str)

        # auc: group by user, weight_clk
        offline_metrics_str = "Group_WeightClk_user_AUC: \n"
        group_weight_method_list = ["impression", "click"]
        group_weight_method = group_weight_method_list[1]
        metric_sets = get_offline_metrics_auc_group_weight_df(df, group_method="uuid", group_weight_method=group_weight_method)

        group_user_auc_clk = metric_sets[CLICK]
        group_user_auc_ord = metric_sets[ORDER]
        offline_metrics_str += "group_weightClk_user_auc_clk: {m}\n".format(m=group_user_auc_clk)
        offline_metrics_str += "group_weightClk_user_auc_ord: {m}\n".format(m=group_user_auc_ord)
        log_to_file(offline_metrics_str, out_file_test)
        print(offline_metrics_str)

        # #choose best: auc_f1
        # if(float(group_user_auc_clk_ord_f1) > max_value):
        #     max_value = float(group_user_auc_clk_ord_f1)
        #     max_key = str(click_weight) + "_" + str(order_weight)

        # choose best: group_user_auc_clk
        # if (float(group_user_auc_clk) > max_value):
        #     max_value = float(group_user_auc_clk)
        #     max_key = str(click_weight) + "_" + str(order_weight)

        offline_metrics_str = "+" * 100
        log_to_file(offline_metrics_str, out_file_test)
        print(offline_metrics_str)

    #print("+" * 100)
    # print("grid_measure_max_action:", grid_measure_max_action)
    # print("grid_measure_max_k:", grid_measure_max_k)

    offline_metrics_str = "+" * 100 + "\n"
    offline_metrics_str += "max_key:{0}\n".format(max_key)
    offline_metrics_str += "max_value:{0}\n".format(max_value)
    log_to_file(offline_metrics_str, out_file_test)
    print(offline_metrics_str)


def split_group(df):
    #grp = df.groupby('sid')
    grp = df.groupby(['uuid', 'sid'])
    process_num = mp.cpu_count()
    process_grp_list = {}
    for pid in range(process_num):
        process_grp_list[pid] = []
    gid = 0

    for name, group in grp:
        process_grp_list[gid % process_num].append(group)
        gid = gid + 1

    return gid, process_grp_list, process_num


def calculate_pre_mrr(gid, process_grp_list, process_num):
    #clk_output = mp.Queue()
    #ord_output = mp.Queue()
    clk_outputs_pre = [mp.Queue() for pid in range(process_num)]
    ord_outputs_pre = [mp.Queue() for pid in range(process_num)]
    clk_outputs_mrr = [mp.Queue() for pid in range(process_num)]
    ord_outputs_mrr = [mp.Queue() for pid in range(process_num)]

    # Setup a list of processes that we want to run
    processes = [mp.Process(target=grid_search_metric_pre_mrr, \
                            args=(process_grp_list[pid], clk_outputs_pre[pid], ord_outputs_pre[pid], clk_outputs_mrr[pid], ord_outputs_mrr[pid])) for pid in
                 range(process_num)]
    #sys.stdout.write("========================begin calculate========================\n")
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    #sys.stdout.write("========================join========================\n")
    # Get process results from the output queue
    # ord_results = [np.array(ord_output.get()) for p in processes]
    # clk_results = [np.array(clk_output.get()) for p in processes]
    # sum = np.zeros(len(at_list))
    # for per_result in ord_results:
    #     sum += per_result
    # avg = sum / gid
    # ord_metric = avg
    # sum = np.zeros(len(at_list))
    # for per_result in clk_results:
    #     sum += per_result
    # avg = sum / gid
    # clk_metric = avg
    # return clk_metric, ord_metric

    clk_results_pre = [np.array(clk_outputs_pre[pid].get()) for pid in range(process_num)]
    ord_results_pre = [np.array(ord_outputs_pre[pid].get()) for pid in range(process_num)]

    clk_results_mrr = [np.array(clk_outputs_mrr[pid].get()) for pid in range(process_num)]
    ord_results_mrr = [np.array(ord_outputs_mrr[pid].get()) for pid in range(process_num)]

    sum = np.zeros(len(at_list))
    for per_result in ord_results_pre:
        sum += per_result
    avg = sum / gid
    ord_metric_pre = avg

    sum = np.zeros(len(at_list))
    for per_result in clk_results_pre:
        sum += per_result
    avg = sum / gid
    clk_metric_pre = avg

    sum = np.zeros(len(at_list))
    for per_result in ord_results_mrr:
        sum += per_result
    avg = sum / gid
    ord_metric_mrr = avg

    sum = np.zeros(len(at_list))
    for per_result in clk_results_mrr:
        sum += per_result
    avg = sum / gid
    clk_metric_mrr = avg

    return {CLICK: (clk_metric_pre, clk_metric_mrr),
            ORDER: (ord_metric_pre, ord_metric_mrr)}, at_list  # return ord_metric, clk_metric, at_list

def calculate_mrr(click_weight, gid, order_weight, process_grp_list, process_num):
    clk_output = mp.Queue()
    ord_output = mp.Queue()
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=grid_search_metric, \
                            args=(process_grp_list[pid], clk_output, ord_output, click_weight,
                                   order_weight)) for pid in
                 range(process_num)]
    #sys.stdout.write("========================begin calculate========================\n")
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    #sys.stdout.write("========================join========================\n")
    # Get process results from the output queue
    ord_results = [np.array(ord_output.get()) for p in processes]
    clk_results = [np.array(clk_output.get()) for p in processes]
    sum = np.zeros(len(at_list))
    for per_result in ord_results:
        sum += per_result
    avg = sum / gid
    ord_metric = avg
    sum = np.zeros(len(at_list))
    for per_result in clk_results:
        sum += per_result
    avg = sum / gid
    clk_metric = avg
    return clk_metric, ord_metric


def separate_metric(dfs, clk_output, ord_output):
    at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #sys.stdout.write('{} begin grid_search_metric\n'.format(os.getpid()))
    total_count = len(dfs)
    index = 0
    for df in dfs:
        index += 1
        #print("{}--separate---{}/{} is completed".format(os.getpid(), index, total_count))
        click_df = df.sort_values(by=['click_score', 'label'], ascending=[False, True])
        order_df = df.sort_values(by=['order_score', 'label'], ascending=[False, True])
        for i in range(len(at_list)):
            at_n_list_clk[i] += get_pre_at_n(click_df, at_list[i], CLICK)
            at_n_list_ord[i] += get_pre_at_n(order_df, at_list[i], ORDER)

    clk_output.put(at_n_list_clk)
    ord_output.put(at_n_list_ord)
    #print('{} end grid_search_metric\n'.format(os.getpid()))


def separate_mrr(gid, process_grp_list, process_num):
    clk_output = mp.Queue()
    ord_output = mp.Queue()
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=separate_metric, \
                            args=(process_grp_list[pid], clk_output, ord_output)) for pid in
                 range(process_num)]
    #sys.stdout.write("========================begin calculate========================\n")
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    #sys.stdout.write("========================join========================\n")
    # Get process results from the output queue
    ord_results = [np.array(ord_output.get()) for p in processes]
    clk_results = [np.array(clk_output.get()) for p in processes]

    sum = np.zeros(len(at_list))
    for per_result in ord_results:
        sum += per_result
    avg = sum / gid
    ord_metric = avg
    sum = np.zeros(len(at_list))
    for per_result in clk_results:
        sum += per_result
    avg = sum / gid
    clk_metric = avg
    return clk_metric, ord_metric


def get_accuracy(test_df):
    return (test_df['label_01'] == test_df['predict']).astype(int).mean()


if __name__ == '__main__':
    schema = ['uuid', 'pin', 'pos', 'expo_report_ts', 'content_id', 'sku', 'sid', 'navcnt', 'label']
    df = pd.read_csv("./test.csv")
    out_name = 'result.txt'
    get_offline_metrics(df, out_name)  # offline_order_metrics,offline_click_metrics =
