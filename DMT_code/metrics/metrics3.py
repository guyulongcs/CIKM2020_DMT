import pandas as pd
import multiprocessing as mp
import numpy as np
import os
import sys
import subprocess

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path + '../data_feed')
import gc
from ctypes import cdll, CDLL

sys.path.append(file_path + '../util')
from util import *

cdll.LoadLibrary("libc.so.6")
libc = CDLL("libc.so.6")


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


CLICK = 2
ORDER = 5

#at_list = [2, 4, 6, 10, 14, 20]
at_list = [2, 4, 6, 8, 10, 12, 14]


def grid_search_metric(dfs, clk_output, nav_output, ord_output, click_weight, order_weight):
    at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sys.stdout.write('{} begin grid_search_metric\n'.format(os.getpid()))
    total_count = len(dfs)
    index = 0
    for df in dfs:
        index += 1
        print("{}--{}_{}---{}/{} is completed".format(os.getpid(), click_weight, order_weight,
                                                         index, total_count))
        # df['score'] = click_weight * df['click_score'] + detail_weight * df['detail_score'] + order_weight * df[
        #     'order_score']
        final_df = df.sort_values(by=['score', 'label'], ascending=[False, True])
        for i in range(len(at_list)):
            at_n_list_clk[i] += get_pre_at_n(final_df, at_list[i], CLICK)
            at_n_list_ord[i] += get_pre_at_n(final_df, at_list[i], ORDER)

    clk_output.put(at_n_list_clk)
    ord_output.put(at_n_list_ord)
    sys.stdout.write('{} end grid_search_metric\n'.format(os.getpid()))


def handle_partition_metric(dfs, clk_output, ord_output):
    at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
                  test_order_logits_sigmoid_values, file_name, version, checkpoint):
    if sys.version_info[0] < 3:
        df = pd.DataFrame([value.split('\t') for value in headers], columns=header_schema)
    else:
        df = pd.DataFrame([value.decode().strip().split('\t') for value in headers], columns=header_schema)

    df['click_score'] = pd.Series(test_click_logits_sigmoid_values)
    df['order_score'] = pd.Series(test_order_logits_sigmoid_values)
    df.to_csv("./res/{0}_test_{1}.csv".format(version,checkpoint), index=None)
    # with open("./res/{0}_test_{1}.csv".format(version, checkpoint), 'w') as w:
    #     w.write(",".join(header_schema + ['click_score', 'detail_score', 'order_score']) + '\n')
    #     for i, header in enumerate(headers):
    #         if i % 1000 == 0:
    #             print('save result', i)
    #         w.write(",".join(header.strip().split('\t') + [str(test_click_logits_sigmoid_values[i])] +
    #                          [str(test_detail_logits_sigmoid_values[i])] + [
    #                              str(test_order_logits_sigmoid_values[0])]) + '\n')


def save_weights_to_local(click_weight_value_list, order_weight_value_list):
    df = pd.DataFrame([])
    df['click_weight'] = pd.Series(click_weight_value_list)
    df['order_weight'] = pd.Series(order_weight_value_list)
    df.to_csv("./res/weight.csv", index=None)
    # print('save weight')
    # with open("./output/weight.csv", 'w') as w:
    #     w.write(','.join(['click_weight','detail_weight','order_weight'])+'\n')
    #     for i, v in enumerate(click_weight_value_list):
    #         if i % 1000 == 0:
    #             print('save weight', i)
    #         w.write(",".join([str(v), str(detail_weight_value_list[i]), str(order_weight_value_list[i])]) + '\n')


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


def get_offline_metrics(df, out_file_test):
    # df = df[['label', 'sid', 'score', 'uuid']]
    df = df[['label', 'click_score', 'order_score', 'uuid']]

    gid, process_grp_list, process_num = split_group(df)
    print('begin separate_metric\n')

    # separate_metric
    clk_metric, ord_metric = separate_mrr(gid, process_grp_list, process_num)
    res = {ORDER: ord_metric, CLICK: clk_metric}
    log_to_file('separate_metric', out_file_test)
    print('separate_metric as follows:')
    for action, metric in res.items():
        offline_metrics_str = ''
        metric_threshlod_pair = zip(at_list, metric)
        for tuple0, tuple1 in metric_threshlod_pair:
            offline_metrics_str += "action_{a}_at_{n}: {m}\n".format(a=action, n=tuple0, m=tuple1)
        log_to_file(offline_metrics_str, out_file_test)
        print(offline_metrics_str)

    max_value = 0.0
    max_key = ''
    for click_weight in np.arange(0.1, 1.1, 0.1):
        for order_weight in np.arange(0.1, 1.1, 0.1):
            del gid, process_grp_list, process_num
            print("begin search {}_{}_{}\n".format(click_weight, order_weight))
            df['score'] = click_weight * df['click_score'] + order_weight * df[
                'order_score']
            gid, process_grp_list, process_num = split_group(df)
            clk_metric, nav_metric, ord_metric = calculate_mrr(click_weight, gid, order_weight,
                                                               process_grp_list, process_num)
            res = { ORDER: ord_metric, CLICK: clk_metric}
            log_to_file(str(click_weight)  + "_" + str(order_weight) + '\n',
                        out_file_test)
            print(str(click_weight) + "_" + str(order_weight) + '\n')
            for action, metric in res.items():
                offline_metrics_str = ''
                metric_threshlod_pair = zip(at_list, metric)
                for tuple0, tuple1 in metric_threshlod_pair:
                    offline_metrics_str += "action_{a}_at_{n}: {m}\n".format(a=action, n=tuple0, m=tuple1)
                    if str(action) == '2' and str(tuple0) == '4' and float(tuple1) > max_value:
                        max_value = float(tuple1)
                        max_key = str(click_weight)  + "_" + str(order_weight)

                log_to_file(offline_metrics_str, out_file_test)
                print(offline_metrics_str)
    print("+" * 100)
    print("max_value:{0}".format(max_value))
    log_to_file("max_value:{0}".format(max_value), out_file_test)
    print("max_key:{0}".format(max_key))
    log_to_file("max_key:{0}".format(max_key), out_file_test)


def split_group(df):
    #grp = df.groupby('uuid')
    grp = df.groupby(['uuid', 'sid'])
    cpu_count = get_available_cpu_num()
    process_num = int(cpu_count * 0.7)
    process_grp_list = {}
    for pid in range(process_num):
        process_grp_list[pid] = []
    gid = 0

    for name, group in grp:
        process_grp_list[gid % process_num].append(group)
        gid = gid + 1

    return gid, process_grp_list, process_num


def calculate_mrr(click_weight, gid, order_weight, process_grp_list, process_num):
    clk_output = mp.Queue()
    nav_output = mp.Queue()
    ord_output = mp.Queue()
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=grid_search_metric, \
                            args=(process_grp_list[pid], clk_output, nav_output, ord_output, click_weight,
                                   order_weight)) for pid in
                 range(process_num)]
    sys.stdout.write("========================begin calculate========================\n")
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    sys.stdout.write("========================join========================\n")
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


def separate_metric(dfs, clk_output, nav_output, ord_output):
    at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sys.stdout.write('{} begin grid_search_metric\n'.format(os.getpid()))
    total_count = len(dfs)
    index = 0
    for df in dfs:
        index += 1
        print("{}--separate---{}/{} is completed".format(os.getpid(), index, total_count))
        click_df = df.sort_values(by=['click_score', 'label'], ascending=[False, True])
        order_df = df.sort_values(by=['order_score', 'label'], ascending=[False, True])
        for i in range(len(at_list)):
            at_n_list_clk[i] += get_pre_at_n(click_df, at_list[i], CLICK)
            at_n_list_ord[i] += get_pre_at_n(order_df, at_list[i], ORDER)

    clk_output.put(at_n_list_clk)
    ord_output.put(at_n_list_ord)
    print('{} end grid_search_metric\n'.format(os.getpid()))


def separate_mrr(gid, process_grp_list, process_num):
    clk_output = mp.Queue()
    ord_output = mp.Queue()
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=separate_metric, \
                            args=(process_grp_list[pid], clk_output, ord_output)) for pid in
                 range(process_num)]
    sys.stdout.write("========================begin calculate========================\n")
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    sys.stdout.write("========================join========================\n")
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
    df = pd.read_csv("../output/test.csv")
    out_name = 'result.txt'
    get_offline_metrics(df, out_name)  # offline_order_metrics,offline_click_metrics =
