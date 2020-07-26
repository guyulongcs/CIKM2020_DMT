import pandas as pd
import multiprocessing as mp
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path + '../data_feed')


# args is a tuple
# return a float
def get_pre_at_n(df, N, action):
    # descending by score, and if scores are same then ascending by label
    sorted_df = df.head(N)
    # length is N if df have at least N lines or
    # length is smaller than N if df have less than N lines
    (length, width) = sorted_df.shape
    if length == 0:
        return 0
    check = sorted_df['label'] >= action
    # print check
    # print type(check)
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

CLICK = 2
ORDER = 5
#at_list = [2, 4, 10, 12, 20, 24, 40]
at_list = [2, 4, 6, 8, 10, 12, 14]



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


def handle_partition_metric_old(dfs, clk_output, ord_output):
    at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for df in dfs:
        sorted_df = df.sort_values(by=['score', 'label'], ascending=[False, True])
        for i in range(len(at_list)):
            at_n_list_clk[i] += get_pre_at_n(sorted_df, at_list[i], CLICK)
            at_n_list_ord[i] += get_pre_at_n(sorted_df, at_list[i], ORDER)

    clk_output.put(at_n_list_clk)
    ord_output.put(at_n_list_ord)

def handle_partition_metric(dfs, clk_output_pre, ord_output_pre, clk_output_mrr, ord_output_mrr):
    pre_at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pre_at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mrr_at_n_list_clk = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mrr_at_n_list_ord = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for df in dfs:
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


def handle_partition_metric_auc(dfs, clk_output, ord_output):
    auc_clk = [0.0]
    auc_ord = [0.0]

    for df in dfs:
        auc_clk[0] += cal_auc(df, CLICK)
        auc_ord[0] += cal_auc(df, ORDER)

    clk_output.put(auc_clk)
    ord_output.put(auc_ord)


def get_offline_metrics(header_schema, headers, test_eval_logits_sigmoid_values):
    if sys.version_info[0] < 3:
        df = pd.DataFrame([value.split('\t') for value in headers], columns=header_schema)
    else:
        df = pd.DataFrame([value.decode().strip().split('\t') for value in headers], columns=header_schema)
    df['label'] = df['label'].astype(int)

    df['score'] = pd.Series(test_eval_logits_sigmoid_values)
    df = df[['label', 'sid', 'score']]

    grp = df.groupby('sid')

    process_num = int(mp.cpu_count() * 0.7)
    #process_num = 1
    process_grp_list = {}
    for pid in range(process_num):
        process_grp_list[pid] = []

    gid = 0
    for name, group in grp:
        process_grp_list[gid % process_num].append(group)
        gid = gid + 1

    clk_outputs_pre = [mp.Queue() for pid in range(process_num)]
    ord_outputs_pre = [mp.Queue() for pid in range(process_num)]
    clk_outputs_mrr = [mp.Queue() for pid in range(process_num)]
    ord_outputs_mrr = [mp.Queue() for pid in range(process_num)]


    # Setup a list of processes that we want to run
    processes = [mp.Process(target=handle_partition_metric, \
                            args=(process_grp_list[pid], clk_outputs_pre[pid], ord_outputs_pre[pid], clk_outputs_mrr[pid], ord_outputs_mrr[pid])) for pid in
                 range(process_num)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
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


    del df

    return {CLICK: (clk_metric_pre, clk_metric_mrr), ORDER: (ord_metric_pre, ord_metric_mrr)}, at_list  # return ord_metric, clk_metric, at_list




def get_offline_metrics_auc(header_schema, headers, test_eval_logits_sigmoid_values, group_method="uuid"):
    print("get_offline_metrics_auc")
    #group_method: 'uuid' / 'sid'
    print("group_method:", group_method)

    if sys.version_info[0] < 3:
        df = pd.DataFrame([value.split('\t') for value in headers], columns=header_schema)
    else:
        df = pd.DataFrame([value.decode().strip().split('\t') for value in headers], columns=header_schema)
    df['label'] = df['label'].astype(int)

    df['score'] = pd.Series(test_eval_logits_sigmoid_values)
    df = df[['label', 'sid', 'score', 'uuid']]

    #df.to_csv("auc.csv", sep='\t', index=False)

    return get_offline_metrics_auc_df(df, group_method)

def get_offline_metrics_auc_df(df, group_method="uuid"):
    print("get_offline_metrics_auc_df")

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
    print("cnt_group_len1:", cnt_group_len1)
    print("cnt_group_valid:", gid)

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

    return {CLICK: clk_metric, ORDER: ord_metric}  # return ord_metric, clk_metric


def get_accuracy(test_df):
    return (test_df['label_01'] == test_df['predict']).astype(int).mean()


if __name__ == '__main__':
    schema = ['sid', 'sku', 'label', 'label_01', 'score']
    df = pd.read_csv(
        "./validation-data",
        header=None,
        names=schema,
        sep='\t',
        encoding="utf-8")
    get_offline_metrics(df)  # offline_order_metrics,offline_click_metrics =
