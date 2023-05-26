"""
对构造的数据集进行分析
"""

import json
from re import X
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from tqdm import tqdm

DPI = 200

## 函数

def draw_hist(data_list, start, end, width, x_label, title, file_name, lambda_expression=None):
    if lambda_expression:
        example = [lambda_expression(item) for item in data_list]
    else:
        example = data_list
    hist_range = np.arange(start, int(end + 1), width)
    frequency_each,_,_= plt.hist(example, hist_range, color='red', width=width,alpha=0.7)  #alpha设置透明度，0为完全透明
    plt.xlabel(x_label)
    plt.ylabel('count')
    plt.xlim(start,end)
    plt.title(title)
    # plt.plot(hist_range[1:]-(hist_range//2),frequency_each,color='palevioletred')  #利用返回值来绘制区间中点连线
    plt.savefig(file_name, dpi=DPI)
    plt.clf()

def get_time_series(data_list, start, end, width, lambda_expression=None):
    if lambda_expression:
        example = [lambda_expression(item) for item in data_list]
    else:
        example = data_list
    hist_range = np.arange(start, int(end + 1), width)
    frequency_each,_,_= plt.hist(example, hist_range, color='red', width=width, alpha=0.7)  #alpha设置透明度，0为完全透明
    plt.clf()
    return frequency_each


## 分析开始
images_overall_save_dir = "images_overall"
images_save_dir = "images"
for dir in [images_overall_save_dir, images_save_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

data_dir = "/home/disk/disk2/lw/covid-19-weibo-processed/renminribao"
repost_dir = os.path.join(data_dir, 'repost_data')

content_file = os.path.join(data_dir, "content.csv")

content_df = pd.read_csv(content_file)

texts = content_df['content'].to_list()
repost_nums = content_df['repost_num'].to_list()
publish_time = content_df['publish_time'].to_list()

legal_weibo_idx = []
for i in range(len(repost_nums)):
    if repost_nums[i] < 100:
        continue
    else:
        legal_weibo_idx.append(i)

print("Legal weibo num:", len(legal_weibo_idx))

legal_weibo = []
legal_weibo_repost_num = []
legal_weibo_publish_time = []
legal_weibo_time_series = []
repost_95_time = []
repost_90_time = []

for idx in tqdm(legal_weibo_idx):
    legal_weibo.append(texts[idx])
    legal_weibo_repost_num.append(repost_nums[idx])
    legal_weibo_publish_time.append(publish_time[idx])

    csv_file = os.path.join(repost_dir, f"{idx}.csv")
    repost_time = pd.read_csv(csv_file)["time"].tolist()
    assert repost_nums[idx] == len(repost_time), "uncorrect weibo."
    ## calc 95% 90% repost time
    repost_len = len(repost_time)
    idx95, idx90 = int(repost_len*0.95), int(repost_len*0.90)
    repost_95_time.append(repost_time[idx95])
    repost_90_time.append(repost_time[idx90])

    repost_time_series = get_time_series(repost_time, 0, 24*3600, 1800)
    legal_weibo_time_series.append(repost_time_series)

# Figure 3: Time to reach .. percent of the repost
print("*****************Repost Limit Statistics************")
repost_95_time, repost_90_time = np.array(repost_95_time) / 3600, np.array(repost_90_time) / 3600
frequency_each, _, _= plt.hist(repost_90_time, np.arange(0, 73))
plt.title("Time to reach 90 percent of the repost")
plt.xlabel("hours")
plt.ylabel("Weibo number")
plt.xlim(0, 72)
plt.savefig(os.path.join(images_overall_save_dir, "repost90p.png"), dpi=DPI)
plt.clf()
frequency_each, _, _= plt.hist(repost_95_time, np.arange(0, 73))
plt.title("Time to reach 95 percent of the repost")
plt.xlabel("hours")
plt.ylabel("Weibo number")
plt.xlim(0, 72)
plt.savefig(os.path.join(images_overall_save_dir, "repost95p.png"), dpi=DPI)
plt.clf()
print("Repost 95%% time: %.2f, std: %.2f; Repost 90%% time: %.2f, std: %.2f" % (np.mean(repost_95_time), np.std(repost_95_time), np.mean(repost_90_time), np.std(repost_90_time)))

# Figure 4: A micro-blog posted at ...
i = 4
x_label = np.arange(0.5, 24.5, 0.5)
## 增量转发数图
plt.clf()
idx = legal_weibo_idx[i]
plt.plot(x_label, legal_weibo_time_series[i], color='red', marker='v',linestyle='--')
plt.xlim(0, 24)
plt.ylim(bottom=0)
plt.xlabel("hours")
plt.ylabel("repost number in last half hour")
plt.title("Repost Number Series")
plt.savefig(os.path.join(images_save_dir,  f"{idx}_repost.png"), dpi=DPI)

## 累计转发数图
new_array = np.zeros_like(legal_weibo_time_series[i])
for j in range(len(legal_weibo_time_series[i])):
    new_array[j] = np.sum(legal_weibo_time_series[i][:j+1])
new_array = np.concatenate([[0], new_array])
x_label = np.concatenate([[0], x_label])
plt.clf()
plt.plot(x_label, new_array, color='green', marker='v',linestyle='--')
plt.xlim(0, 24)
plt.ylim(bottom=0)
plt.xlabel("hours")
plt.ylabel("cumulative repost number")
plt.title("Cumulative Repost Number Series")
plt.savefig(os.path.join(images_save_dir,  f"{idx}_cumulative_repost.png"), dpi=DPI)


# Figure 5 : Relation between prediction error and observation time
result_dir = 'images_overall/reg/'

mse_series_metrics = [str(float(i)) + 'h_mse' for i in range(13)]
mse_last_series_metrics = [str(float(i)) + 'h_mse_last' for i in range(13)]

mse_result = {'rnn': [], 'tcn': [], 'RepostLSTM-AF': [], 'RepostLSTM-MF': []}
mse_last_result = {'rnn': [], 'tcn': [], 'RepostLSTM-AF': [], 'RepostLSTM-MF': []}

for key in mse_result:
    file_path = result_dir + key + '.csv'
    df = pd.read_csv(file_path)
    for metric in mse_series_metrics:
        mse_result[key].append(df.iloc[0][metric])
    for metric in mse_last_series_metrics:
        mse_last_result[key].append(df.iloc[0][metric])
    mse_result[key], mse_last_result[key] = np.array(mse_result[key]), np.array(mse_last_result[key])

X = [i for i in range(13)]
plt.clf()
plt.plot(X, mse_result['rnn'], color='red', marker='o',linestyle='dotted', label='LSTM')
plt.plot(X, mse_result['tcn'], color='orange', marker='o',linestyle='dotted', label='TCN')
plt.plot(X, mse_result['RepostLSTM-AF'], color='darkcyan', marker='o',linestyle=None, label='RepostLSTM-AF')
plt.plot(X, mse_result['RepostLSTM-MF'], color='blue', marker='o',linestyle=None, label='RepostLSTM-MF')
plt.xlim(0, 12)
plt.ylim(0, 1.25)
plt.xlabel("Observe Time (hours)")
plt.ylabel("MSE")
plt.title("MSE of Weibo Series Prediction")
plt.legend()
plt.savefig('images_overall/observation_error1.png', dpi=200)

plt.clf()
plt.plot(X, mse_last_result['rnn'], color='red', marker='o',linestyle='dotted', label='LSTM')
plt.plot(X, mse_last_result['tcn'], color='orange', marker='o',linestyle='dotted', label='TCN')
plt.plot(X, mse_last_result['RepostLSTM-AF'], color='darkcyan', marker='o',linestyle=None, label='RepostLSTM-AF')
plt.plot(X, mse_last_result['RepostLSTM-MF'], color='blue', marker='o',linestyle=None, label='RepostLSTM-MF')
plt.xlim(0, 12)
plt.ylim(0, 1.3)
plt.xlabel("Observe Time (hours)")
plt.ylabel("MSE")
plt.title("MSE of Weibo Popularity Prediction")
plt.legend()
plt.savefig('images_overall/observation_error2.png', dpi=200)

