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

def load_repost_data(idx, data_dir="repost_data"):
    file = os.path.join(data_dir, str(idx) + ".csv")
    df = pd.read_csv(file)
    return df['time'].to_list()

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

# 统计分位数
print("**********************Repost Num*********************")
print("Repost Num: 99%: " + str(np.percentile(legal_weibo_repost_num, 99)))
print("Repost Num: 95%: " + str(np.percentile(legal_weibo_repost_num, 95)))
print("Repost Num: 90%: " + str(np.percentile(legal_weibo_repost_num, 90)))
print("Repost Num: 10%: " + str(np.percentile(legal_weibo_repost_num, 50)))
print("Repost Num: 5%: " + str(np.percentile(legal_weibo_repost_num, 5)))
print("Repost Num: 1%: " + str(np.percentile(legal_weibo_repost_num, 1)))


# 筛选编号
def filter(array, max_length=5000):
    """
    return a index list where array[index] < max_length
    """
    result = []
    for i in range(len(array)):
        if array[i] < max_length:
            result.append(i)
    return result

print("min repost num:", min(legal_weibo_repost_num), "max repost num:", max(legal_weibo_repost_num))
print("avg_repost_num:", np.average(legal_weibo_repost_num))

base_time = time.mktime(time.strptime('2020/01/01 00:00:00', '%Y/%m/%d %H:%M:%S'))

print("min weibo time idx:", legal_weibo_idx[ np.argmin(legal_weibo_publish_time) ] )
print("min weibo time:", time.localtime( min(legal_weibo_publish_time) + base_time) )

print("max weibo time idx:", legal_weibo_idx[ np.argmax(legal_weibo_publish_time) ] )
print("max weibo time:", time.localtime( max(legal_weibo_publish_time) + base_time) )

draw_hist(legal_weibo_repost_num, 0, 11000, 200, "repost number", "Repost Number Histogram", "images_overall/repost_num.png")
# 挑选几个例子画图

print("**********************Content Len*********************")
legal_weibo_content_len = [len(weibo) for weibo in legal_weibo]
print("Content Len: 99%: " + str(np.percentile(legal_weibo_content_len, 99)))
print("Content Len: 95%: " + str(np.percentile(legal_weibo_content_len, 95)))
print("Content Len: 90%: " + str(np.percentile(legal_weibo_content_len, 90)))
print("max:", max(legal_weibo_content_len), "min:", min(legal_weibo_content_len))
print("avg:", np.average(legal_weibo_content_len))
draw_hist(legal_weibo_content_len, 0, max(legal_weibo_content_len) + 100, 100, "content length", "Content Length Histogram", "images_overall/content_len.png")

legal_weibo_time_series = np.array(legal_weibo_time_series)

for i in range(10):
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

## 全部样本的平均转发数趋势
gross_repost_series = np.average(legal_weibo_time_series, axis=0)
## 增量转发数图
x_label = np.arange(0.5, 24.5, 0.5)
plt.clf()
plt.plot(x_label, gross_repost_series, color='red', marker='^',linestyle=None)
plt.xlim(0, 24)
plt.ylim(bottom=0)
plt.xlabel("hours")
plt.ylabel("repost number in last half hour")
plt.title("Average Repost Number Series")
plt.savefig(os.path.join(images_overall_save_dir,  f"gross_repost.png"), dpi=DPI)

## 累计转发数图
new_array = np.zeros_like(gross_repost_series)
for i in range(len(gross_repost_series)):
    new_array[i] = np.sum(gross_repost_series[:i+1])
new_array = np.concatenate([[0], new_array])
x_label = np.concatenate([[0], x_label])
plt.clf()
plt.plot(x_label, new_array, color='green', marker='^',linestyle=None)
plt.xlim(0, 24)
plt.ylim(bottom=0)
plt.xlabel("hours")
plt.ylabel("cumulative repost number")
plt.title("Cumulative Averge Repost Number Series")
plt.savefig(os.path.join(images_overall_save_dir,  f"gross_cumulative_repost.png"), dpi=DPI)

# repost_vectors_48h = []
# repost_vectors_24h = []
# text_df_48h = {'content': []}
# text_df_24h = {'content': []}

# print("loading time series...")
# for number in tqdm(set_95):
#     repost_vector = get_time_series(data_list=load_repost_data(number), start=0, end=48, width=1)
#     repost_vectors_48h.append(np.array(repost_vector))
#     text_df_48h['content'].append(content_text[number])
#     repost_vector = get_time_series(data_list=load_repost_data(number), start=0, end=24, width=1)
#     repost_vectors_24h.append(np.array(repost_vector))
#     text_df_24h['content'].append(content_text[number])

# np.save('data/48', np.array(repost_vectors_48h))
# np.save('data/24', np.array(repost_vectors_24h))
# pd.DataFrame(text_df_48h).to_csv('data/48.csv')
# pd.DataFrame(text_df_24h).to_csv('data/24.csv')

