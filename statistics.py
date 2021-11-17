"""
对test.py产生的数据进行简单的分析
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


def load_repost_data(idx, data_dir="repost_data"):
    file = os.path.join(data_dir, str(idx) + ".csv")
    df = pd.read_csv(file)
    return df['time'].to_list()

def draw_hist(data_list, start, end, width, x_label, file_name, lambda_expression=None):
    if lambda_expression:
        example = [lambda_expression(item) for item in data_list]
    else:
        example = data_list
    hist_range = np.arange(start, int(end + 1), width)
    frequency_each,_,_= plt.hist(example, hist_range, color='red', width=width,alpha=0.7)  #alpha设置透明度，0为完全透明
    plt.xlabel(x_label)
    plt.ylabel('count')
    plt.xlim(start,end)
    # plt.plot(hist_range[1:]-(hist_range//2),frequency_each,color='palevioletred')  #利用返回值来绘制区间中点连线
    plt.savefig(file_name)
    plt.clf()

def get_time_series(data_list, start, end, width, lambda_expression=None):
    if lambda_expression:
        example = [lambda_expression(item) for item in data_list]
    else:
        example = data_list
    hist_range = np.arange(start, int(end + 1), width)
    frequency_each,_,_= plt.hist(example, hist_range, color='red', width=width,alpha=0.7)  #alpha设置透明度，0为完全透明
    plt.clf()
    return frequency_each

file_name = []
for root, dirs, files in os.walk("repost_data", topdown=False):
    for name in files:
        file_name.append(os.path.join(root, name))

print(len(file_name))

content_df = pd.read_csv("content.csv")

content_text = content_df["content"].to_list()
repost_num = content_df["repost_num"].to_list()

# 统计分位数
print("Repost Num: 99%: " + str(np.percentile(repost_num, 99)))
print("Repost Num: 95%: " + str(np.percentile(repost_num, 95)))
print("Repost Num: 90%: " + str(np.percentile(repost_num, 90)))

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

set_95 = filter(repost_num, max_length=np.percentile(repost_num, 95))

# 挑选几个例子画图

random.shuffle(set_95)
# for i in range(20):
#     draw_hist(data_list=load_repost_data(set_95[i]), start=0, end=200, width=10,
#                 x_label='days', file_name='images/days'+str(i)+'.jpg', 
#                 lambda_expression=lambda x: x/24)
#     draw_hist(data_list=load_repost_data(set_95[i]), start=0, end=48, width=1,
#                 x_label='hours', file_name='images/hours'+str(i)+'.jpg')
#     draw_hist(data_list=load_repost_data(set_95[i]), start=0, end=200, width=10,
#                 x_label='log(days)', file_name='images/log_days'+str(i)+'.jpg', 
#                 lambda_expression=lambda x: np.log(x/24+1))
#     draw_hist(data_list=load_repost_data(set_95[i]), start=0, end=48, width=1,
#                 x_label='log(hours)', file_name='images/log_hours'+str(i)+'.jpg',
#                 lambda_expression=lambda x: np.log(x+1))

repost_vectors_48h = []
repost_vectors_24h = []
text_df_48h = {'content': []}
text_df_24h = {'content': []}

print("loading time series...")
for number in tqdm(set_95):
    repost_vector = get_time_series(data_list=load_repost_data(number), start=0, end=48, width=1)
    repost_vectors_48h.append(np.array(repost_vector))
    text_df_48h['content'].append(content_text[number])
    repost_vector = get_time_series(data_list=load_repost_data(number), start=0, end=24, width=1)
    repost_vectors_24h.append(np.array(repost_vector))
    text_df_24h['content'].append(content_text[number])

np.save('data/48', np.array(repost_vectors_48h))
np.save('data/24', np.array(repost_vectors_24h))
pd.DataFrame(text_df_48h).to_csv('data/48.csv')
pd.DataFrame(text_df_24h).to_csv('data/24.csv')

# repost_vectors_48h = np.load('data/48.npy')
# repost_vectors_24h = np.load('data/24.npy')

LOG = True
if LOG:
    repost_vectors_48h = np.log(repost_vectors_48h + 1)
    repost_vectors_24h = np.log(repost_vectors_24h + 1)

# K-均值聚类
from sklearn.cluster import KMeans
x = np.arange(2,11,1)
cluster_loss_48h = []
cluster_loss_24h = []
for i in x:
    n_clusters=i
    cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(repost_vectors_48h)
    # print("Kmeans%d Loss:" % n_clusters, cluster.inertia_)
    cluster_loss_48h.append(cluster.inertia_)
    cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(repost_vectors_24h)
    cluster_loss_24h.append(cluster.inertia_)


plt.plot(x, cluster_loss_24h)
plt.xlabel("Cluster Num")
plt.ylabel("Cluster Loss")
plt.title("24h time series cluster result")
plt.savefig("LOG24h.png" if LOG else "24h.png")
plt.clf()
plt.plot(x, cluster_loss_48h)
plt.xlabel("Cluster Num")
plt.ylabel("Cluster Loss")
plt.title("48h time series cluster result")
plt.savefig("LOG48h.png" if LOG else "48h.png")
plt.clf()