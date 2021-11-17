"""
处理最原始的数据
renminribao-all-posts-reposts.json
"""


import json
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm


def describe(num_list, name):
    num_list = np.array(num_list)
    max_num, min_num = np.max(num_list), np.min(num_list)
    mean, std = np.mean(num_list), np.std(num_list)
    print("**%s** max:%.2f, min:%.2f, interval:%.2f, mean:%.2f, std:%.2f" % (name, max_num, min_num, max_num-min_num, mean, std))

def get_absolute_hours(str):
    # 应对没有秒数的脏数据
    if len(str.split(':')) == 2:
        str += ':00'
    if str[4] == '-':
        return time.mktime(time.strptime(str, '%Y-%m-%d %H:%M:%S'))/3600
    else:
        return time.mktime(time.strptime(str, '%Y/%m/%d %H:%M:%S'))/3600


raw_data_dir = "/home/disk/disk1/weibo_original_cleaned"
data_file = "/home/disk/disk2/lyj-files/tmp-data/covid-19-weibo-from-daiyixin/renminribao-all-posts-reposts.json"

print("Program Start")
start = time.time()

all_data = None
with open(data_file, 'r') as f:
    lines = 0
    for line in f.readlines():
        all_data = json.loads(line.strip(), encoding='utf-8')
        lines += 1
    assert lines == 1

end = time.time()
print("*************LOAD COST: %.2fmin**************" % ((end-start)/60) )

"""
data item:
{
    'original_content': "XXXX", 
    'original_author': "XXXX", 
    'reposts': [
        {   
            'repost_content': "XXXXXXXX",
            'repost_timestamp': "2020-02-06 18:20:00",
            'verification': "XXXX",
            'area': "XXXXX",
            'sex': "XXXXXX",
        },
    ]
}
"""
content_len = []
content = []
repost_num = []
repost_interval = []
repost_time_avg = []
repost_time_std = []


repost_time = []

start = time.time()

for item in tqdm(all_data):
    content_len.append(len(item['original_content']))
    content.append(item['original_content'])

    temp_time_list = []
    min_time = 9999999
    max_time = 0
    for repost in item['reposts']:
        item_repost_time = get_absolute_hours(repost['repost_timestamp'])
        temp_time_list.append(item_repost_time)
        min_time = min(min_time, item_repost_time)
        max_time = max(max_time, item_repost_time)
    
    repost_time.append([value-min_time for value in temp_time_list])    
    repost_interval.append(max_time-min_time)
    repost_num.append(len(repost_time[-1]))
    repost_time_avg.append(np.mean(repost_time[-1]))
    repost_time_std.append(np.std(repost_time[-1]))

end = time.time()

content_file = "content.csv"
df_content = {'content': content, 'repost_num': repost_num, 'repost_time_avg': repost_time_avg, 'repost_time_std': repost_time_std}

# content单独保存
DataFrame(df_content).to_csv(content_file)

# 各个微博转发数序列单独保存
repost_data_dir = 'repost_data'
if not os.path.exists(repost_data_dir):
    os.makedirs(repost_data_dir)
for i in range(len(repost_time)):
    df_repost = {}
    df_repost['time'] = repost_time[i]
    DataFrame(df_repost).to_csv(os.path.join(repost_data_dir, str(i) + '.csv'))


print("*************PROCESSING COST: %.2fmin**************" % ((end-start)/60) )

describe(content_len, "Content Character length")
describe(repost_num, "Gross Reposter Num")
describe(repost_interval, "Repost Interval")
describe(repost_time_avg, "Avg of repost time")
describe(repost_time_std, "Std of repost time")

start = time.time()

def draw_hist(data_list, start, end, width, x_label, file_name, div=24):
    if div != 1:
        example = [item/div for item in data_list]
    else:
        example = data_list
    hist_range = np.arange(start, int(end + 1), width)
    frequency_each,_,_= plt.hist(example, hist_range, color='red', width=width,alpha=0.7)#alpha设置透明度，0为完全透明
    plt.xlabel(x_label)
    plt.ylabel('count')
    plt.xlim(start,end)
    # plt.plot(hist_range[1:]-(hist_range//2),frequency_each,color='palevioletred')#利用返回值来绘制区间中点连线
    plt.savefig(file_name)
    plt.clf()

for i in range(20):
    draw_hist(data_list=repost_time[i], start=0, end=200, width=10,
                x_label='days', file_name='images/days'+str(i)+'.jpg', div=24)
    draw_hist(data_list=repost_time[i], start=0, end=48, width=1,
                x_label='hours', file_name='images/hours'+str(i)+'.jpg', div=1)

draw_hist(data_list=content_len, start=0, end=max(content_len), width=20,
            x_label='length', file_name='content_len.jpg', div=1)
draw_hist(data_list=repost_num, start=0, end=max(repost_num), width=1000,
            x_label='repost number', file_name='repost_num.jpg', div=1)

end = time.time()
print("*************DRAWING COST: %.2fmin**************" % ((end-start)/60) )

print("Debug")
