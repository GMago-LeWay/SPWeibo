"""
Process data from raw json or txt.
"""
import time
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import os


def get_absolute_hours(str):
    # 应对没有秒数的脏数据
    base_time = time.mktime(time.strptime('2020/01/01 00:00:00', '%Y/%m/%d %H:%M:%S'))
    if len(str.split(':')) == 2:
        str += ':00'
    if str[4] == '-':
        return time.mktime(time.strptime(str, '%Y-%m-%d %H:%M:%S'))/3600 - base_time/3600
    else:
        return time.mktime(time.strptime(str, '%Y/%m/%d %H:%M:%S'))/3600 - base_time/3600

def get_absolute_seconds(str):
    # 应对没有秒数的脏数据
    base_time = time.mktime(time.strptime('2020/01/01 00:00:00', '%Y/%m/%d %H:%M:%S'))
    if len(str.split(':')) == 2:
        str += ':00'
    if str[4] == '-':
        return time.mktime(time.strptime(str, '%Y-%m-%d %H:%M:%S')) - base_time
    else:
        return time.mktime(time.strptime(str, '%Y/%m/%d %H:%M:%S')) - base_time


class Ori2Json:
    def __init__(self, dir) -> None:
        self.dir_origin = dir

    ### To be finished.

class Process:
    def __init__(self, dataset_name) -> None:
        dir_map = {
            "renminribao": "/home/disk/disk2/lw/covid-19-weibo/renminribao.json"
        }
        save_map = {
            "renminribao": "/home/disk/disk2/lw/covid-19-weibo-processed/renminribao"
        }

        self.dir = dir_map[dataset_name]
        self.save_dir = save_map[dataset_name]
        self.json = None

    def load_json(self):
        with open(self.dir, 'r') as f:
            self.json = json.load(f)

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
    
    def process_items(self):
        if self.json == None:
            self.load_json()

        content_len = []            # content text lengths
        content = []                # content texts
        repost_num = []             # repost numbers
        repost_interval = []        # time interval when repost actions exist
        repost_time_avg = []        # avg repost time
        repost_time_std = []        # std repost time

        repost_time = []            # repost time lists of all weibo
        publish_time = []           # publish time = min repost time, seconds
        raw_publish_time = []               # the original time string

        for item in tqdm(self.json):
            content_len.append(len(item['original_content']))
            content.append(item['original_content'])

            temp_time_list = []
            min_time = 9999999
            max_time = 0
            min_timestamp = ''
            for repost in item['reposts']:
                item_repost_time = get_absolute_seconds(repost['repost_timestamp'])
                temp_time_list.append(item_repost_time)
                if item_repost_time < min_time:
                    min_time = item_repost_time
                    min_timestamp = repost['repost_timestamp']
                max_time = max(max_time, item_repost_time)
            
            # assert time scale
            assert max_time < 9999999
            assert min_time > 0

            repost_time.append([value-min_time for value in temp_time_list])    
            repost_interval.append(max_time-min_time)
            repost_num.append(len(repost_time[-1]))
            repost_time_avg.append(np.mean(repost_time[-1]))
            repost_time_std.append(np.std(repost_time[-1]))
            publish_time.append(min_time)
            raw_publish_time.append(min_timestamp)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        content_file = os.path.join(self.save_dir, "content.csv")
        df_content = {'content': content, 'repost_num': repost_num, 'publish_time': publish_time, 'raw_publish_time': raw_publish_time, 'repost_time_avg': repost_time_avg, 'repost_time_std': repost_time_std}

        # content单独保存
        pd.DataFrame(df_content).to_csv(content_file, index=None)

        # 各个微博转发数序列单独保存
        repost_data_dir = os.path.join(self.save_dir, 'repost_data')
        if not os.path.exists(repost_data_dir):
            os.makedirs(repost_data_dir)
        for i in range(len(repost_time)):
            df_repost = {}
            df_repost['time'] = sorted(repost_time[i])
            pd.DataFrame(df_repost).to_csv(os.path.join(repost_data_dir, str(i) + '.csv'), index=None)


if __name__ == "__main__":
    processor = Process("renminribao")
    processor.process_items()
