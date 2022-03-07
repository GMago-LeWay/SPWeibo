import logging
import random
import urllib
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import os
import pickle
import time
from tqdm import tqdm

from config import Config
from utils import getTime


def get_time_series(data_list, start, end, width, lambda_expression=None):
    """
    Get histogram data in [start, end] (data distribution) of data_list. Note that width > 1.
    """
    if lambda_expression:
        example = [lambda_expression(item) for item in data_list]
    else:
        example = data_list
    hist_range = np.arange(start, int(end + 1), width)
    frequency_each, _, _= plt.hist(example, hist_range)
    plt.clf()
    return frequency_each


class WeiboData(Dataset):
    def __init__(self, args, config):
        """
        Warning: To be removed.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        self.config = config
        self.args = args

    def get_data(self):
        sequence = np.load('data/%d.npy' % self.config.seq_dim)
        content_csv = pd.read_csv('data/%d.csv' % self.config.seq_dim)
        texts = content_csv['content'].to_list()

        assert len(sequence) == len(texts), "length is not consisitent."

        # TODO: Label Prompt
        labels = []
        pattern = '[SEP]<T>[SEP]'
        def add_label_prompt(string, label):
            prompt = pattern.replace('<T>', label)
            return prompt + string

        def add(array):
            new_array = np.zeros_like(array)
            for i in range(len(array)):
                new_array[i] = np.sum(array[:i+1])
            return new_array
        
        return [[texts[i], add(sequence[i])] for i in range(len(texts))]

    def get_train_val_dataloader(self):
        gross_data = self.get_data()
        test_num = int(self.config.test * len(gross_data))
        val_num = int(self.config.validate * len(gross_data))
        test = gross_data[-test_num:]
        # select data
        selected_data = gross_data[:-test_num]
        train, val = random_split(selected_data, [len(selected_data)-val_num, val_num])
        logging.info(getTime() + 'Total Train samples: %d, Total Valid samples: %d, Total Test samples: %d' % (len(train), len(val), len(test)))
        return DataLoader(train, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(), drop_last=False), \
                DataLoader(val, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(), drop_last=False), \
                DataLoader(test, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(), drop_last=False)


    def get_collate_fn(self):
        def collate_fn(batch):
            batch_size = len(batch)
            texts = [batch[i][0] for i in range(batch_size)]
            labels = torch.log(torch.FloatTensor([batch[i][1] + 1. for i in range(batch_size)]))
            texts = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
            return {'texts': texts, 'labels': labels}
        return collate_fn


class WeiboDataTimeSeries(Dataset):
    def __init__(self, args, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        self.config = config
        self.args = args

    def get_series_and_texts(self):
        """
        Get time series and texts from filtered raw data. 
        """
        repost_dir = os.path.join(self.config.data_dir, "repost_data")
        content_df = pd.read_csv(os.path.join(self.config.data_dir, "content.csv"))
        texts = content_df['content'].to_list()
        repost_nums = content_df['repost_num'].to_list()
        legal_weibo_idx = []
        for i in range(len(repost_nums)):
            if repost_nums[i] < self.config.min_repost or repost_nums[i] > self.config.max_repost:
                continue
            else:
                legal_weibo_idx.append(i)
        legal_weibo_time_series = []
        legal_weibo = []
        for idx in tqdm(legal_weibo_idx):
            legal_weibo.append(texts[idx])

            csv_file = os.path.join(repost_dir, f"{idx}.csv")
            repost_time = pd.read_csv(csv_file)["time"].tolist()
            assert repost_nums[idx] == len(repost_time), "uncorrect weibo."
            repost_time_series = get_time_series(repost_time, 0, self.config.valid_time, self.config.interval)
            legal_weibo_time_series.append(repost_time_series)

        return legal_weibo, np.array(legal_weibo_time_series), legal_weibo_idx


    def get_framing(self, legal_weibo_idx):
        content_framing_file = os.path.join(self.config.data_dir, "content_with_framing.csv")
        df = pd.read_csv(content_framing_file)
        ids = df["id"].tolist()
        keys = ['Economic consequences', 'Human interest',
                'Morality/Religion', 'Attribution of Responsibility', 'Fear/Scaremongering', 'Hope',]
        result = np.array([df[key].tolist() for key in keys]).T
        legal_weibo_framing = []
        for idx in legal_weibo_idx:
            legal_weibo_framing.append(result[ids.index(idx)])

        return legal_weibo_framing


    def get_topics_and_embedding(self, legal_weibo_idx):
        topic_file = os.path.join(self.config.data_dir, "topic.pkl")
        content_topic_file = os.path.join(self.config.data_dir, "content_topic.csv")

        # Load topic class and embeddings, from class -1 to last topic
        with open(topic_file, 'rb') as f:
            topics = pickle.load(f, encoding='utf-8')

        # Load first topic for every weibo.
        df = pd.read_csv(content_topic_file)
        weibo_topic0 = df["topic0"].tolist()
        legal_weibo_topic0 = [weibo_topic0[idx] for idx in legal_weibo_idx]
        legal_weibo_topic0_embeddings = [topics[t]["embedding"] for t in legal_weibo_topic0]

        return legal_weibo_topic0, legal_weibo_topic0_embeddings

    
    def get_absolute_daytime(self, legal_weibo_idx):
        content_df = pd.read_csv(os.path.join(self.config.data_dir, "content.csv"))
        seconds = content_df['publish_time'].to_list()
        legal_weibo_seconds = [seconds[idx] for idx in legal_weibo_idx]
        daytime = []
        base_time = time.mktime(time.strptime('2020/01/01 00:00:00', '%Y/%m/%d %H:%M:%S'))
        for second in legal_weibo_seconds:
            time_struct = time.localtime(second + base_time)
            hour = time_struct.tm_hour
            minute = time_struct.tm_min
            sec = time_struct.tm_sec
            daytime.append(hour*60*60 + minute*60 + sec)
        
        return daytime


    def get_time_embedding_series(self, absolute_daytime):
        """
        return relative_time_embeddings, absolute_time_embeddings from 0 to 24th hour.
        """
        interval = self.config.interval
        relative_time = list(0, 24*60*60 + 1, interval)
        absolute_time = [relative_timestamp + absolute_daytime for relative_timestamp in relative_time]
        absolute_time = [time % (24*60*60) for time in absolute_time]

        def get_embedding(time_sec):
            hour = time_sec // 3600
            minute = (time_sec % 3600) // 60
            second = time_sec - 3600 * hour - 60 * minute
            return [hour/12 - 1, minute/30 - 1, second/30 - 1]

        relative_time_embeddings = [get_embedding(time_sec) for time_sec in relative_time]
        absolute_time_embeddings = [get_embedding(time_sec) for time_sec in absolute_time]

        return [relative_time_embeddings, absolute_time_embeddings]


    def get_data(self):

        print("Get weibo information...")
        texts, sequences, idxs = self.get_series_and_texts()
        
        print("Get weibo topics information...")
        topics, topic_embeddings = self.get_topics_and_embedding(idxs)

        print("Get weibo framings information...")
        framings = self.get_framing(idxs)

        print("Get weibo time information...")
        daytimes = self.get_absolute_daytime(idxs)

        assert len(sequences) == len(texts) == len(topic_embeddings) == len(framings) == len(daytimes), "length is not consisitent."

        # TODO: Label Prompt
        labels = []
        pattern = '[SEP]<T>[SEP]'
        def add_label_prompt(string, label):
            prompt = pattern.replace('<T>', label)
            return prompt + string

        def add(array):
            new_array = np.zeros_like(array)
            for i in range(len(array)):
                new_array[i] = np.sum(array[:i+1])
            return new_array
        
        def log_function(array):
            return np.log(array + 1.)
        
        # added time series
        # function: log(num + 1)
        return [[texts[i], log_function(add(sequences[i])), daytimes[i], topic_embeddings[i], framings[i]] for i in range(len(texts))]


    def get_train_val_dataloader(self):
        gross_data = self.get_data()
        test_num = int(self.config.test * len(gross_data))
        val_num = int(self.config.validate * len(gross_data))
        test = gross_data[-test_num:]
        # select data
        selected_data = gross_data[:-test_num]
        train, val = random_split(selected_data, [len(selected_data)-val_num, val_num])
        logging.info(getTime() + 'Total Train samples: %d, Total Valid samples: %d, Total Test samples: %d' % (len(train), len(val), len(test)))
        return DataLoader(train, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(), drop_last=False), \
                DataLoader(val, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(), drop_last=False), \
                DataLoader(test, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(), drop_last=False)


    def get_collate_fn(self):
        def collate_fn(batch):
            batch_size = len(batch)
            texts = [batch[i][0] for i in range(batch_size)]
            labels = torch.FloatTensor([batch[i][1] for i in range(batch_size)])
            time_embeds = torch.FloatTensor([self.get_time_embedding_series(batch[i][2]) for i in range(batch_size)])
            topic_embeds = torch.FloatTensor([batch[i][3] for i in range(batch_size)])
            decoder_input = torch.zeros((batch_size, 1, 1))
            decoder_inputs = torch.cat([decoder_input, labels[:, :-1].unsqueeze(-1)], dim=1)
            texts = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
            return {'texts': texts, 'labels': labels, 'dec_inputs': decoder_inputs,
                    'others': {'rlt_time': time_embeds[:, 0, :, :], 'abs_time': time_embeds[:, 0, :, :], 'topics': topic_embeds, 'framing': None}}
        return collate_fn


def getData(modelName):
    MODEL_MAP = {
        'spw': WeiboData,
        'rnn': WeiboDataTimeSeries,
        'tcn': WeiboDataTimeSeries,
        'spwrnn': WeiboDataTimeSeries,
    }

    assert modelName in MODEL_MAP.keys(), 'Not support ' + modelName

    return MODEL_MAP[modelName]


if __name__ == "__main__":
    dataclass = WeiboDataTimeSeries(args=None, config=Config('spwrnn', 'renminribao').get_config())
    data = dataclass.get_data()
    for data_item in data:
        print(data_item)