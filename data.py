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
            if repost_nums[i] < self.config.min_repost:
                continue
            else:
                legal_weibo_idx.append(i)
        legal_weibo_time_series = []
        legal_weibo = []
        for idx in legal_weibo_idx:
            legal_weibo.append(texts[legal_weibo_idx])

            csv_file = os.path.join(repost_dir, f"{idx}.csv")
            repost_time = pd.read_csv(csv_file).tolist()
            repost_time_series = get_time_series(repost_time, 0, self.config.valid_time, self.config.interval)
            legal_weibo_time_series.append(repost_time_series)

        return legal_weibo, np.array(legal_weibo_time_series), legal_weibo_idx
            

    def get_data(self):
        texts, sequences, _ = self.get_series_and_texts()

        assert len(sequences) == len(texts), "length is not consisitent."

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
        return [[texts[i], log_function(add(sequences[i]))] for i in range(len(texts))]


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
            decoder_input = torch.zeros((batch_size, 1, 1))
            decoder_inputs = torch.cat([decoder_input, labels[:, :-1].unsqueeze(-1)], dim=1)
            texts = self.tokenizer(texts, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
            return {'texts': texts, 'labels': labels, 'dec_inputs': decoder_inputs}
        return collate_fn


def getData(modelName):
    MODEL_MAP = {
        'spw': WeiboData,
        'spwrnn': WeiboDataTimeSeries,
    }

    assert modelName in MODEL_MAP.keys(), 'Not support ' + modelName

    return MODEL_MAP[modelName]


if __name__ == "__main__":
    data = WeiboData(config=Config('spwrnn', 'reminribao').get_config())
    train_loader, val_loader = data.get_train_val_dataset()
    for batch in train_loader:
        print(batch)
        exit