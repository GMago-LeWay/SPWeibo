import random
from utils import Storage
import warnings


class Config:
    def __init__(self, modelName, dataset, tune=False) -> None:
        self.modelName = modelName

        MODEL_MAP = {
            'spw': self.__SPW,
            'spwrnn': self.__SPWRNN,
        }

        DATA_MAP = {
            'renminribao': self.__RMRB,
        }

        commonArgs = MODEL_MAP[modelName](tune)
        dataArgs = DATA_MAP[dataset]()

        self.args = Storage({**commonArgs, **dataArgs})


    def solve_conflict(self):
        return

    
    def get_config(self):
        self.solve_conflict()
        return self.args

    def __RMRB(self):

        dataConfig = {
            # 数据载入
            'data_dir': "/home/disk/disk2/lw/covid-19-weibo-processed/renminribao",

            'interval': 600,        # 计数时间间隔 600, 900, 1200, 1800, 3600s
            'min_repost': 100,      # 最低转发次数
            'observe_seq': [0, 1*3600, 2*3600, 3*3600, 4*3600, 5*3600, 6*3600],  # 观察时间长度
            'test_point': 24*3600,    # 预测时间长度
            'max_seq_len': 256,     # 模型最大长度

            # 数据集设置
            'validate': 0.15,
            'test': 0.1,
            'batch_size': 32,
            'text_cut': 200,       # 文本截断长度
        }

        return dataConfig

    def __SPW(self, tune):

        Config = {
            # 标识符
            'name': 'SPW',

            # 预训练模型设置
            'pretrained_model': 'pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            'use_prompt': False,

            # 学习参数设置
            'max_epochs': 50,
            'learning_rate_bert': 5e-05,
            'learning_rate_other': 0.0005,
            'weight_decay_bert': 0.,
            'weight_decay_other': 0.0001,         
            'early_stop': 8,

            # 评估设置
            'KeyEval': 'Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'SPW',

            # 预训练模型设置
            'pretrained_model': 'pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置

            # 评估设置
            'KeyEval': 'Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 8,
            'max_epochs': 50,

            # 调参
            'learning_rate_bert': random.choice([1e-05, 5e-5, 5e-4, 1e-3]),
            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002]),
            'weight_decay_bert': random.choice([0, 0.0001]),
            'weight_decay_other': random.choice([0, 0.0001]),    
        }

        return TuneConfig if tune else Config


    def __SPWRNN(self, tune):

        Config = {
            # 标识符
            'name': 'SPW',

            # 预训练模型设置
            'pretrained_model': 'pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            'use_prompt': False,


            # 学习参数设置
            'max_epochs': 50,
            'learning_rate_bert': 1e-05,
            'learning_rate_other': 0.002,
            'weight_decay_bert': 0.,
            'weight_decay_other': 0.0001,         
            'early_stop': 8,

            # 评估设置
            'KeyEval': 'Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'SPW',

            # 预训练模型设置
            'pretrained_model': 'pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置

            # 评估设置
            'KeyEval': 'Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 8,
            'max_epochs': 50,

            # 调参
            'learning_rate_bert': random.choice([1e-05, 5e-5, 5e-4, 1e-3]),
            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002]),
            'weight_decay_bert': random.choice([0, 0.0001]),
            'weight_decay_other': random.choice([0, 0.0001]),    
        }

        return TuneConfig if tune else Config
