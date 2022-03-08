import random
from utils import Storage
import warnings


class Config:
    def __init__(self, modelName, dataset, tune=False) -> None:
        self.modelName = modelName

        MODEL_MAP = {
            'rnn': self.__RNN,
            'tcn': self.__TCN,
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
            'load_from_temp': True,

            'interval': 1800,        # 计数时间间隔 600, 900, 1200, 1800, 3600s
            'min_repost': 100,      # 最低转发次数
            'max_repost': 10000,    # 最大转发次数
            'observe_time': [0, 1*3600, 2*3600, 3*3600, 4*3600, 5*3600, 6*3600],  # 观察时间长度
            'valid_time': 24*3600,    # 预测时间长度
            'max_seq_len': 256,     # 模型最大长度

            # 数据集设置
            'validate': 0.15,
            'test': 0.1,
            'batch_size': 32,
            'text_cut': 200,       # 文本截断长度
        }

        return dataConfig
        

    def __SPWRNN(self, tune):

        Config = {
            # 标识符
            'name': 'SPWRNN',

            # 预训练模型设置
            'pretrained_model': 'pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            # 'use_prompt': False,
            'hidden_size': 128,
            'public_size': 128,
            'topic_size': 384,
            'framing_size': 6,
            'use_framing': False,

            # 学习参数设置
            'max_epochs': 50,
            'learning_rate_bert': 1e-05,
            'learning_rate_other': 0.002,
            'weight_decay_bert': 0.,
            'weight_decay_other': 0.0001,         
            'early_stop': 8,

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'SPWRNN',

            # 预训练模型设置
            'pretrained_model': 'pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            'topic_size': 256,
            'framing_size': 6,

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 8,
            'max_epochs': 50,

            # 调参
            'hidden_size': random.choice([64, 128, 256]),
            'public_size': random.choice([64, 128, 256]),

            'learning_rate_bert': random.choice([1e-05, 5e-5, 5e-4, 1e-3]),
            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002]),
            'weight_decay_bert': random.choice([0, 0.0001]),
            'weight_decay_other': random.choice([0, 0.0001]),    
        }

        return TuneConfig if tune else Config

    
    def __RNN(self, tune):
        Config = {
            # 标识符
            'name': 'RNN',

            # 模型设置
            # 'use_prompt': False,
            'hidden_size': 128,
            'layer': 1,

            # 学习参数设置
            'max_epochs': 50,
            'learning_rate_bert': 1e-05,
            'learning_rate_other': 0.002,
            'weight_decay_bert': 0.,
            'weight_decay_other': 0.0001,         
            'early_stop': 8,

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'RNN',

            # 模型设置

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 8,
            'max_epochs': 50,
            'learning_rate_bert': 1e-05,
            'weight_decay_bert': 0., 

            # 调参
            'hidden_size': random.choice([32, 64, 128, 256]), 
            'layer': random.choice([1, 2, 3, 4]), 

            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002]),
            'weight_decay_other': random.choice([0, 0.0001]),    
        }

        return TuneConfig if tune else Config


    def __TCN(self, tune):
        Config = {
            # 标识符
            'name': 'TCN',

            # 模型设置
            # 'use_prompt': False,
            'kernel': 5,
            'layer': 4,
            'dropout': 0.2,

            # 学习参数设置
            'max_epochs': 50,
            'learning_rate_bert': 1e-05,
            'learning_rate_other': 0.002,
            'weight_decay_bert': 0.,
            'weight_decay_other': 0.0001,         
            'early_stop': 8,

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'TCN',

            # 模型设置

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 8,
            'max_epochs': 50,
            'learning_rate_bert': 1e-05,
            'weight_decay_bert': 0., 

            # 调参
            'kernel': random.choice([3, 4, 5, 6, 7, 8, 9, 10]), 
            'layer': random.choice([3, 4, 5, 6, 7, 8, 9, 10]), 

            'dropout': random.choice([0.1, 0.2, 0.3]),  

            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002]),
            'weight_decay_other': random.choice([0, 0.0001]),    
        }

        return TuneConfig if tune else Config
