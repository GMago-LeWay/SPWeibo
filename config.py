import random
from utils import Storage
import warnings


class Config:
    def __init__(self, modelName, tune=False) -> None:
        self.modelName = modelName

        MODEL_MAP = {
            'spw': self.__SPW,
        }

        commonArgs = MODEL_MAP[modelName](tune)
        self.args = Storage(dict(commonArgs))


    def solve_conflict(self):
        return

    
    def get_config(self):
        self.solve_conflict()
        return self.args

    def __SPW(self, tune):

        Config = {
            # 标识符
            'name': 'SPW',

            # 预训练模型设置
            'pretrained_model': 'pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置

            # 数据集设置
            'seq_dim': 24,
            'validate': 0.15,
            'test': 0.1,
            'batch_size': 32,
            'text_cut': 200,       # 文本截断长度

            # 学习参数设置
            'max_epochs': 50,
            'learning_rate_bert': 1e-05,
            'learning_rate_other': 0.002,
            'weight_decay_bert': 0.0001,
            'weight_decay_other': 0,         
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

            # 数据集设置
            'seq_dim': 24,
            'validate': 0.15,
            'test': 0.1,
            'batch_size': 32,
            'text_cut': 200,       # 文本截断长度

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

