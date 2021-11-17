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
            'validate': 0.2,
            'batch_size': 32,

            # 学习参数设置
            'max_epochs': 20,
            'learning_rate_bert': 1e-05,
            'learning_rate_other': 0.002,
            'weight_decay_bert': 0.0001,
            'weight_decay_other': 0,         
            'evaluation_steps': -1, # evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
            'early_stop': 20,

            'text_cut': 200,       # 文本截断长度

            # 评估设置
            'KeyEval': 'Loss',
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
        }

        return TuneConfig if tune else Config

