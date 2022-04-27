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
            'spwrnn2': self.__SPWRNN,
            'spwrnn_beta': self.__SPWRNN_BETA,
            'spwrnn_wo_l': self.__SPWRNN_WO_L,
            'framing': self.__FRAMING,
        }

        DATA_MAP = {
            'renminribao': self.__RMRB,
        }

        commonArgs = MODEL_MAP[modelName](tune)
        dataArgs = DATA_MAP[dataset](tune)

        self.args = Storage({**commonArgs, **dataArgs})


    def solve_conflict(self):

        if self.modelName == 'spwrnn_beta':
            # model preconfig conflict
            if self.args.use_framing == False:
                self.args.use_predicted_framing = False
                self.args.constant_framing = True
                self.args.framing_loss_weight = 0
            else:        # use framing
                if self.args.use_predicted_framing == True:
                    self.args.constant_framing = True
                    self.args.framing_loss_weight = 0
                else:    # use framing but not predicted from framing model
                    if self.args.constant_framing == True:  # use manual labeled framing
                        self.args.framing_loss_weight = 0

        return

    
    def get_config(self):
        self.solve_conflict()
        return self.args

    def __RMRB(self, tune):

        dataConfig = {
            # 数据载入
            'data_dir': "/home/disk/disk2/lw/covid-19-weibo-processed/renminribao",
            'load_from_temp': True,

            'interval': 1800,        # 计数时间间隔 600, 900, 1200, 1800, 3600s
            'min_repost': 100,      # 最低转发次数
            'max_repost': 10000,    # 最大转发次数
            'observe_time': [0*3600, 1*3600, 2*3600, 3*3600, 6*3600],  # 观察时间长度
            'valid_time': 24*3600,    # 预测时间长度
            'max_seq_len': 256,     # 模型最大长度
            'topic_num': 219,        # 主题的个数

            # 数据集设置
            'validate': 0.1,
            'test': 0.15,
            'batch_size': 32,
            'text_cut': 200,       # 文本截断长度

            # 是否加载framing的预测结果
            'use_predicted_framing': False,
        }

        dataTuneConfig = {
            'data_dir': "/home/disk/disk2/lw/covid-19-weibo-processed/renminribao",
            'load_from_temp': True,

            'interval': 1800,        # 计数时间间隔 600, 900, 1200, 1800, 3600s
            'min_repost': 100,      # 最低转发次数
            'max_repost': 10000,    # 最大转发次数
            'observe_time': [0*3600, 1*3600, 2*3600, 3*3600, 6*3600],  # 观察时间长度
            'valid_time': 24*3600,    # 预测时间长度
            'max_seq_len': 256,     # 模型最大长度
            'topic_num': random.choice([100, 219]),        # 主题的个数[10, 100, 219]

            # 数据集设置
            'validate': 0.1,
            'test': 0.15,
            'batch_size': 32,
            'text_cut': 200,       # 文本截断长度

            # 是否加载framing的预测结果
            'use_predicted_framing': random.choice([False, True]),            
        }

        return dataTuneConfig if tune else dataConfig
        

    def __SPWRNN(self, tune):

        Config = {
            # 标识符
            'name': 'SPWRNN',
            'actual_model': 'SPWRNN',

            # 预训练模型设置
            'language_model': True,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 模型固定参数
            'topic_size': 384,
            'framing_size': 6,
            'time_size': 3,

            # 模型可调参数
            'hidden_size': 64,             # history
            'public_size': 128,             # public vector size
            'language_proj_size': 8,
            'topic_proj_size': 16,
            'medium_features': 8,
            'use_framing': True,
            'initialize_steps': 2,

            # 学习参数设置
            'max_epochs': 100,
            'learning_rate_bert': 0.001,
            'learning_rate_other': 0.002,
            'weight_decay_bert': 0.0001,
            'weight_decay_other': 0.0001,         
            'early_stop': 10,

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'scheduler_mode': 'min',
            'scheduler_patience': 4,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'SPWRNN',
            'actual_model': 'SPWRNN',

            # 预训练模型设置
            'language_model': True,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            'topic_size': 384,
            'framing_size': 6,
            'time_size': 3,
            'use_framing': True,

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'scheduler_mode': 'min',
            'scheduler_patience': 4,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 10,
            'max_epochs': 100,

            # 调参
            'hidden_size': random.choice([32, 64, 128, 256]),
            'public_size': random.choice([32, 64, 128]),
            'language_proj_size': random.choice([8, 16, 32, 64]),
            'topic_proj_size': random.choice([8, 16, 32, 64]),
            'medium_features': random.choice([8, 16, 32, 64]),
            'initialize_steps': random.choice([1, 2, 3]),

            'learning_rate_bert': random.choice([0, 0, 0, 0, 1e-05, 5e-5, 5e-4, 1e-3]),
            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002]),
            'weight_decay_bert': random.choice([0, 0.001, 0.0001]),
            'weight_decay_other': random.choice([0, 0.001, 0.0001]),    
        }

        return TuneConfig if tune else Config


    def __SPWRNN_BETA(self, tune):

        Config = {
            # 标识符
            'name': 'SPWRNN',
            'actual_model': 'SPWRNN_BETA',

            # 预训练模型设置
            'language_model': True,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 模型固定参数
            'topic_size': 384,
            'framing_size': 6,
            'time_size': 3,

            # 模型可调参数
            'hidden_size': 64,             # history
            'public_size': 128,             # public vector size
            'language_proj_size': 8,
            'topic_proj_size': 16,
            'medium_features': 8,
            'use_framing': True,
            'constant_framing': False,
            'initialize_steps': 2,
            'unique_fusion_weights': False,
            
            'framing_loss_weight': 1,

            # 学习参数设置
            'max_epochs': 40,
            'learning_rate_bert': 0.001,
            'learning_rate_other': 0.002,
            'weight_decay_bert': 0.0001,
            'weight_decay_other': 0.0001,         
            'early_stop': 6,

            # 评估设置
            'KeyEval': '3.0h_mse',
            'scheduler_mode': 'min',
            'scheduler_factor': 0.2,
            'scheduler_patience': 3,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'SPWRNN',
            'actual_model': 'SPWRNN_BETA',

            # 预训练模型设置
            'language_model': True,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            'topic_size': 384,
            'framing_size': 6,
            'time_size': 3,
            'use_framing': True,
            'constant_framing': random.choice([False, True]),

            # 评估设置
            'KeyEval': random.choice(['2.0h_mse']),
            'scheduler_mode': 'min',
            'scheduler_factor': random.choice([0.1, 0.25]),
            'scheduler_patience': 3,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 8,
            'max_epochs': 50,

            # 调参
            'hidden_size': random.choice([32, 64, 128, 256]),
            'public_size': random.choice([32, 64, 128]),
            'language_proj_size': random.choice([8, 16, 32, 64]),
            'topic_proj_size': random.choice([8, 16, 32, 64]),
            'medium_features': random.choice([8, 16, 32, 64]),
            'initialize_steps': random.choice([1, 2, 3, 200]),
            'unique_fusion_weights': random.choice([False, True]),
            'framing_loss_weight': random.choice([0.5, 0.8, 1, 1.2, 1.5, 2]),

            'learning_rate_bert': random.choice([0, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
            'learning_rate_other': random.choice([0.001, 0.002, 0.005]),
            'weight_decay_bert': random.choice([0, 0.0001]),
            'weight_decay_other': random.choice([0, 0.0001]),      
        }

        return TuneConfig if tune else Config


    def __SPWRNN_WO_L(self, tune):
        Config = {
            # 标识符
            'name': 'SPWRNN_WO_L',

            # 预训练模型设置
            'language_model': False,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            'time_size': 3,

            # 模型可调参数
            'hidden_size': 64,             # history
            'medium_features': 16,

            # 学习参数设置
            'max_epochs': 100,
            'learning_rate_bert': 0.,
            'learning_rate_other': 0.002,
            'weight_decay_bert': 0.,
            'weight_decay_other': 0.0001,         
            'early_stop': 10,

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'scheduler_mode': 'min',
            'scheduler_patience': 4,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'SPWRNN_WO_L',

            # 预训练模型设置
            'language_model': False,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            'time_size': 3,

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'scheduler_mode': 'min',
            'scheduler_patience': 4,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 10,
            'max_epochs': 100,

            # 调参
            'hidden_size': random.choice([32, 64, 128, 256]),
            'medium_features': random.choice([8, 16, 32, 64]),

            'learning_rate_bert': random.choice([0,]),
            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002, 0.005, 0.01]),
            'weight_decay_bert': random.choice([0,]),
            'weight_decay_other': random.choice([0, 0.0001]),    
        }

        return TuneConfig if tune else Config

    
    def __RNN(self, tune):
        Config = {
            # 标识符
            'name': 'RNN',

            # tokenizer设置
            'language_model': False,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            # 'use_prompt': False,
            'hidden_size': 64,
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
            'scheduler_mode': 'min',
            'scheduler_patience': 4,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'RNN',

            # tokenizer设置
            'language_model': False,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置
            
            # 评估设置
            'KeyEval': '3.0h_Loss',
            'scheduler_mode': 'min',
            'scheduler_patience': 4,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 8,
            'max_epochs': 50,
            'learning_rate_bert': 1e-05,
            'weight_decay_bert': 0., 

            # 调参
            'hidden_size': random.choice([16, 32, 64, 128, 256]), 
            'layer': random.choice([1, 2, 3, 4]), 

            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002, 0.005]),
            'weight_decay_other': random.choice([0, 0.0001]),    
        }

        return TuneConfig if tune else Config


    def __TCN(self, tune):
        Config = {
            # 标识符
            'name': 'TCN',

            # tokenizer设置
            'language_model': False,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

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
            'scheduler_mode': 'min',
            'scheduler_patience': 4,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'TCN',

            # tokenizer设置
            'language_model': False,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 模型设置

            # 评估设置
            'KeyEval': '3.0h_Loss',
            'scheduler_mode': 'min',
            'scheduler_patience': 4,
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

            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002, 0.005]),
            'weight_decay_other': random.choice([0, 0.0001]),    
        }

        return TuneConfig if tune else Config


    def __FRAMING(self, tune):

        Config = {
            # 标识符
            'name': 'FRAMING',

            # 预训练模型设置
            'language_model': True,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',


            # 模型可调参数
            'dropout': 0.1,

            # 学习参数设置
            'max_epochs': 100,
            'learning_rate_bert': 5e-05,
            'learning_rate_other': 0.005,
            'weight_decay_bert': 0.0001,
            'weight_decay_other': 0.0001,         
            'early_stop': 8,

            # 评估设置
            'KeyEval': 'f1_avg',
            'scheduler_mode': 'max',
            'scheduler_patience': 4,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch
        }

        TuneConfig = {     
            # 不变参数
            # 标识符
            'name': 'FRAMING',

            # 预训练模型设置
            'language_model': True,
            'pretrained_model': '/home/disk/disk2/lw/pretrained_model/chinese-roberta-wwm-ext',

            # 评估设置
            'KeyEval': random.choice(['acc_avg', 'recall_avg', 'f1_avg']), 
            'scheduler_mode': 'max',
            'scheduler_patience': 4,
            'eval_step': None,        # eval间隔的step数, None表示1eval/epoch

            # 学习参数设置
            'early_stop': 8,
            'max_epochs': 100,

            # 调参
            'dropout': random.choice([0.1, 0.2, 0.3]), 
            'learning_rate_bert': random.choice([1e-05, 5e-5, 5e-4, 1e-3]),
            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002, 0.005, 0.01]),
            'weight_decay_bert': random.choice([0, 0.0001, 0.001, 0.01]),
            'weight_decay_other': random.choice([0, 0.0001, 0.001, 0.01]),    
        }

        return TuneConfig if tune else Config