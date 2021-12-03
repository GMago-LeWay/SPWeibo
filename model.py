import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import math
import torch.functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, pretrained_model='bert-base-multilingual-cased') -> None:
        super(BaseModel, self).__init__()

        self.language_model = AutoModel.from_pretrained(pretrained_model)

    def forward(self, input_ids, input_mask):
        hidden_states = self.language_model(input_ids)['last_hidden_state']
        words_num = torch.sum(input_mask, dim=-1).unsqueeze(-1)

        # normalize
        hidden_states_sum = torch.sum(hidden_states * input_mask.unsqueeze(-1), dim=1)
        representation = torch.div(hidden_states_sum, words_num)

        return representation


class SPWRNN(torch.nn.Module):
    def __init__(self, config, args) -> None:
        super(SPW, self).__init__()
        self.config = config
        self.args = args
        self.language_model = BaseModel(self.config.pretrained_model)
        self.language_model_config = AutoConfig.from_pretrained(self.config.pretrained_model)

        self.predict_model = torch.nn.LSTM(
            input_size=1,
            hidden_size=self.language_model_config.hidden_size,
            batch_first=True,
            proj_size=1,
        )

    def forward(self, text, dec_input):
        text_representation = self.language_model(text['input_ids'], text['attention_mask'])
        initial_hidden_state = text_representation.unsqueeze(1)
        prediction, (_, _) = self.predict_model(dec_input, (initial_hidden_state, torch.zeros_like(initial_hidden_state)))

        return prediction


class SPW(torch.nn.Module):
    def __init__(self, config, args) -> None:
        super(SPW, self).__init__()
        self.config = config
        self.args = args
        self.language_model = BaseModel(self.config.pretrained_model)
        self.language_model_config = AutoConfig.from_pretrained(self.config.pretrained_model)

        self.predict_model = torch.nn.Linear(self.language_model_config.hidden_size, self.config.seq_dim)

    def forward(self, text):
        text_representation = self.language_model(text['input_ids'], text['attention_mask'])
        prediction = self.predict_model(text_representation)

        return prediction


def getModel(modelName):
    MODEL_MAP = {
        'spw': SPW,
        'spwrnn': SPWRNN,
    }

    assert modelName in MODEL_MAP.keys(), 'Not support ' + modelName

    return MODEL_MAP[modelName]

