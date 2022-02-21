import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import math
import torch.functional as F
from torch.nn.utils import weight_norm

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
        super(SPWRNN, self).__init__()
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
        initial_state = text_representation.unsqueeze(0)
        prediction, (_, _) = self.predict_model(dec_input, (torch.zeros((1, text_representation.shape[0], 1)).to(self.args.device), initial_state))

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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def getModel(modelName):
    MODEL_MAP = {
        'spw': SPW,
        'spwrnn': SPWRNN,
    }

    assert modelName in MODEL_MAP.keys(), 'Not support ' + modelName

    return MODEL_MAP[modelName]

