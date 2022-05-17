import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from RepostLSTM import RepostLSTM, FusionNet2


class SPWRNN_WO_L(torch.nn.Module):
    def __init__(self, config, args) -> None:
        super(SPWRNN_WO_L, self).__init__()
        self.config = config
        self.args = args

        # series related model
        self.series_model = torch.nn.LSTM(
            input_size=1,
            hidden_size=self.config.hidden_size,
            batch_first=True,
        )
        self.series_vec_func = nn.ReLU()

        # time related model
        self.abs_time_fusion = FusionNet2(self.config.hidden_size, self.config.time_size, self.config.medium_features, 1)

        # final weighted results
        self.one_weight = nn.Linear(1, 1)
        # self.weight_norm = nn.Softmax(dim=0)

    def get_features(self, text, dec_input, others, mode='TRAIN'):
        """
        Get features from raw input.
        series_features(seq), language_features, topic_features, abs_time_features(seq), framing_features
        """
        seq_len = dec_input.shape[1]

        # extract series features
        history_features, (_, _) = self.series_model(dec_input)

        # abs time features
        abs_time = others['abs_time'][:, :seq_len]

        return history_features, None, None, abs_time, None


    def forward(self, text, dec_input, others, eval=False):
        """
        Train forward.
        Return all prediction. [batch_size, seq_len, 1]
        """
        seq_len = dec_input.shape[1]
        history_seq, language, topic, abs_time_seq, framing = self.get_features(text, dec_input, others)

        # feature fusion for every timestamp and prediction
        prediction = []
        for i in range(seq_len):
            # time-series result
            result_a_s = self.abs_time_fusion(history_seq[:, i, :], abs_time_seq[:, i, :])
            prediction_i = self.one_weight(result_a_s)
            prediction.append(prediction_i)             
        
        prediction = torch.cat(prediction, dim=1)
        if eval:
            return F.relu(prediction.unsqueeze(-1)) + dec_input
        else:
            return prediction.unsqueeze(-1) + dec_input


class RNN(torch.nn.Module):
    def __init__(self, config, args) -> None:
        super(RNN, self).__init__()
        self.config = config
        self.args = args
        self.rnn_model = torch.nn.LSTM(
            input_size=1,
            hidden_size=self.config.hidden_size,
            batch_first=True,
            num_layers=self.config.layer,
        )
        self.predict_fc = torch.nn.Linear(self.config.hidden_size, 1)

    def forward(self, text, dec_input, others=None, eval=False):
        representation, _ = self.rnn_model(dec_input)
        batch_size, time_len, _ = representation.shape
        representation = representation.reshape(batch_size*time_len, -1)
        predict = self.predict_fc(representation)

        if eval:
            return F.relu(predict.view(batch_size, time_len, 1)) + dec_input
        else:
            return predict.view(batch_size, time_len, 1) + dec_input


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

        self.conv2 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
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


class TCN(torch.nn.Module):
    def __init__(self, config, args) -> None:
        super(TCN, self).__init__()
        self.config = config
        self.args = args
        self.tcn_model = TemporalConvNet(
            num_inputs=1,
            num_channels=[1] * config.layer,
            kernel_size=config.kernel,
            dropout=config.dropout,
        )

    def forward(self, text, dec_input, others=None, eval=False):
        x = dec_input.permute(0, 2, 1)
        out = self.tcn_model(x)
        out = out.permute(0, 2, 1)
        return out


def getModel(modelName):
    MODEL_MAP = {
        'rnn': RNN,
        'tcn': TCN,
        'spwrnn': RepostLSTM,
        'spwrnn_beta': RepostLSTM,
        'spwrnn_wo_l': SPWRNN_WO_L,
    }

    assert modelName in MODEL_MAP.keys(), 'Not support ' + modelName

    return MODEL_MAP[modelName]

