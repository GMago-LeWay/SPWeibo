import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import math
import torch.functional as F
from torch.nn.utils import weight_norm

class FusionNet(torch.nn.Module):
    def __init__(self, main_features, auxiliary_features, medium_features, target_features) -> None:
        super(FusionNet, self).__init__()
        
        self.fc1 = torch.nn.Linear(main_features+auxiliary_features, medium_features)
        self.func1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(medium_features, medium_features)
        self.func2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(medium_features, target_features)
        self.func3 = torch.nn.LeakyReLU()
    
    def forward(self, main, auxiliary):
        features = torch.cat([main, auxiliary], dim=1)
        features = self.fc1(features)
        features = self.func1(features)
        medium_features = self.fc2(features)
        medium_features = self.func2(medium_features) + features
        result = self.fc3(medium_features)
        result = self.func3(result)
        return result


class SPWRNN(torch.nn.Module):
    def __init__(self, config, args) -> None:
        super(SPWRNN, self).__init__()
        self.config = config
        self.args = args

        # language related model
        self.language_model = AutoModel.from_pretrained(self.config.pretrained_model)
        self.language_model_config = AutoConfig.from_pretrained(self.config.pretrained_model)
        self.language_hidden_size = self.language_model_config.hidden_size
        self.language_fusion = FusionNet(self.config.hidden_size, self.config.language_proj_size, self.config.medium_features, 1)
        # sentence vector extract
        self.public_vector = nn.Parameter(torch.randn((self.config.public_size)))
        self.word_key = nn.Linear(self.language_hidden_size, self.config.public_size)
        self.softmax = nn.Softmax(dim=1)
        self.language_proj = nn.Linear(self.language_hidden_size, self.config.language_proj_size)
        self.language_vec_func = nn.ReLU()

        # series related model
        self.series_model = torch.nn.LSTM(
            input_size=1,
            hidden_size=self.config.hidden_size,
            batch_first=True,
        )
        self.series_vec_func = nn.ReLU()

        # topics related model
        self.topic_proj = nn.Linear(self.config.topic_size, self.config.topic_proj_size)
        self.topic_vec_func = nn.ReLU()
        self.topic_fusion = FusionNet(self.config.hidden_size, self.config.topic_proj_size, self.config.medium_features, 1)

        # framing related model
        self.framing_fusion = FusionNet(self.config.hidden_size, self.config.framing_size, self.config.medium_features, 1)

        # time related model
        self.abs_time_fusion = FusionNet(self.config.hidden_size, self.config.time_size, self.config.medium_features, 1)

        # grand fusion related model
        auxiliary_dim = self.config.language_proj_size + self.config.topic_proj_size + self.config.time_size
        if self.config.use_framing:
            auxiliary_dim += self.config.framing_size 
        self.grand_fusion = FusionNet(self.config.hidden_size, auxiliary_dim, self.config.medium_features, 1)

        # final weighted results
        # self.weighed_sum = nn.Linear(5, 1) if self.config.use_framing else nn.Linear(4, 1)
        in_dim = 5 if self.config.use_framing else 4

        self.sum_weight = [nn.Linear(in_dim, 1).to(self.args.device) for i in range(self.config.initialize_steps)]
        self.one_weight = nn.Linear(1, 1)
        # self.weight_norm = nn.Softmax(dim=0)

        # eval cache (only available when do_test)
        self.cache_language = None
        self.cache_topic = None
    
    def clear_eval_cache(self):
        """
        After infering one weibo, the cache of features must be removed.
        """
        self.cache_language = None
        self.cache_topic = None        
    
    def check_cache(self):
        return self.cache_language is not None

    def write_cache(self, language_features, topic_features):
        self.cache_language, self.cache_topic = language_features, topic_features

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

        if mode == 'VAL':
            # validating and there's cached data
            if self.check_cache():
                text_features, topic_features = self.cache_language, self.cache_topic
                return history_features, text_features, topic_features, abs_time, others['framing']

        # extract language features
        text_representation = self.language_model(text['input_ids'], text['attention_mask'])['last_hidden_state']
        batch_size, text_len, text_dim = text_representation.shape
        
        # get interest semantics (sentence vector)
        text_representation_ravel = text_representation.view(batch_size*text_len, -1)
        key = self.word_key(text_representation_ravel)
        scores = self.softmax(self.public_vector.unsqueeze(0) * key).view(batch_size, text_len, -1)
        scores = scores.sum(dim=-1) * text['attention_mask']
        interest_text = (text_representation * scores.unsqueeze(-1)).sum(dim=1)
        # proj of text
        text_features = self.language_vec_func(self.language_proj(interest_text))

        # proj of topics
        topic_features = self.topic_vec_func(self.topic_proj(others['topics']))

        if mode == 'VAL':
            self.write_cache(text_features, topic_features)

        return history_features, text_features, topic_features, abs_time, others['framing']


    def forward(self, text, dec_input, others):
        """
        Train forward.
        Return all prediction. [batch_size, seq_len, 1]
        """
        seq_len = dec_input.shape[1]
        history_seq, language, topic, abs_time_seq, framing = self.get_features(text, dec_input, others)

        # feature fusion for every timestamp and prediction
        prediction = []
        for i in range(seq_len):
            if i < self.config.initialize_steps:
                # text-series result
                result_l_s = self.language_fusion(history_seq[:, i, :], language)
                # time-series result
                result_a_s = self.abs_time_fusion(history_seq[:, i, :], abs_time_seq[:, i, :])
                # topic-series result
                result_t_s = self.topic_fusion(history_seq[:, i, :], topic)
                # framing-series result
                result_f_s = self.framing_fusion(history_seq[:, i, :], framing)
                # grand fusion result
                if self.config.use_framing:
                    features = [language, abs_time_seq[:, i, :], topic, framing]
                else:
                    features = [language, abs_time_seq[:, i, :], topic]
                fused_features = torch.cat(features, dim=1)
                result_all = self.grand_fusion(history_seq[:, i, :], fused_features)
                
                # all results
                if self.config.use_framing:
                    prediction_i = self.sum_weight[i](torch.cat([result_l_s, result_a_s, result_t_s, result_f_s, result_all], dim=1))
                else:
                    prediction_i = self.sum_weight[i](torch.cat([result_l_s, result_a_s, result_t_s, result_all], dim=1))

                prediction.append(prediction_i)
            else:
                # time-series result
                result_a_s = self.abs_time_fusion(history_seq[:, i, :], abs_time_seq[:, i, :])
                prediction_i = self.one_weight(result_a_s)
                prediction.append(prediction_i)             
        
        prediction = torch.cat(prediction, dim=1)
        return prediction.unsqueeze(-1)


    def predict_next(self, text, dec_input, others):
        """
        Train forward.
        Return last position (next timestamp) prediction. [batch_size, 1, 1]
        """
        seq_len = dec_input.shape[1]
        history_seq, language, topic, abs_time_seq, framing = self.get_features(text, dec_input, others, 'VAL')

        # feature fusion for last timestamp and prediction
        if seq_len - 1 < self.config.initialize_steps:
            # text-series result
            result_l_s = self.language_fusion(history_seq[:, -1, :], language)
            # time-series result
            result_a_s = self.abs_time_fusion(history_seq[:, -1, :], abs_time_seq[:, -1, :])
            # topic-series result
            result_t_s = self.topic_fusion(history_seq[:, -1, :], topic)
            # framing-series result
            result_f_s = self.framing_fusion(history_seq[:, -1, :], framing)
            # grand fusion result
            if self.config.use_framing:
                features = [language, abs_time_seq[:, -1, :], topic, framing]
            else:
                features = [language, abs_time_seq[:, -1, :], topic]
            fused_features = torch.cat(features, dim=1)
            result_all = self.grand_fusion(history_seq[:, -1, :], fused_features)
            
            # all results
            if self.config.use_framing:
                prediction = self.sum_weight[seq_len-1](torch.cat([result_l_s, result_a_s, result_t_s, result_f_s, result_all], dim=1))
            else:
                prediction = self.sum_weight[seq_len-1](torch.cat([result_l_s, result_a_s, result_t_s, result_all], dim=1))
        
        else:
            # time-series result
            result_a_s = self.abs_time_fusion(history_seq[:, -1, :], abs_time_seq[:, -1, :])
            prediction = self.one_weight(result_a_s)      

        return prediction.unsqueeze(-1)


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
        self.abs_time_fusion = FusionNet(self.config.hidden_size, self.config.time_size, self.config.medium_features, 1)

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


    def forward(self, text, dec_input, others):
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
        return prediction.unsqueeze(-1)


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

    def forward(self, text, dec_input, others=None):
        representation, _ = self.rnn_model(dec_input)
        batch_size, time_len, _ = representation.shape
        representation = representation.reshape(batch_size*time_len, -1)
        predict = self.predict_fc(representation)

        return predict.view(batch_size, time_len, 1)


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

    def forward(self, text, dec_input, others=None):
        x = dec_input.permute(0, 2, 1)
        out = self.tcn_model(x)
        out = out.permute(0, 2, 1)
        return out


def getModel(modelName):
    MODEL_MAP = {
        'rnn': RNN,
        'tcn': TCN,
        'spwrnn': SPWRNN,
        'spwrnn2': SPWRNN,
        'spwrnn_wo_l': SPWRNN_WO_L,
    }

    assert modelName in MODEL_MAP.keys(), 'Not support ' + modelName

    return MODEL_MAP[modelName]

