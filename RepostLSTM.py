import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F


class FusionNet2(torch.nn.Module):
    def __init__(self, main_features, auxiliary_features, medium_features, target_features) -> None:
        super(FusionNet2, self).__init__()
        
        self.fc1 = torch.nn.Linear(main_features+auxiliary_features, medium_features)
        self.func1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(medium_features, medium_features)
        self.func2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(medium_features, medium_features)
        self.func3 = torch.nn.LeakyReLU()
        self.fc_final = torch.nn.Linear(medium_features, target_features)
        self.func4 = torch.nn.LeakyReLU()
    
    def forward(self, main, auxiliary):
        features = torch.cat([main, auxiliary], dim=1)
        features = self.fc1(features)
        features = self.func1(features)
        medium_features = self.fc2(features)
        medium_features = self.func2(medium_features) + features
        final_features = self.fc3(medium_features)
        final_features = self.func3(final_features)
        result = self.func4(self.fc_final(final_features))
        return result


class RepostLSTM(torch.nn.Module):
    def __init__(self, config, args) -> None:
        super(RepostLSTM, self).__init__()
        self.config = config
        self.args = args

        # language related model
        self.language_model = AutoModel.from_pretrained(self.config.pretrained_model)
        self.language_model_config = AutoConfig.from_pretrained(self.config.pretrained_model)
        self.language_hidden_size = self.language_model_config.hidden_size
        # calc dim of fusion module input
        if config.features_proj:
            language_fusion_dim = self.config.language_proj_size
            topic_fusion_dim = self.config.topic_proj_size  
        else:
            language_fusion_dim = self.language_hidden_size
            topic_fusion_dim = self.config.topic_size           
        self.language_fusion = FusionNet2(self.config.hidden_size, language_fusion_dim, self.config.medium_features, 1)
        # sentence vector extract
        self.public_vector = nn.Parameter(torch.randn((self.config.public_size)))
        self.word_key = nn.Linear(self.language_hidden_size, self.config.public_size)
        self.softmax = nn.Softmax(dim=1)
        self.language_proj = nn.Linear(self.language_hidden_size, self.config.language_proj_size)
        self.language_vec_func = nn.ReLU()
        # framing classification
        self.framing_fc = nn.Linear(self.language_hidden_size, self.config.framing_size)
        self.framing_vec_func = nn.Sigmoid()

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
        self.topic_fusion = FusionNet2(self.config.hidden_size, topic_fusion_dim, self.config.medium_features, 1)

        # framing related model
        self.framing_fusion = FusionNet2(self.config.hidden_size, self.config.framing_size, self.config.medium_features, 1)

        # time related model
        self.abs_time_fusion = FusionNet2(self.config.hidden_size, self.config.time_size, self.config.medium_features, 1)

        # grand fusion related model
        auxiliary_dim = language_fusion_dim + topic_fusion_dim + self.config.time_size
        if self.config.use_framing:
            auxiliary_dim += self.config.framing_size 
        self.grand_fusion = FusionNet2(self.config.hidden_size, auxiliary_dim, self.config.medium_features, 1)

        # final weighted results
        # self.weighed_sum = nn.Linear(5, 1) if self.config.use_framing else nn.Linear(4, 1)
        in_dim = 5 if self.config.use_framing else 4

        self.unique_sum_weight = nn.Linear(in_dim, 1)
        self.sum_weight = [nn.Linear(in_dim, 1).to(self.args.device) for i in range(self.config.initialize_steps)]
        self.one_weight = nn.Linear(1, 1)
        # self.weight_norm = nn.Softmax(dim=0)

        # eval cache (only available when do_test)
        self.cache_language = None
        self.cache_topic = None
        self.cache_framing = None
    
    def clear_eval_cache(self):
        """
        After infering one weibo, the cache of features must be removed.
        """
        self.cache_language = None
        self.cache_topic = None        
    
    def check_cache(self):
        return self.cache_language is not None

    def write_cache(self, language_features, topic_features, framing_features):
        self.cache_language, self.cache_topic, self.cache_framing = language_features, topic_features, framing_features

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
                text_features, topic_features, framing_features = self.cache_language, self.cache_topic, self.cache_framing
                return history_features, text_features, topic_features, abs_time, framing_features

        # extract language features
        text_representation = self.language_model(text['input_ids'], text['attention_mask'])['last_hidden_state']
        batch_size, text_len, text_dim = text_representation.shape
        
        # get interest semantics (sentence vector)
        text_representation_ravel = text_representation.view(batch_size*text_len, -1)
        key = self.word_key(text_representation_ravel)
        # inner product between word vectors and public vectors
        scores = (self.public_vector.unsqueeze(0) * key).view(batch_size, text_len, -1).sum(dim=-1)
        # softmax -> word weights in sentence (except cls token)
        pad_mask = (1 - text['attention_mask']) * -999.
        scores = (scores * text['attention_mask'] + pad_mask)[:, 1:]
        scores = self.softmax(scores)
        interest_text = (text_representation[:, 1:, :] * scores.unsqueeze(-1)).sum(dim=1)
        # proj of text
        if self.config.features_proj:
            text_features = self.language_vec_func(self.language_proj(interest_text))
        else:
            text_features = self.language_vec_func(interest_text)
        
        # auto framing
        framing_features = self.framing_fc(text_representation[:, 0, :])
        framing_features = self.framing_vec_func(framing_features)

        # proj of topics
        if self.config.features_proj:
            topic_features = self.topic_vec_func(self.topic_proj(others['topics']))
        else:
            topic_features = others['topics']

        if mode == 'VAL':
            self.write_cache(text_features, topic_features, framing_features)

        return history_features, text_features, topic_features, abs_time, framing_features


    def forward(self, text, dec_input, others):
        """
        Train forward.
        Return all prediction. [batch_size, seq_len, 1]
        """
        seq_len = dec_input.shape[1]
        history_seq, language, topic, abs_time_seq, framing = self.get_features(text, dec_input, others)
        if self.config.constant_framing:
            framing = others['framing']
        if self.config.use_predicted_framing:
            framing = others['predicted_framing']

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
                # if use different weight in different timestamps
                if self.config.unique_fusion_weights:
                    if self.config.use_framing:
                        prediction_i = self.unique_sum_weight(torch.cat([result_l_s, result_a_s, result_t_s, result_f_s, result_all], dim=1))
                    else:
                        prediction_i = self.unique_sum_weight(torch.cat([result_l_s, result_a_s, result_t_s, result_all], dim=1))                    
                else:
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
        return prediction.unsqueeze(-1) + dec_input, framing


    def predict_next(self, text, dec_input, others):
        """
        Train forward.
        Return last position (next timestamp) prediction. [batch_size, 1, 1]
        """
        seq_len = dec_input.shape[1]
        history_seq, language, topic, abs_time_seq, framing = self.get_features(text, dec_input, others, 'VAL')
        if self.config.constant_framing:
            framing = others['framing']
        if self.config.use_predicted_framing:
            framing = others['predicted_framing']

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
            # if use different weight in different timestamps
            if self.config.unique_fusion_weights:
                if self.config.use_framing:
                    prediction = self.unique_sum_weight(torch.cat([result_l_s, result_a_s, result_t_s, result_f_s, result_all], dim=1))
                else:
                    prediction = self.unique_sum_weight(torch.cat([result_l_s, result_a_s, result_t_s, result_all], dim=1))                    
            else:
                if self.config.use_framing:
                    prediction = self.sum_weight[seq_len-1](torch.cat([result_l_s, result_a_s, result_t_s, result_f_s, result_all], dim=1))
                else:
                    prediction = self.sum_weight[seq_len-1](torch.cat([result_l_s, result_a_s, result_t_s, result_all], dim=1))
        
        else:
            # time-series result
            result_a_s = self.abs_time_fusion(history_seq[:, -1, :], abs_time_seq[:, -1, :])
            prediction = self.one_weight(result_a_s)      

        return F.relu(prediction.unsqueeze(-1)) + dec_input[:, -1:, :], framing
