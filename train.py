import os
import time
import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_beta import SPWRNN_BETA


from utils import dict_to_str, getTime
from metrics import Metrics


class SPWRNN():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.criterion = nn.MSELoss()
        self.metrics = Metrics().get_metrics(self.args.modelName)

    def do_train(self, model, train_dataloader, val_dataloader):
        # OPTIMIZER: finetune Bert Parameters.
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.language_model.named_parameters()) if self.config.language_model else []

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'language_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.config.weight_decay_bert, 'lr': self.config.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.config.learning_rate_bert},
            {'params': model_params_other, 'weight_decay': self.config.weight_decay_other, 'lr': self.config.learning_rate_other}
        ]

        optimizer = optim.Adam(optimizer_grouped_parameters)

        # SCHEDULER
        scheduler = ReduceLROnPlateau(optimizer,
                    mode=self.config.scheduler_mode,
                    factor=0.5, patience=self.config.scheduler_patience, verbose=True)
        # initilize results
        epochs = 0
        valid_num, best_valid_num = 0, 0
        min_or_max = self.config.scheduler_mode
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while epochs < self.config.max_epochs: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            train_loss = []
            steps = 0
            with tqdm(train_dataloader) as td:
                for batch_data in td:
                    steps += 1
                    model.train()
                    labels = batch_data['labels'].to(self.args.device)
                    texts = batch_data['texts'].to(self.args.device)
                    dec_inputs = batch_data['dec_inputs'].to(self.args.device)
                    for key in batch_data['others']:
                        batch_data['others'][key] = batch_data['others'][key].to(self.args.device)
                    # framing substitution under test mode if predicted framings are used
                    if self.config.use_predicted_framing:
                        batch_data['others']['framing'] = batch_data['others']['predicted_framing']                    
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(text=texts, dec_input=dec_inputs, others=batch_data['others'])
                    outputs = outputs.squeeze(-1)
                    # compute loss
                    loss = self.criterion(outputs, labels)
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss.append(loss.item())
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())

                    if steps == len(train_dataloader) or (self.config.eval_step and steps % self.config.eval_step == 0):
                        valid_num += 1
                        # calc data of training
                        train_loss_avg = np.mean(train_loss)
                        train_loss = []
                        pred, true = torch.cat(y_pred), torch.cat(y_true)
                        y_pred, y_true = [], []
                        train_results = self.metrics(pred, true)
                        logging.info(getTime() + "TRAIN-(%s) (%d/%d/%d)>> loss: %.4f %s" % (self.args.modelName, \
                                    epochs, valid_num - best_valid_num, valid_num, train_loss_avg, dict_to_str(train_results)))

                        # validation
                        val_results, _, _ = self.do_test(model, val_dataloader, mode="VAL")
                        cur_valid = val_results[self.config.KeyEval]
                        # scheduler step
                        scheduler.step(cur_valid)
                        # save best model
                        isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
                        # save best model
                        if isBetter:
                            # save model
                            best_valid, best_valid_num = cur_valid, valid_num
                            torch.save(model.cpu().state_dict(), self.args.model_save_path)
                            model.to(self.args.device)
                        # early stop
                        if valid_num - best_valid_num >= self.config.early_stop:
                            return best_valid
        
        return best_valid


    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        gross_eval_results = {}
        for observe_time in self.config.observe_time:
            y_pred, y_true = [], []
            eval_loss = 0.0
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        labels = batch_data['labels'].to(self.args.device)
                        texts = batch_data['texts'].to(self.args.device)
                        dec_inputs = batch_data['dec_inputs'].to(self.args.device)
                        for key in batch_data['others']:
                            batch_data['others'][key] = batch_data['others'][key].to(self.args.device)
                        # framing substitution under test mode if predicted framings are used
                        if self.config.use_predicted_framing:
                            batch_data['others']['framing'] = batch_data['others']['predicted_framing']
                        # calc observing steps
                        steps = int(observe_time / self.config.interval)
                        dec_inputs = dec_inputs[:, 0:steps+1, :]
                        
                        out_len = steps
                        outputs = [dec_inputs[:, 1:steps+1, :]]
                        # inference
                        while out_len < labels.shape[1]:
                            if self.config.name == 'SPWRNN':
                                new_prediction = model.predict_next(text=texts, dec_input=dec_inputs, others=batch_data['others'])
                            else:
                                prediction = model(text=texts, dec_input=dec_inputs, others=batch_data['others'])
                                new_prediction = prediction[:, -1:, :]
                            outputs.append(new_prediction)
                            dec_inputs = torch.cat([dec_inputs, new_prediction], dim=1)
                            out_len += 1
                        # rm cache of current weibo
                        if self.config.name == 'SPWRNN':
                            model.clear_eval_cache()

                        outputs = torch.cat(outputs, dim=1).squeeze(-1)
                        loss = self.criterion(outputs, labels)
                        eval_loss += loss.item()
                        y_pred.append(outputs.cpu())
                        y_true.append(labels.cpu())
                eval_loss = eval_loss / len(dataloader)
                pred, true = torch.cat(y_pred), torch.cat(y_true)
                # clip predicted part of results for metric calc
                pred, true = pred[:, steps:], true[:, steps:]
                # metric calc
                eval_results = self.metrics(pred, true)
                eval_results["Loss"] = round(eval_loss, 4)

                # merge results
                hours_prefix = str(observe_time/3600) + 'h_'
                for key in eval_results:
                    gross_eval_results[hours_prefix+key] = eval_results[key]


        logging.info(getTime() + "%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(gross_eval_results)))
        return gross_eval_results, pred, true

    def do_infer(self, model, dataloader):
        # for new weibo without any repost number
        model.eval()
        y_pred = []
        with torch.no_grad():
            for batch_data in dataloader:
                texts = batch_data['texts'].to(self.args.device)
                dec_inputs = torch.zeros((texts.shape[0], 1, 1)).to(self.args.device)
                for key in batch_data['others']:
                    batch_data['others'][key] = batch_data['others'][key].to(self.args.device)
                # inference
                out_len = 0
                outputs = []
                while out_len < self.config.seq_dim:
                    if self.config.name == 'SPWRNN':
                        new_prediction = model.predict_next(text=texts, dec_input=dec_inputs, others=batch_data['others'])
                    else:
                        prediction = model(text=texts, dec_input=dec_inputs, others=batch_data['others'])
                        new_prediction = prediction[:, -1:, :]
                    outputs.append(new_prediction)
                    dec_inputs = torch.cat([dec_inputs, new_prediction], dim=1)
                    out_len += 1
                # rm cache of current weibo
                if self.config.name == 'SPWRNN':
                    model.clear_eval_cache()
                
                outputs = torch.cat(outputs, dim=1)
                y_pred.append(outputs.cpu())
        predictions = torch.cat(y_pred)
        return predictions.numpy()


class SPWRNN_WITH_FRAMING():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.criterion = nn.MSELoss()
        self.criterion_framing = nn.MSELoss()
        self.metrics = Metrics().get_metrics(self.args.modelName)
        self.metrics_framing = Metrics().get_metrics('framing')

    def do_train(self, model, train_dataloader, val_dataloader):
        # OPTIMIZER: finetune Bert Parameters.
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.language_model.named_parameters()) if self.config.language_model else []

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'language_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.config.weight_decay_bert, 'lr': self.config.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.config.learning_rate_bert},
            {'params': model_params_other, 'weight_decay': self.config.weight_decay_other, 'lr': self.config.learning_rate_other}
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters)

        # SCHEDULER
        scheduler = ReduceLROnPlateau(optimizer,
                    mode=self.config.scheduler_mode,
                    factor=self.config.scheduler_factor, patience=self.config.scheduler_patience, verbose=True)
        # initilize results
        epochs = 0
        valid_num, best_valid_num = 0, 0
        min_or_max = self.config.scheduler_mode
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while epochs < self.config.max_epochs: 
            epochs += 1
            # train
            y_pred, y_true = [], []
            train_loss = []
            train_loss_framing = []
            steps = 0
            with tqdm(train_dataloader) as td:
                for batch_data in td:
                    steps += 1
                    model.train()
                    labels = batch_data['labels'].to(self.args.device)
                    texts = batch_data['texts'].to(self.args.device)
                    dec_inputs = batch_data['dec_inputs'].to(self.args.device)
                    for key in batch_data['others']:
                        batch_data['others'][key] = batch_data['others'][key].to(self.args.device)
                    
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs, framing = model(text=texts, dec_input=dec_inputs, others=batch_data['others'])
                    outputs = outputs.squeeze(-1)
                    # compute loss
                    loss_series = self.criterion(outputs, labels)
                    loss_framing = self.criterion_framing(framing, batch_data['others']['framing'])
                    loss = loss_series + self.config.framing_loss_weight * loss_framing
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss.append(loss_series.item())
                    train_loss_framing.append(loss_framing.item())
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())

                    if steps == len(train_dataloader) or (self.config.eval_step and steps % self.config.eval_step == 0):
                        valid_num += 1
                        # calc data of training
                        train_loss_avg = np.mean(train_loss)
                        train_loss_framing_avg = np.mean(train_loss_framing)
                        train_loss = []
                        train_loss_framing = []

                        pred, true = torch.cat(y_pred), torch.cat(y_true)
                        y_pred, y_true = [], []
                        train_results = self.metrics(pred, true)
                        logging.info(getTime() + "TRAIN-(%s) (%d/%d/%d)>> loss: %.4f framing loss: %.4f; %s" % (self.args.modelName, \
                                    epochs, valid_num - best_valid_num, valid_num, train_loss_avg, train_loss_framing_avg, dict_to_str(train_results)))

                        # validation
                        val_results, _, _ = self.do_test(model, val_dataloader, mode="VAL")
                        cur_valid = val_results[self.config.KeyEval]
                        # scheduler step
                        scheduler.step(cur_valid)
                        # save best model
                        isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
                        # save best model
                        if isBetter:
                            # save model
                            best_valid, best_valid_num = cur_valid, valid_num
                            torch.save(model.cpu().state_dict(), self.args.model_save_path)
                            model.to(self.args.device)
                        # early stop
                        if valid_num - best_valid_num >= self.config.early_stop:
                            return best_valid
        
        return best_valid


    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        gross_eval_results = {}
        for observe_time in self.config.observe_time:
            y_pred, y_true = [], []
            framing_pred, framing_true = [], []
            eval_loss = 0.0
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        labels = batch_data['labels'].to(self.args.device)
                        texts = batch_data['texts'].to(self.args.device)
                        dec_inputs = batch_data['dec_inputs'].to(self.args.device)
                        for key in batch_data['others']:
                            batch_data['others'][key] = batch_data['others'][key].to(self.args.device)
                        # calc observing steps
                        steps = int(observe_time / self.config.interval)
                        dec_inputs = dec_inputs[:, 0:steps+1, :]
                        
                        out_len = steps
                        outputs = [dec_inputs[:, 1:steps+1, :]]
                        # inference
                        while out_len < labels.shape[1]:
                            new_prediction, framing = model.predict_next(text=texts, dec_input=dec_inputs, others=batch_data['others'])
                            outputs.append(new_prediction)
                            dec_inputs = torch.cat([dec_inputs, new_prediction], dim=1)
                            out_len += 1
                        # rm cache of current weibo
                        model.clear_eval_cache()

                        outputs = torch.cat(outputs, dim=1).squeeze(-1)
                        loss = self.criterion(outputs, labels)
                        eval_loss += loss.item()
                        y_pred.append(outputs.cpu())
                        y_true.append(labels.cpu())
                        framing_pred.append(framing.cpu())
                        framing_true.append(batch_data['others']['framing'].cpu())

                eval_loss = eval_loss / len(dataloader)
                pred, true = torch.cat(y_pred), torch.cat(y_true)
                framing_pred, framing_true = torch.cat(framing_pred), torch.cat(framing_true)
                # clip predicted part of results for metric calc
                pred, true = pred[:, steps:], true[:, steps:]
                # metric calc
                eval_results = self.metrics(pred, true)
                eval_results["SeriesLoss"] = round(eval_loss, 4)
                eval_framing_results = self.metrics_framing(framing_pred, framing_true)

                # merge results
                hours_prefix = str(observe_time/3600) + 'h_'
                for key in eval_results:
                    gross_eval_results[hours_prefix+key] = eval_results[key]


        logging.info(getTime() + "%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(gross_eval_results)))
        logging.info(getTime() + "%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(eval_framing_results)))
        return gross_eval_results, pred, true

    def do_infer(self, model, dataloader):
        # for new weibo without any repost number
        model.eval()
        y_pred = []
        with torch.no_grad():
            for batch_data in dataloader:
                texts = batch_data['texts'].to(self.args.device)
                dec_inputs = torch.zeros((texts.shape[0], 1, 1)).to(self.args.device)
                for key in batch_data['others']:
                    batch_data['others'][key] = batch_data['others'][key].to(self.args.device)
                # inference
                out_len = 0
                outputs = []
                while out_len < self.config.seq_dim:
                    if self.config.name == 'SPWRNN':
                        new_prediction = model.predict_next(text=texts, dec_input=dec_inputs, others=batch_data['others'])
                    else:
                        prediction = model(text=texts, dec_input=dec_inputs, others=batch_data['others'])
                        new_prediction = prediction[:, -1:, :]
                    outputs.append(new_prediction)
                    dec_inputs = torch.cat([dec_inputs, new_prediction], dim=1)
                    out_len += 1
                # rm cache of current weibo
                if self.config.name == 'SPWRNN':
                    model.clear_eval_cache()
                
                outputs = torch.cat(outputs, dim=1)
                y_pred.append(outputs.cpu())
        predictions = torch.cat(y_pred)
        return predictions.numpy()


def getTrain(modelName):
    TRAIN_MAP = {
        'rnn': SPWRNN,
        'tcn': SPWRNN,
        'spwrnn': SPWRNN,
        'spwrnn2': SPWRNN,
        'spwrnn_beta': SPWRNN_WITH_FRAMING,
        'spwrnn_wo_l': SPWRNN,
    }

    assert modelName in TRAIN_MAP.keys(), 'Not support ' + modelName

    return TRAIN_MAP[modelName]