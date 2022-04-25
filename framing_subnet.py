import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel


from utils import dict_to_str, getTime
from metrics import Metrics


class BERT_CLS(torch.nn.Module):
    def __init__(self, config, args) -> None:
        super(BERT_CLS, self).__init__()
        self.config = config
        self.args = args
        self.language_model = AutoModel.from_pretrained(self.config.pretrained_model)
        self.predict_fc = torch.nn.Linear(768, 6)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        cls_representation = self.language_model(text['input_ids'], text['attention_mask'])['last_hidden_state'][:, 0, :]
        logits = self.predict_fc(self.dropout(cls_representation))
        probs = self.softmax(logits)

        return logits, probs


class TrainBERT():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.criterion = nn.MSELoss()
        self.metrics = Metrics().get_metrics('framing')

    def do_train(self, model, train_dataloader, val_dataloader):
        # OPTIMIZER: finetune Bert Parameters.
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.language_model.named_parameters())

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
                    labels = batch_data['others']['framing'].to(self.args.device)
                    texts = batch_data['texts'].to(self.args.device)

                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    logits, probs = model(text=texts)
                    # compute loss
                    loss = self.criterion(probs, labels)
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss.append(loss.item())
                    y_pred.append(probs.cpu())
                    y_true.append(labels.cpu())

                    if steps == len(train_dataloader) or (self.config.eval_step and steps % self.config.eval_step == 0):
                        valid_num += 1
                        # calc data of training
                        train_loss_avg = np.mean(train_loss)
                        train_loss = []
                        pred, true = torch.cat(y_pred), torch.cat(y_true)
                        y_pred, y_true = [], []
                        train_results = self.metrics(pred, true)
                        logging.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f %s" % (self.args.modelName, \
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
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    labels = batch_data['others']['framing'].to(self.args.device)
                    texts = batch_data['texts'].to(self.args.device)
                    # inference
                    logits, probs = model(text=texts)
                    loss = self.criterion(probs, labels)
                    eval_loss += loss.item()
                    y_pred.append(probs.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logging.info("%s-(%s) >> %s" % (mode, self.args.modelName, dict_to_str(eval_results)))
        return eval_results, pred, true

    def do_infer(self, model, dataloader):
        model.eval()
        y_pred = []
        with torch.no_grad():
            for batch_data in dataloader:
                texts = batch_data['texts'].to(self.args.device)
                # inference
                logits, probs = model(text=texts)
                y_pred.append(probs.cpu())
        predictions = torch.cat(y_pred)
        predictions = ( predictions > 0.5 ).int()
        return predictions.numpy()
