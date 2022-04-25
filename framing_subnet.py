import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import time
import os
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoConfig, AutoModel, AutoTokenizer
import math
import torch.functional as F
import random
from utils import Storage


from utils import dict_to_str, getTime
from metrics import Metrics
from train import getTrain
from data import getData
from model import getModel
from config import Config


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
        self.metrics = Metrics().get_metrics(self.args.modelName)

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
        return predictions.numpy()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str, default='framing',
                        help='spwrnn/rnn/tcn')    
    parser.add_argument('--dataset', type=str, default='renminribao',
                        help='weibo dataset name')  
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU id.')
    parser.add_argument('--tune', type=bool, default=False,
                        help='True if run tune task.')
    parser.add_argument('--infer', type=bool, default=False,
                        help='True if run infer task.')
    parser.add_argument('--load', type=str, default='results/models/framing.pth',
                        help='model to be loaded in infer task.')
    return parser.parse_args()


def run(args, config):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,
                                        f'{args.modelName}.pth')
    dataset_ = getData('spwrnn')(args=args, config=config)
    model = BERT_CLS(config=config, args=args).to(args.device)
    train = TrainBERT(args=args, config=config)
    train_loader, val_loader, test_loader = dataset_.get_train_val_dataloader()
    # do train
    best_score = train.do_train(model=model, train_dataloader=train_loader, val_dataloader=val_loader)
    # save result
    logging.info(getTime() + '本次最优结果：%.4f' % best_score)

    model.load_state_dict(torch.load(args.model_save_path))
    test_results, pred, true = train.do_test(model=model, dataloader=test_loader, mode="TEST")

    return test_results

def run_eval(args, config):
    
    # model settings
    # BUG
    dataset_ = getData('spwrnn')(args=args, config=config)
    train_loader, val_loader, test_loader = dataset_.get_train_val_dataloader()
    model = BERT_CLS(config=config, args=args).to(args.device)
    model.load_state_dict(torch.load(args.load))
    train = TrainBERT(args=args, config=config)

    # infer
    results, pred, true = train.do_test(model, test_loader, mode='TEST')

    return results


def run_task(args, seeds, config):
    logging.info('************************************************************')
    logging.info(getTime() + '本轮参数：' + str(config))
    logging.info(getTime() + '本轮Args：' + str(args))

    original_result = {}

    for seed in seeds:
        setup_seed(seed)
        # 每个种子训练开始
        logging.info(getTime() + 'Seed：%d 训练开始' % seed)      
        current_res = run(args, config)
        if not original_result:
            for key in current_res:
                original_result[key] = [current_res[key]]
        else:
            for key in current_res:
                original_result[key].append(current_res[key])

    # 保存实验结果
    result = {}
    for key in original_result:            
        mean, std = round(np.mean(original_result[key]), 3), round(np.std(original_result[key]), 3)
        result[key] = str(mean)
        result[key + '-std'] = str(std)
    for key in config:
        result[key] = config[key]
    result['Args'] = args
    result['Config'] = config

    logging.info('本轮效果均值：%s, 标准差：%s' % (result[config.KeyEval], result[config.KeyEval + '-std']))

    mode = "tune" if args.tune else "single"
    save_path = os.path.join(args.res_save_dir, f'{args.modelName}-{args.dataset}-{mode}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        columns = set(df.columns)
        if set(result.keys()) == columns:       # 如果与已检测到的结果文件格式相符，直接保存
            df = df.append(result, ignore_index=True)
        else:  # 如果格式不符，另存一个文件
            for key in result:
                result[key] = [result[key]]
            df = pd.DataFrame(result)
            save_path = save_path[:-4] + "new.csv" 
            logging.info('Warning: 结果格式不符，另存一个新文件.')
    else:       # 不存在结果，新建文件
        for key in result:
            result[key] = [result[key]]
        df = pd.DataFrame(result)
   
    df.to_csv(save_path, index=None)
    logging.info('Results are added to %s...' % (save_path))
    logging.info('************************************************************')


def run_tune(args, seeds, tune_times=50):
    has_debuged = []
    for i in range(tune_times):
        logging.info('-----------------------------------Tune(%d/%d)----------------------------' % (i+1, tune_times))
        # 加载之前的结果参数以去重
        save_path = os.path.join(args.res_save_dir, f'{args.modelName}-{args.dataset}-tune.csv')
        if i == 0 and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            for j in range(len(df)):
                has_debuged.append(df.loc[j, "Config"])

        setup_seed(int(time.time()))              # 随机选取种子以初始化随机的config
        config = Config(args.modelName, args.dataset, tune=True).get_config()

        if str(config) in has_debuged:
            logging.info(getTime() + '该参数已经被搜索过.')
            time.sleep(1.)
            continue

        try:
            run_task(args=args, seeds=seeds, config=config)
            has_debuged.append(str(config))
        except Exception as e:
            logging.info(getTime() + '运行时发生错误. ' + str(e))


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists('log'):
        os.makedirs('log')
    file_name = f'log/{args.modelName}_tune.log' if args.tune else f'log/{args.modelName}_reg.log'
    logging.basicConfig(filename=file_name, level=logging.INFO)
    args.device = 'cuda:'+ str(args.device)
    if args.infer:
        configure = Config(modelName=args.modelName, dataset=args.dataset).get_config()
        run_eval(args=args, config=configure)
    elif not args.tune:
        args.model_save_dir = os.path.join(args.model_save_dir, 'regression')
        configure = Config(modelName=args.modelName, dataset=args.dataset).get_config()
        run_task(args=args, seeds=[111], config=configure)
    else:
        args.model_save_dir = os.path.join(args.model_save_dir, 'tune')
        run_tune(args=args, seeds=[111], tune_times=100)

