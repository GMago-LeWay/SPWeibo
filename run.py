import torch
import torch.nn as nn
import random
import numpy as np
import logging
import argparse
import time
import os
import pandas as pd
import urllib
from tqdm import tqdm

from train import getTrain
from utils import getTime
from data import getData
from model import getModel
from config import Config


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str, default='spw',
                        help='spw')    
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
    parser.add_argument('--load', type=str, default='results/models/spw.pth',
                        help='model to be loaded in infer task.')
    return parser.parse_args()


def run(args, config):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,
                                        f'{args.modelName}.pth')
    dataset_ = getData(args.modelName)(args=args, config=config)
    model_to_be_init = getModel(modelName=args.modelName)
    model = model_to_be_init(config=config, args=args).to(args.device)
    train_to_be_init = getTrain(modelName=args.modelName)
    train = train_to_be_init(args=args, config=config)
    train_loader, val_loader = dataset_.get_train_val_dataloader()
    # do train
    best_score = train.do_train(model=model, train_dataloader=train_loader, val_dataloader=val_loader)
    # save result
    logging.info(getTime() + '本次最优结果：%.4f' % best_score)

    return best_score

def run_eval(args, config):
    
    # model settings
    # BUG
    dataset_ = getData(args.modelName)(args=args, config=config)
    model_to_be_init = getModel(modelName=args.modelName)
    model = model_to_be_init(config=config, args=args).to(args.device)
    model.load_state_dict(torch.load(args.load))
    train_to_be_init = getTrain(modelName=args.modelName)
    train = train_to_be_init(args=args, config=config)

    # infer
    results = train.do_infer(model, dataset_.get_test_dataloader())

    return results


def run_task(args, seeds, configure):
    logging.info('************************************************************')
    logging.info(getTime() + '本轮参数：' + str(configure))

    result = []

    for seed in seeds:
        setup_seed(seed)
        # 每个种子训练开始
        logging.info(getTime() + 'Seed：%d 训练开始' % seed)      
        score = run(args, configure)
        result.append(score)

    # 保存实验结果
    mean, std = round(np.mean(result)*100, 2), round(np.std(result)*100, 2)
    logging.info('本轮效果均值：%f, 标准差：%f' % (mean, std))

    save_path = os.path.join(args.res_save_dir,
                            f'results.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model", "ValidateAvg", "ValidateStd", "Args", "Config"])
   
    res = [args.modelName, mean, std, str(args), str(configure)]
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logging.info('Results are added to %s...' % (save_path))
    logging.info('************************************************************')


def run_tune(args, seeds, tune_times=50):
    has_debuged = []
    for i in range(tune_times):
        logging.info('-----------------------------------Tune(%d/%d)----------------------------' % (i+1, tune_times))
        # 加载之前的结果参数以去重
        save_path = os.path.join(args.res_save_dir, f'results.csv')
        if i == 0 and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            for j in range(len(df)):
                has_debuged.append(df.loc[j, "Config"])

        setup_seed(int(time.time()))              # 随机选取种子以初始化随机的config
        config = Config(args.modelName, tune=True).get_config()

        if str(config) in has_debuged:
            logging.info(getTime() + '该参数已经被搜索过.')
            time.sleep(1.)
            continue

        try:
            run_task(args=args, seeds=seeds, configure=config)
            has_debuged.append(str(config))
        except Exception as e:
            logging.info(getTime() + '运行时发生错误. ' + str(e))


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists('log'):
        os.makedirs('log')
    logging.basicConfig(filename=f'log/{args.modelName}.log', level=logging.INFO)
    args.device = 'cuda:'+ str(args.device)
    if args.infer:
        configure = Config(modelName=args.modelName).get_config()
        run_eval(args=args, config=configure)
    elif not args.tune:
        configure = Config(modelName=args.modelName).get_config()
        run_task(args=args, seeds=[11111], configure=configure)
    else:
        run_tune(args=args, seeds=[111, 1111, 11111], tune_times=100)

