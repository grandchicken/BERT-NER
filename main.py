from load_data import load_data,load_data_twitter,Prepare,NER_Dataset
import argparse
import torch
import numpy as np
import random
# from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import logging
from datetime import datetime
import os
from torch.utils.data import DataLoader
from model import BERT_NER
from train import Trainer
import fitlog
def define_log(args):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    file = logging.FileHandler(filename=log_dir, mode='a', encoding='utf-8')
    fmt = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s",
                            datefmt='%Y-%m-%d %H:%M:%S')
    file.setFormatter(fmt)
    logger = logging.Logger(name='logger', level=logging.INFO)
    logger.addHandler(file)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_path',
                        default='/home/data_ti6_d/lich/Multimodal_NER/Rp_BERT/pretrained/bert-base-cased',
                        type=str)
    parser.add_argument('--dataset',default='conll2003',type=str,choices=['conll2003','twitter'])
    parser.add_argument('--train_path',default='data/conll2003/train.txt',type=str)
    parser.add_argument('--dev_path',default='data/conll2003/dev.txt',type=str)
    parser.add_argument('--test_path',default='data/conll2003/test.txt',type=str)
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else 'cpu'), type=str)
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--hidden_size',default=768,type=int)
    parser.add_argument('--epochs',default=25,type=int)
    parser.add_argument('--warmup', default=0.1, type=float, help='warmup')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
    parser.add_argument('--lr', default=7e-5, type=float, )
    parser.add_argument('--grad_clip', default=5, type=float, help='grad_clip')
    parser.add_argument('--seed', type=int, default=66, help='max_len')
    parser.add_argument('--eval_every', type=int, default=1, help='eval_every')
    log_dir = 'log/'
    parser.add_argument('--log_dir',default=log_dir,type=str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args = parser.parse_args()
    return args


args = parse_args()
logger = define_log(args)
label_list = ['[PAD]',"O", "B-OTHER", "I-OTHER", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
fitlog_dir = 'flog/'
if not os.path.exists(fitlog_dir):
    os.makedirs(fitlog_dir)
fitlog.set_log_dir(fitlog_dir)
fitlog.add_hyper(args)  # 通过这种方式记录ArgumentParser的参数
fitlog.add_hyper_in_file(__file__)  # 记录本文件中写死的超参数

def fix_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
fix_seed(args)

if args.dataset == 'conll2003':
    train_data,train_label = load_data(args.train_path)
    dev_data,dev_label = load_data(args.dev_path)
    test_data,test_label = load_data(args.test_path)
elif args.dataset == 'twitter':
    train_data, train_label = load_data_twitter(args.train_path)
    dev_data, dev_label = load_data_twitter(args.dev_path)
    test_data, test_label = load_data_twitter(args.test_path)

p_train = Prepare(args,label_list,train_data,train_label)
p_dev = Prepare(args,label_list,dev_data,dev_label)
p_test = Prepare(args,label_list,test_data,test_label)
train_dict = p_train.prepare_data()
dev_dict = p_dev.prepare_data()
test_dict = p_test.prepare_data()

label_map = train_dict['label_map']

train_dataset = NER_Dataset(train_dict)
dev_dataset = NER_Dataset(dev_dict)
test_dataset = NER_Dataset(test_dict)


train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
dev_dataloader = DataLoader(dataset=dev_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

NER_model = BERT_NER(args,label_map)
NER_model.to(args.device)
t = Trainer(args,logger,NER_model,label_map,train_dataloader,dev_dataloader,test_dataloader)
t.train()