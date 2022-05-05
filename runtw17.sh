#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python main.py --seed=66 --dataset=twitter \
--train_path=data/twitter2017/train.txt \
--dev_path=data/twitter2017/dev.txt \
--test_path=data/twitter2017/test.txt