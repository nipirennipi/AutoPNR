import pandas as pd
import os
import random
import sys
import numpy as np
import torch

from tqdm import tqdm

import torch.distributed as dist

from logger import Logger

GREEN = "\033[32m"
RESET = "\033[0m"


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2024"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_news(file_path, tokenizer):
    column_names = [
        'nid', 'cate', 'subcate', 'title', 'abstract'
    ]
    raw_data = pd.read_csv(
        file_path, 
        sep='\t', 
        header=None, 
        names=column_names,
    )

    news_dict = {}
    cate_dict = {}
    cate_id = 1
    for idx, row in tqdm(raw_data.iterrows()):
    # for idx, row in tqdm(raw_data.iloc[:500].iterrows()):
        title = tokenizer.encode(row['title'], add_special_tokens=False)
        cate = row['cate']
        news_dict[row['nid']] = {'title': title, 'cate': cate}
        if cate not in cate_dict.keys():
            cate_dict[cate] = cate_id
            cate_id += 1
    return news_dict, cate_dict


def green_print(rank, values):
    print_once(rank, GREEN + values + RESET)


def print_once(rank, values):
    if rank == 0:
        print(values)


def logging(args, file_name):
    if args.log_tag is not None:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        log_file_path = os.path.join(args.log_path, args.log_tag)
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)
        train_log_file = os.path.join(log_file_path, file_name)
        sys.stdout = Logger(train_log_file, sys.stdout)


def get_optimizer_params(model, lr_bert, lr_prompt, lr_answer):
    optimizer_params = [
        {'params': [], 'lr': lr_bert}, 
        {'params': [], 'lr': lr_prompt},
        {'params': [], 'lr': lr_answer},
    ]
    for name, param in model.named_parameters():
        if 'prompt_generate' in name:
            optimizer_params[1]['params'].append(param)
        elif 'answer_search' in name:
            optimizer_params[2]['params'].append(param)
        else:
            optimizer_params[0]['params'].append(param)
    return optimizer_params
    