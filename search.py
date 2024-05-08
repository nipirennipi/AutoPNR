import os
import pprint
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'

from collections import OrderedDict
from datetime import datetime
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertTokenizerFast, 
    AutoConfig, 
    get_constant_schedule_with_warmup,
)

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from arguments import get_args
from dataset import MindDataset
from model import BertAplo4NR
from utils import *
from metrics import *
from train import train, test


def main(rank, world_size):
    ddp_setup(rank, world_size)

    args = get_args()
    init_seed(args.seed)

    if rank == 0:
        logging(args, 'search_log.txt')

    green_print(rank, '### arguments:')
    if rank == 0:
        pprint.pprint(args.__dict__, width=1)

    green_print(rank, '### 1. Load news and tokenizer')
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [args.nsep_token]
    })
    new_num_tokens = len(tokenizer)

    news_dict, cate_dict = read_news(
        file_path=os.path.join(args.data_path, 'news.txt'), 
        tokenizer=tokenizer,
    )
    args.n_cates = len(cate_dict)
    print_once(rank, f"# news: {len(news_dict)}")
    print_once(rank, f"# cate: {len(cate_dict)}")

    green_print(rank, '### 2. Load config')
    config = AutoConfig.from_pretrained(
        args.model_name,
    )
    config.update({
        'mask_token_id': tokenizer.mask_token_id,
        'n_cates': args.n_cates,
        'prefix_size': args.prefix_size,
        'generator_hidden_size': args.generator_hidden_size,
        'cate_dim': args.cate_dim,
        'tau': args.tau,
        'gru_num_layers': args.gru_num_layers,
        'n_labels': args.n_labels,
    })
    args.max_input_size = config.max_position_embeddings
    print_once(rank, 'done.')

    green_print(rank, '### 3. Load data and split') 
    mind_dataset = MindDataset(
        args=args,
        file_path=os.path.join(args.data_path, 'train_behaviors.txt'),
        news_dict=news_dict,
        cate_dict=cate_dict,
        tokenizer=tokenizer,
        mode='train',
        npratio=args.npratio,
    )
    imps_len = mind_dataset.imps_len()
    val_imps_len = int(imps_len * args.val_ratio)
    train_imps_len = imps_len - val_imps_len
    print_once(
        rank, 
        f'# total impressions: {imps_len:>6}\n' \
        f'# train impressions: {train_imps_len:>6} | {1 - args.val_ratio:6.2%}\n' \
        f'# valid impressions: {val_imps_len:>6} | {args.val_ratio:6.2%}\n' \
        f'# total samples: {len(mind_dataset)}' \
    )
    
    train_dataset, val_dataset = mind_dataset.train_val_split(val_imps_len)

    train_kwargs = {
        'batch_size': args.train_batch_size, 
        'collate_fn': mind_dataset.collate_fn,
        'sampler': DistributedSampler(train_dataset, shuffle=True),
    }
    val_kwargs = {
        'batch_size': args.infer_batch_size, 
        'collate_fn': mind_dataset.collate_fn,
        'sampler': DistributedSampler(val_dataset, shuffle=False),
    }
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    green_print(rank, '### 4. Load model and optimizer')
    model = BertAplo4NR.from_pretrained(
        args.model_name,
        config=config,
    )
    model.resize_token_embeddings(new_num_tokens)
    model.set_answer_search(need_search_module=True)
    model.freeze()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_once(rank, name)
    model.to(rank)
    model = DDP(
        module=model,
        device_ids=[rank],
        # find_unused_parameters=True,
    )
    optimizer_params = get_optimizer_params(
        model=model,
        lr_bert=args.learning_rate_bert,
        lr_prompt=args.learning_rate_prompt,
        lr_answer=args.learning_rate_answer,
    )
    optimizer = AdamW(
        params=optimizer_params,
        weight_decay=args.weight_decay,
    )
    scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
    )
    dist.barrier()
    
    green_print(rank, '### 5. Start search')
    print_once(rank, f'time: {datetime.now()}')
    for epoch in range(args.search_epochs):
        print_once(rank, '-' * 88)
        print_once(rank, f'epoch: {epoch}')
        train_logloss = train(rank, args, model, optimizer, scheduler, train_loader)
        print_once(rank, f'train info || logloss: {train_logloss:.4f}')
        val_logloss, auc, mrr, ndcg5, ndcg10 = (
            test(rank, args, model, val_loader)
        )
        print_once(
            rank, 
            f'valid info || logloss: {val_logloss:.4f} | auc: {auc:.4f} ' \
            f'| mrr: {mrr:.4f} | ndcg@5: {ndcg5:.4f} | ndcg@10: {ndcg10:.4f}' \
        )
        dist.barrier()

    green_print(rank, '### 6. Save answer')
    if rank == 0:
        if not os.path.exists(args.ckpt_path):
            os.makedirs(args.ckpt_path)
        answer_path = os.path.join(args.ckpt_path, 'answer.pt')
        answer_dict = model.module.get_answer_dict()
        torch.save(answer_dict, answer_path)
        print_once(rank, f'save at {answer_path}')

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    
