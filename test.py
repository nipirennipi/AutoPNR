import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '2, 5, 6, 7'

from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoConfig

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from arguments import get_args
from dataset import MindDataset
from model import BertAplo4NR
from utils import *
from metrics import *
from train import test


def main(rank, world_size):
    ddp_setup(rank, world_size)
    
    args = get_args()

    if rank == 0:
        logging(args, 'test_log.txt')

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

    green_print(rank, '### 3. Load testset') 
    test_dataset = MindDataset(
        args=args,
        file_path=os.path.join(args.data_path, 'test_behaviors.txt'),
        news_dict=news_dict,
        cate_dict=cate_dict,
        tokenizer=tokenizer,
        mode='test',
    )
    imps_len = test_dataset.imps_len()
    print_once(
        rank, 
        f'# test impressions: {imps_len}\n' \
        f'# test samples: {len(test_dataset)}' \
    )

    test_kwargs = {
        'batch_size': args.infer_batch_size, 
        'collate_fn': test_dataset.collate_fn,
        'sampler': DistributedSampler(test_dataset, shuffle=False),
    }
    test_loader = DataLoader(test_dataset, **test_kwargs)

    green_print(rank, '### 4. Load model and checkpoint')
    model = BertAplo4NR(
        config=config,
    )
    model.resize_token_embeddings(new_num_tokens)
    model.set_answer_search(need_search_module=False)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    save_path = os.path.join(args.ckpt_path, args.ckpt_name)
    print_once(rank, f'load from {save_path}')
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint)

    green_print(rank, '### 5. Start testing')
    print_once(rank, f'time: {datetime.now()}')
    test_logloss, auc, mrr, ndcg5, ndcg10 = (
        test(rank, args, model, test_loader)
    )
    print_once(
        rank,
        f'test info || logloss: {test_logloss:.4f} | auc: {auc:.4f} ' \
        f'| mrr: {mrr:.4f} | ndcg@5: {ndcg5:.4f} | ndcg@10: {ndcg10:.4f}' \
    )


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
