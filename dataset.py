import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import torch


class MindDataset(Dataset):
    def __init__(
        self,
        args,
        file_path,
        news_dict,
        cate_dict,
        tokenizer,
        mode = 'train',
        npratio = 4,
        nrows = None,
    ):  
        self.ctitle_size = args.ctitle_size
        self.htitle_size = args.htitle_size
        self.max_his_size = args.max_his_size
        self.max_input_size = args.max_input_size
        self.prefix_size = args.prefix_size
        self.prompt = args.prompt

        self.file_path = file_path
        self.news_dict = news_dict
        self.cate_dict = cate_dict
        self.tokenizer = tokenizer 
        self.mode = mode
        self.nrows = nrows

        self.prompt_encode()
        self.nsep_token_id = tokenizer.convert_tokens_to_ids(
            tokens=args.nsep_token,
        )
        self.cls_token_id = tokenizer.convert_tokens_to_ids(
            tokens=tokenizer.cls_token,
        )
        self.sep_token_id = tokenizer.convert_tokens_to_ids(
            tokens=tokenizer.sep_token,
        )
        self.special_token_size = 2
        self.max_fill_size = (
            self.max_input_size 
            - self.prefix_size 
            - self.prompt_size 
            - self.special_token_size 
        )

        if mode == 'train':
            self.npratio = npratio

        self.samples = []
        self.impid2idx = {}
        self.impid2history = {}

        self.gene_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def imps_len(self):
        return len(self.impid2idx)

    def prompt_encode(self):
        self.prompt_id = self.prompt.split(' ')
        self.prompt_size = 0
        for idx, token in enumerate(self.prompt_id):
            if token != '<USER>' and token != '<CANDIDATE>':
                self.prompt_id[idx] = self.tokenizer.convert_tokens_to_ids(token)
                self.prompt_size += 1

    def prompt_addition(self, history, imp):
        candidate = self.news_dict[imp]['title'][:self.ctitle_size]
        candidate = [self.nsep_token_id] + candidate

        max_user_size = self.max_fill_size - len(candidate)
        
        history = history.split()
        user = []
        hcate = []
        for nid in history[-self.max_his_size:][::-1]:
            title = self.news_dict[nid]['title'][:self.htitle_size]
            title = [self.nsep_token_id] + title
            if len(user) + len(title) > max_user_size:
                break
            user = title + user
            cate = self.news_dict[nid]['cate']
            hcate = [cate] + hcate
        hcate = [self.cate_dict[cate] for cate in hcate]
        
        input = []
        for token in self.prompt_id:
            if token == '<USER>':
                input.extend(user)
            elif token == '<CANDIDATE>':
                input.extend(candidate)
            else:
                input.append(token)
        input = [self.cls_token_id] + input + [self.sep_token_id]
        input, atten = self.padding(input)
        
        ccate = self.news_dict[imp]['cate']
        ccate = self.cate_dict[ccate]
        return input, atten, ccate, hcate
    
    def padding(self, input):
        input_size = len(input)
        padding_size = (
            self.max_input_size
            - input_size 
            - self.prefix_size
        )
        atten = [1] * input_size + [0] * padding_size
        input = input + [0] * padding_size
        return input, atten

    def gene_samples(self):
        """
        Generate samples from impressions
        """
        column_names = ['impid', 'uid', 'time', 'history', 'imps']
        raw_data = pd.read_csv(
            self.file_path, sep='\t',
            header=None,
            names=column_names,
            nrows=self.nrows,
        )
        raw_data['history'] = raw_data['history'].fillna('')
        for _, row in tqdm(raw_data.iterrows()):
            history = row['history']
            if len(history) == 0:
                continue
            self.impid2history[row['impid']] = history
            imps = row['imps'].split()
            imps = [imp.split('-') for imp in imps]
            if self.mode == 'train':
                imps_pos = [imp[0] for imp in imps if imp[1] == '1']
                imps_neg = [imp[0] for imp in imps if imp[1] == '0']
                for pos in imps_pos:
                    self.samples.append({
                        'impid': row['impid'], 'imp': pos, 'label': 1
                    })
                    if len(imps_neg) >= self.npratio:
                        imps_samp_neg = random.sample(imps_neg, k=self.npratio)
                    else:
                        imps_samp_neg = np.random.choice(imps_neg, self.npratio).tolist()
                    for neg in imps_samp_neg:
                        self.samples.append({
                            'impid': row['impid'], 'imp': neg, 'label': 0
                        })
            elif self.mode == 'test':
                for imp in imps:
                    self.samples.append({
                        'impid': row['impid'], 'imp': imp[0], 'label': int(imp[1])
                    })
        for idx, sample in enumerate(self.samples):
            impid = sample['impid']
            if impid not in self.impid2idx:
                self.impid2idx[impid] = []
            self.impid2idx[impid].append(idx)

    def train_val_split(self, val_imps_len):
        """ 
        Split dataset by impressions
        """
        if self.mode == 'test':
            return
        
        val_imps = random.sample(self.impid2idx.keys(), val_imps_len)
        val_imps = set(val_imps)
        train_indices = []
        val_indices = []
        for impid, idx in self.impid2idx.items():
            if impid in val_imps:
                val_indices.extend(idx)
            else:
                train_indices.extend(idx)
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        return train_dataset, val_dataset
    
    def collate_fn(self, batch):
        batch_impid = [x['impid'] for x in batch]
        batch_label = [x['label'] for x in batch]
        batch_input, batch_atten, batch_ccate, batch_hcate = [], [], [], []
        
        for x in batch:
            history = self.impid2history[x['impid']]
            input, atten, ccate, hcate = self.prompt_addition(history, x['imp'])
            batch_input.append(input)
            batch_atten.append(atten)
            batch_ccate.append(ccate)
            batch_hcate.append(hcate)
            
        batch_hlen = [len(hcate) for hcate in batch_hcate]
        for i, hcate in enumerate(batch_hcate):
            batch_hcate[i] = hcate + (self.max_his_size - len(hcate)) * [0]

        batch_impid = torch.LongTensor(batch_impid)
        batch_input = torch.LongTensor(batch_input)
        batch_atten = torch.LongTensor(batch_atten)
        batch_ccate = torch.LongTensor(batch_ccate)
        batch_hcate = torch.LongTensor(batch_hcate)
        batch_hlen = torch.LongTensor(batch_hlen)
        batch_label = torch.LongTensor(batch_label)

        return (
            batch_impid, batch_input, batch_atten, 
            batch_ccate, batch_hcate, batch_hlen, batch_label
        )
