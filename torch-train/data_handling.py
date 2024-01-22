import pandas as pd

import torch

import transformers
from utils import *


class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        if len(self.labels) > 0:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.pair_dataset['input_ids'])


def get_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir,
        # use_fast=False
    )
    return tokenizer


def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = i[1:-1].split(',')[0].split(':')[1]
        j = j[1:-1].split(',')[0].split(':')[1]

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    return out_dataset


def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    return tokenized_sentences


def return_dataloader(args, tokenizer):
    train_dataset = load_data(args.train_path)
    dev_dataset = load_data(args.dev_path)
    test_dataset = load_data(args.test_path)

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)
    test_label = []

    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)

    train_dataset = RE_Dataset(tokenized_train, train_label)
    val_dataset = RE_Dataset(tokenized_dev, dev_label)
    test_dataset = RE_Dataset(tokenized_test, test_label)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        # sampler=sampler, 
        num_workers=0,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=0,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=0,
    )

    return {
        'train':train_dataloader, 
        'val':val_dataloader, 
        'test':test_dataloader,
    }