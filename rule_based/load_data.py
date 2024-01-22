import pickle as pickle
import os
import pandas as pd
import torch


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  sub_list = []
  obj_list = []
  type_list = []
  sentence_list = []

  for sub, obj, sen in zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence']):
    sub = eval(sub)
    obj = eval(obj)
    sub_entity = sub['word']
    obj_entity = obj['word']
    sub_type = sub['type']
    obj_type = obj['type']
    new_sen = sen.replace(sub_entity, f"[S{sub_type}]{sub_entity}[/S{sub_type}]").replace(obj_entity, f"[O{obj_type}]{obj_entity}[/O{obj_type}]")
    sub_list.append(sub_entity)
    obj_list.append(obj_entity)
    sentence_list.append(new_sen)
    type_list.append(sub_type)
  
  dataset['subject_entity'] = sub_list
  dataset['object_entity'] = obj_list
  dataset['sentence'] = sentence_list
  dataset['sub_type'] = type_list

  per_dataset = dataset[(dataset['sub_type'] == 'PER') | (dataset['sub_type'] == 'LOC')].drop(['sub_type', 'source'], axis=1)
  org_dataset = dataset[dataset['sub_type'] == 'ORG'].drop(['sub_type', 'source'], axis=1)
  return per_dataset, org_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  per_dataset, org_dataset = preprocessing_dataset(pd_dataset)
  
  return per_dataset, org_dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset["sentence"]),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences
