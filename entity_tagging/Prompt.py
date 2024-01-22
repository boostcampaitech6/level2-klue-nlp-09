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

def prompt_preprocessing(dataset):
  prompt = []

  for i,j in zip(dataset['subject_entity'],dataset['object_entity']):
    i = eval(i)['word'] # subject_entity
    j = eval(j)['word'] # object_entity

    prompt.append(i+"와 "+j+"의 관계")

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],
                              'prompt': prompt, 'label': dataset['label']})
  
  return out_dataset

    
def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = prompt_preprocessing(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  tokenized_sentences = tokenizer(
      list(dataset['prompt']),
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  
  return tokenized_sentences