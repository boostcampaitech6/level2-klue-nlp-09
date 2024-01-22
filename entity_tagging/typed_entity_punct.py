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

def preprocessing(dataset):
    sub = []
    obj = []
    sentences = [] 
    
    for i,j,s in zip(dataset['subject_entity'],dataset['object_entity'],dataset['sentence']):
        i = eval(i) # subject entity
        j = eval(j) # object entity
        
        sub_entity = i['word']
        obj_entity = j['word']
        
        s = s.replace(sub_entity,"@ ~ ["+i['type']+"] ~ '"+sub_entity + "' @")
        s = s.replace(obj_entity, "# ^ ["+j['type']+"] ^ '"+obj_entity + "' #")
        
        sub.append("@ ~ ["+i['type']+"] ~ '"+sub_entity + "' @")
        obj.append("# ^ ["+j['type']+"] ^ '"+obj_entity + "' #")
        sentences.append(s)
    
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': sentences,
                                'subject_entity':sub, 'object_entity': obj,
                                'label':dataset['label']})

    return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    #temp = e01 + tokenizer.sep_token + e02
    temp = e01 +"와"+ e02 +"의 관계"
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