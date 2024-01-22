from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from train import set_seed

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('KSW/dict_num_to_label.pkl', 'rb') as f: # ****** 여기에 dict_num_to_label.pkl 파일 위치 입력 ******
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  per_test_dataset, org_test_dataset = load_data(dataset_dir)
  per_test_label = list(map(int,per_test_dataset['label'].values))
  org_test_label = list(map(int,org_test_dataset['label'].values))
  
  # tokenizing dataset
  per_tokenized_test = tokenized_dataset(per_test_dataset, tokenizer)
  org_tokenized_test = tokenized_dataset(org_test_dataset, tokenizer)
  return per_test_dataset['id'], per_tokenized_test, per_test_label, org_test_dataset['id'], org_tokenized_test, org_test_label

def main(args):
  set_seed(42)
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = "klue/roberta-large" # ****** 여기에 모델명 입력 ******
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
  num_added_toks = tokenizer.add_tokens(['[/ODAT]', '[/OLOC]', '[/ONOH]', '[/OORG]', '[/OPER]', '[/OPOH]', '[/SORG]', '[/SPER]', '[ODAT]', '[OLOC]', '[ONOH]', '[OORG]', '[OPER]', '[OPOH]', '[SORG]', '[SPER]'], special_tokens=True)

  ## load test datset
  test_dataset_dir = "/data/ephemeral/level2_klue/data/test_data.csv" # ****** 여기에 test_data.csv 파일 위치 입력 ******
  per_test_id, per_test_dataset, per_test_label, org_test_id, org_test_dataset, org_test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_per_test_dataset = RE_Dataset(per_test_dataset, per_test_label)
  Re_org_test_dataset = RE_Dataset(org_test_dataset, org_test_label)

  ## load my model
  output_list = []
  for entity in ['per', 'org']:
    MODEL_NAME = f'KSW/best_model/240113_{entity}_rule_based_newtoken_large_aug' # ****** 여기에 train에서 학습한 모델 위치 입력 model.save_pretrained('') 입력한거 그대로 입력 ******
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    # model.resize_token_embeddings(len(tokenizer))
    model.parameters
    model.to(device)

    ## predict answer
    pred_answer, output_prob = inference(model, eval(f'Re_{entity}_test_dataset'), device) # model에서 class 추론
    pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    
    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id':eval(f'{entity}_test_id'), 'pred_label':pred_answer,'probs':output_prob,})
    #### 필수!! ##############################################
    output_list.append(output)
    print(f'---- {entity} predict Finish! ----')
  result = pd.concat(output_list)
  result = result.sort_values(by=['id'])
  result.to_csv(f'./KSW/result_csv/result.csv', index=False) # ****** 최종 csv 파일 저장할 좌표 입력 ******
  print('---- Finish! ----')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  print(args)
  main(args)