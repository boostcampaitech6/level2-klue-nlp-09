import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
import transformers
from load_data import *
import numpy as np
import random

import wandb

# wandb 및 기타 설정
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['WANDB_PROJECT'] = ''
os.environ['WANDB_ENTITY'] = ''

def set_seed(seed:int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    print('SEED SET TO:', seed)

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }

def label_to_num(label):
    num_label = []
    with open('KSW/dict_label_to_num.pkl', 'rb') as f: # ****** 여기에 dict_label_to_num.pkl 파일 위치 입력 ******
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
  
    return num_label

def get_label_weights(train_path):
    # df_train = pd.read_csv('../dataset/train/train.csv')
    df_train = pd.read_csv(train_path)

    # 1번 방법 -> weight를 안정성 있게 주는 방법
    df_train['label'] = label_to_num(df_train['label'])
    label_counts = df_train['label'].value_counts().sort_index().to_list()
    weights = [1-(x/sum(label_counts)) for x in label_counts]
    weights = torch.FloatTensor(weights)

    return weights

class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights = get_label_weights("/data/ephemeral/level2_klue/data/train_split2.csv") # ****** 여기에 train_path 추가 ******
        loss_func = torch.nn.CrossEntropyLoss(weight=weights.to(model.device))
        loss = loss_func(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train():
  set_seed(42)
  # load model and tokenizer
  MODEL_NAME = "klue/roberta-large" # ****** 여기에 모델명 추가 ******
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  num_added_toks = tokenizer.add_tokens(['[/ODAT]', '[/OLOC]', '[/ONOH]', '[/OORG]', '[/OPER]', '[/OPOH]', '[/SORG]', '[/SPER]', '[ODAT]', '[OLOC]', '[ONOH]', '[OORG]', '[OPER]', '[OPOH]', '[SORG]', '[SPER]'], special_tokens=True)

  # load dataset
  per_train_dataset, org_train_dataset = load_data("/data/ephemeral/level2_klue/data/train_split2.csv") # ****** 여기에 train_split2.csv 데이터 위치 추가 ******
  per_dev_dataset, org_dev_dataset = load_data("/data/ephemeral/level2_klue/data/validation_split2.csv") # ****** 여기에 validation_split2.csv 데이터 위치 추가 ******

  per_train_label = label_to_num(per_train_dataset['label'].values)
  org_train_label = label_to_num(org_train_dataset['label'].values)
  per_dev_label = label_to_num(per_dev_dataset['label'].values)
  org_dev_label = label_to_num(org_dev_dataset['label'].values)

  # tokenizing dataset
  per_tokenized_train = tokenized_dataset(per_train_dataset, tokenizer)
  org_tokenized_train = tokenized_dataset(org_train_dataset, tokenizer)
  per_tokenized_dev = tokenized_dataset(per_dev_dataset, tokenizer)
  org_tokenized_dev = tokenized_dataset(org_dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_per_train_dataset = RE_Dataset(per_tokenized_train, per_train_label)
  RE_org_train_dataset = RE_Dataset(org_tokenized_train, org_train_label)
  RE_per_dev_dataset = RE_Dataset(per_tokenized_dev, per_dev_label)
  RE_org_dev_dataset = RE_Dataset(org_tokenized_dev, org_dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  for entity in ['per', 'org']:
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config, cache_dir='/data/ephemeral/huggingface')
    model.resize_token_embeddings(len(tokenizer))
    print(model.config)
    model.parameters
    model.to(device)
  
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=f'/data/ephemeral/level2_klue/KSW/results/240111_{entity}_rule_based_newtoken_aug', # ****** 어디에 저장할지 입력 꼭 {entity}로 구분시킬 것 ******
        seed=42,

        save_total_limit=1,
        save_strategy='epoch',
        # save_steps=500,
        logging_dir='./logs',
        logging_strategy='steps',
        logging_steps=1,

        num_train_epochs=20,
        learning_rate=5e-5,
        lr_scheduler_type='cosine',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        max_grad_norm=1.0,

        evaluation_strategy='epoch',
        # eval_steps=500,
            
        load_best_model_at_end=True,
        metric_for_best_model='eval_micro f1 score',
        greater_is_better=True, # if loss False

        report_to='wandb',
        run_name=MODEL_NAME + f'{entity}_run_time',
    )

    wandb.login(key='') # ****** 여기에 wandb 키 입력 ******
    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience=3,
    )

    trainer = CustomTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=eval(f'RE_{entity}_train_dataset'),         # training dataset
        eval_dataset=eval(f'RE_{entity}_dev_dataset'),             # evaluation dataset
        compute_metrics=compute_metrics,         # define metrics function
        callbacks=[early_stopping],
    )

    # train model
    trainer.train()
    model.save_pretrained(f'/data/ephemeral/level2_klue/KSW/best_model/240113_{entity}_rule_based_newtoken_large_aug') # ****** 여기에 저장할 좌표 입력 꼭 {entity}로 구분시킬 것 ******
    wandb.finish()

def main():
  train()

if __name__ == '__main__':
  main()