from tqdm import tqdm
from time import sleep

import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

import wandb

from utils import *
from models import get_model

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


def compute_metrics(logits, labels, mode='val'):
    """ validation을 위한 metrics function """
    labels = labels.cpu().numpy()
    preds = logits.argmax(-1).cpu().numpy()
    probs = logits.cpu().numpy()

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
        f'{mode}/f1': f1,
        f'{mode}/auprc' : auprc,
        f'{mode}/accuracy': acc,
    }



class EarlyStopping:
    def __init__(self, args, metric, save_path, max_patinece=3, mode='min'):
        if mode == 'min':
            self.best_val = np.inf
            self.mode = min
        else:
            self.best_val = -np.inf
            self.mode = max
        
        self.model_type = args.model_type
        self.max_patience = max_patinece
        self.patience_stack = 0
        self.metric = metric

        self.early_stopping = False # if true earlystop
        self.save_path = save_path

    def check_patience(self, model, metrics: dict):
        current_value = metrics[self.metric]
        value = self.mode(self.best_val, current_value)
        if value == current_value:
            # update value
            self.patience_stack = 0
            self.best_val = current_value
            self.save_best_model(model)
        else:
            # stack
            print(f'Patience Stack {self.patience_stack} -> {self.patience_stack+1}')
            self.patience_stack += 1
            if self.patience_stack >= self.max_patience:
                print('Max patience reached, Early Stopping...')
                self.early_stopping = True
                model = self.load_best_model(model)
        return model

    def save_best_model(self, model):
        # transformer 기준
        print(f'Save Best Model metric: {self.metric} -> {self.best_val}')
        model.save_pretrained(self.save_path)

    def load_best_model(self, model):
        print(f'Load Best Model')
        if self.model_type == 'auto':
            model = AutoModelForSequenceClassification.from_pretrained(self.save_path)
        else:
            model.from_pretrained(self.save_path)
        return model



class Pipeline:
    def __init__(self, tokenizer, dataloaders, args, device='cuda:0'):
        self.model_name = args.model_name
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = self.get_model()
        self.loss_func = self.get_loss_func()

        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['val']
        self.test_dataloader = dataloaders['test']

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        total_steps = len(self.train_dataloader) * self.args.max_epoch // self.args.gradient_accumulation_step
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps//20)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.args.warmup_steps // self.args.gradient_accumulation_step, num_training_steps=total_steps
        )

        self.early_stopping = EarlyStopping(
            self.args,
            f'val/{args.metric}',
            f'./pretrained/{args.file_name}_{args.model_name.replace("/","-")}',
            max_patinece=args.patience,
            mode=args.mode
        )

        self.run = None

    def forward(self):
        if self.args.wandb_logger:
            self.setup_wandb()
        for epoch in range(1, self.args.max_epoch + 1):
            print(f'---EPOCH {epoch}/{self.args.max_epoch}-----------------------------------------------------')
            if self.args.gradient_accumulation_step == 1: self.train()
            else: self.train_accumulate()
            metrics = self.validate()
            self.model = self.early_stopping.check_patience(self.model, metrics)
            if self.early_stopping.early_stopping:
                break
        self.model = self.early_stopping.load_best_model(self.model).to(self.device)
        test_metrics = self.validate(mode='test')
        if self.args.wandb_logger:
            self.close_wandb_run()

    def train(self):
        sleep(1)
        self.model.train()
        train_loss = 0.0
        
        for idx, batch in enumerate(tqdm(self.train_dataloader, desc='Train Mode')):
            self.optimizer.zero_grad()
            batch = {key:value.to(self.device) for key, value in batch.items()}
            outputs = self.model(**batch)
            logits = outputs.get('logits')

            loss = self.loss_func(logits.view(-1, self.model.config.num_labels), batch['labels'].view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()
            if self.args.wandb_logger:
                wandb.log({
                    "train/loss": loss,
                    'Charts/lr-AdamW':self.scheduler.get_last_lr()[0],
                    }
                )

        print(f'Train Loss: {train_loss/(len(self.train_dataloader)):.4f}')

    def train_accumulate(self):
        sleep(1)
        self.model.train()
        train_loss = 0.0
        temp_loss = 0.0
        
        for idx, batch in enumerate(tqdm(self.train_dataloader, desc='Train Mode')):
            self.optimizer.zero_grad()
            batch = {key:value.to(self.device) for key, value in batch.items()}
            outputs = self.model(**batch)
            logits = outputs.get('logits')

            loss = self.loss_func(logits.view(-1, self.model.config.num_labels), batch['labels'].view(-1))
            loss /= self.args.gradient_accumulation_step
            loss.backward()
            temp_loss += loss
            if (idx+1) % self.args.gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                train_loss += temp_loss.item()
                if self.args.wandb_logger:
                    wandb.log({
                        "train/loss": temp_loss,
                        'Charts/lr-AdamW':self.scheduler.get_last_lr()[0],
                        }
                    )

                temp_loss = 0.0

        print(f'Train Loss: {train_loss/(len(self.train_dataloader)//self.args.gradient_accumulation_step):.4f}')


    def validate(self, mode='val'):
        sleep(1)
        self.model.eval()
        val_loss = 0.0
        
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.val_dataloader, desc=f'{mode} Mode')):
                batch = {key:value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.get('logits')

                loss = self.loss_func(logits.view(-1, self.model.config.num_labels), batch['labels'].view(-1))
                val_loss += loss.item()
                logits_list.append(logits)
                labels_list.append(batch['labels'])
            
            total_logits = torch.cat(logits_list, dim=0)
            total_labels = torch.cat(labels_list, dim=0)
            metric_dict = compute_metrics(total_logits, total_labels, mode=mode)
        
        metric_dict[f'{mode}/loss'] = val_loss/len(self.val_dataloader)
        metric_dict[f'{mode}/custom'] = (1 - metric_dict[f'{mode}/f1']/100) * self.args.f1_ratio + metric_dict[f'{mode}/loss'] * self.args.loss_ratio
        if self.args.wandb_logger:
            wandb.log(metric_dict)
        print(metric_dict)
        # if mode=='test':
        #     wandb.Table(
        #         columns=["predictions", "targets"],
        #         data = [
        #             [int(x), int(y)] for (x, y) in 
        #             zip(total_logits.argmax(-1).cpu().numpy(), total_labels.cpu().numpy())
        #         ],
        #     )
        return metric_dict

    def inference(self):
        sleep(1)
        self.model.eval()
        logits_list = []
        probs_list = []
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_dataloader, desc='Predict Mode')):
                batch = {key:value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.get('logits')

                probs = torch.nn.functional.softmax(logits, dim=-1)
                logits_list.append(logits)
                probs_list.append(probs)
            
            total_logits = torch.cat(logits_list, dim=0)
            total_probs = torch.cat(probs_list, dim=0)

            result = total_logits.argmax(dim=-1)
        
        return total_probs.detach().cpu().numpy(), result.detach().cpu().numpy()

    def get_model(self):
        model = get_model(self.args)
        if self.args.add_special_token:
            model.resize_token_embeddings(len(self.tokenizer))
        model = model.to(self.device)
        return model


    def get_label_weights(self):
        # df_train = pd.read_csv('../dataset/train/train.csv')
        df_train = pd.read_csv(self.args.train_path)

        df_train['label'] = label_to_num(df_train['label'])
        label_counts = df_train['label'].value_counts().sort_index().to_list()
        weights = [1-(x/sum(label_counts)) for x in label_counts]
        weights = torch.FloatTensor(weights)

        # label = np.array(label_to_num(df_train['label']))
        # weights = sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.unique(label), y=label)
        # weights = np.where(weights > 1, 1.0, weights)
        # weights = torch.FloatTensor(weights)

        # label = label_to_num(df_train['label'])
        # weights = 1.0 / torch.bincount(torch.tensor(label))

        return weights


    def get_loss_func(self):
        weights = self.get_label_weights()
        loss_func = nn.CrossEntropyLoss(weight=weights.to(self.device))
        return loss_func

    def setup_wandb(self):
        wandb.login(key=self.args.wandb_key)
        self.run = wandb.init(
            # Set the project where this run will be logged
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            group=self.args.wandb_group,
            name=self.args.wandb_name, 
            # Track hyperparameters and run metadata
            config=self.args.as_json,
        )
    
    def close_wandb_run(self):
        # self.run.log_code()
        self.run.finish()