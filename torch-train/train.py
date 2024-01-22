from datetime import datetime
import pandas as pd

from utils import *
from modeling import *
from data_handling_tagging import get_tokenizer, return_dataloader

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set Config
class args:
    model_name = 'soddokayo/klue-roberta-large-klue-ner'
    model_type = 'lstm' # auto, t5, lstm -> auto로 설정하세요!
    batch_size = 32
    max_epoch = 5
    shuffle = True
    learning_rate = 1e-5
    warmup_steps = 100
    seed = 109
    loss_type = 'CLE'
    add_special_token = True
    weight_decay = 0.01
    gradient_clip_norm = 1.0
    gradient_accumulation_step = 1
    num_workers = 8

    # dataset file path
    train_path = '../dataset/train/train_aug.csv'
    dev_path = '../dataset/train/train_sampling_777.csv'
    test_path = '../dataset/test/test_data.csv'

    # huggingface save path
    cache_dir = '/data/ephemeral/huggingface'
    
    # metric early stopping
    metric = 'custom' # loss, f1, auprc, accuracy, custom
    patience = 10
    f1_ratio = 0.9
    loss_ratio = 0.1
    mode = 'min' # min, max

    # wandb settings
    wandb_logger = True # False to disable WandB
    wandb_key = ''
    wandb_project = ''
    wandb_entity = ''
    wandb_group = None
    wandb_name = f'_{model_name}_{RUN_TIME}'

    file_name = f'{RUN_TIME}_run'

    as_json = {
        'model_name' : model_name,
        'batch_size' : batch_size,
        'max_epoch' : max_epoch,
        'shuffle' : shuffle,
        'learning_rate' : learning_rate,
        'seed' : seed,
        'loss_type': loss_type,
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path' : test_path,
        'metric':metric,
        'patience' : patience,
        'weight_decay' : weight_decay,
        'gradient_clip_norm':gradient_clip_norm,
    }


if __name__ == '__main__':
    setup()
    environ_set()
    seed_all(args.seed)

    # tokenizer & dataloaders
    tokenizer = get_tokenizer(args)
    dataloaders = return_dataloader(args, tokenizer)

    # model training & inference pipeline
    pipeline = Pipeline(
        tokenizer,
        dataloaders=dataloaders,
        args=args,
        device='cuda:0'
    )

    # train and evaluate
    pipeline.forward()

    # predict
    probs, preds = pipeline.inference()
    pred_answer = num_to_label(preds)

    # save preds
    output = pd.read_csv('../prediction/sample_submission.csv')
    output['pred_label'] = pred_answer
    output['probs'] = probs.tolist()
    output.to_csv(f'./outputs/{args.file_name}_{args.model_name.replace("/","-")}.csv', index=False)