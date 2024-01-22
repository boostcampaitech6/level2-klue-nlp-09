from datetime import datetime
import pandas as pd

from utils import *
from modeling import *
from data_handling_tagging import get_tokenizer, return_dataloader

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set Config
class args:
    model_name = 'klue/roberta-large'
    model_type = 'pipeline_model' # auto, t5, lstm, pipeline_model -> auto로 설정하세요!
    batch_size = 8
    max_epoch = 10
    shuffle = True
    learning_rate = 3e-5
    warmup_steps = 100
    seed = 42
    loss_type = 'CLE'
    add_special_token = True
    weight_decay = 0.01
    gradient_clip_norm = 1.0
    gradient_accumulation_step = 4
    num_workers = 4

    # dataset file path
    train_path = '../dataset/train/train_split2.csv'
    dev_path = '../dataset/train/validation_split2.csv'
    test_path = '../dataset/test/test_data.csv'

    # huggingface save path
    cache_dir = '/data/ephemeral/huggingface'
    
    # metric early stopping
    metric = 'custom' # loss, f1, auprc, accuracy, custom
    patience = 3
    f1_ratio = 0.8
    loss_ratio = 0.2
    mode = 'min' # min, max

    # wandb settings
    wandb_key = '176497a02f725f116c9286a67b092e5dcf5ae8f8'
    wandb_project = 'level2_klue'
    wandb_entity = 'moth2aflame'
    wandb_group = None
    wandb_name = f'JHW_{model_name}_{RUN_TIME}'

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

    # Load Model
    MODEL_PATH = './pretrained/20240115_164342_run_klue-roberta-large'
    pipeline.model = pipeline.model.from_pretrained(MODEL_PATH).to(pipeline.device)

    # predict
    probs, preds = pipeline.inference()
    pred_answer = num_to_label(preds)

    # save preds
    output = pd.read_csv('../prediction/sample_submission.csv')
    output['pred_label'] = pred_answer
    output['probs'] = probs.tolist()
    output.to_csv(f'./outputs/{args.file_name}_{args.model_name.replace("/","-")}.csv', index=False)