import os, random

import pickle, re
import numpy as np

import torch

# num_workers 가능
def environ_set():
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# seed 고정
def seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("SEED VALUE SET:", seed)


def text_preprocess(text):
    emoji_pattern = re.compile("["
        u"\U00010000-\U0010FFFF"
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    
    removal_pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    text = emoji_pattern.sub(r'', text)
    text = removal_pattern.sub(r'', text)
    return text


def apply_preprocess(df_list):
    for df in df_list:
        df['sentence'] = df['sentence'].apply(text_preprocess)
    return df_list


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label

def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label

def make_dir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        print(f'Created {dirname}')

def setup():
    make_dir('pretrained')
    make_dir('outputs')
    make_dir('outputs')