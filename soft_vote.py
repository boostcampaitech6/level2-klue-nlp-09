import pandas as pd
import numpy as np
import pickle

from glob import glob

csv_files = glob('./ensembled/*.csv')

sum_weight = sum([3, 2, 2, 1.5, 1.5])

weight_dict = {
    'lstm':3/sum_weight,
    'klue':2/sum_weight,
    'nlpotato':2/sum_weight,
    'soddokayo':1.5/sum_weight,
    'xlm':1.5/sum_weight,
}

file_counts = {
    'lstm':3,
    'klue':3,
    'nlpotato':3,
    'soddokayo':5,
    'xlm':3,
}

weights = [0] * len(csv_files)
pds = [pd.read_csv(csv) for csv in csv_files]
test_data = pd.read_csv('../dataset/test/test_data.csv')

probs = [x.probs.apply(eval).to_list() for x in pds]
probs = [np.array(x) for x in probs]
probs = np.array(probs)

for i, w in enumerate(weights):
    probs[i] *= w

probs = probs.sum(axis=0)

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

label = num_to_label(probs.argmax(-1))

sub = pd.read_csv('../prediction/sample_submission.csv')
sub['pred_label'] = label
sub['probs'] = list(probs)

sub.probs = sub.probs.apply(list)

sub.to_csv('./ensembled.csv', index=False)