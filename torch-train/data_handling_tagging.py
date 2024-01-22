import pandas as pd

import torch

import transformers
from utils import *


class SpecialTokens:
    SBJ_START = '[sbj]'
    SBJ_END = '[/sbj]'
    OBJ_START = '[obj]'
    OBJ_END = '[/obj]'
    PER = '[per]'
    ORG = '[org]'
    DAT = '[dat]'
    LOC = '[loc]'
    POH = '[poh]'
    NOH = '[noh]'
    DICT = {
        'additional_special_tokens' : [SBJ_START, SBJ_END, OBJ_START, OBJ_END]
    }
    DICT2 = {
        'additional_special_tokens' : [SBJ_START, SBJ_END, OBJ_START, OBJ_END, PER, ORG, DAT, LOC, POH, NOH]
    }

TYPEDICT = {
    'sbj':0,
    'obj':1,
    'per':2,
    'org':3,
    'dat':4,
    'loc':5,
    'poh':6,
    'noh':7,
    'unk':-1
}



class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        if len(self.labels) > 0:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.pair_dataset['input_ids'])


def get_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
    # tokenizer = transformers.DebertaTokenizerFast.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir,
        # use_fast=False
    )

    if args.add_special_token:
        tokenizer.add_special_tokens(SpecialTokens.DICT2)
        print(tokenizer.all_special_tokens)
    return tokenizer


def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)
    return dataset


def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity_words = []
    object_entity_words = []
    subject_entity_idxs = []
    object_entity_idxs = []
    subject_entity_types = []
    object_entity_types = []
    for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
        # i_word, j_word = i[1:-1].split(',')[0].split(':')[1], j[1:-1].split(',')[0].split(':')[1]

        i, j = eval(i), eval(j)
        i_word, j_word = "'" +  i['word'] + "'", "'" +  j['word'] + "'"
        i_start, i_end = i['start_idx'], i['end_idx']
        j_start, j_end = j['start_idx'], j['end_idx']
        i_type, j_type = i['type'].lower(), j['type'].lower()

        subject_entity_idxs.append((i_start, i_end))
        object_entity_idxs.append((j_start, j_end))
        subject_entity_types.append(i_type)
        object_entity_types.append(j_type)
        subject_entity_words.append(i_word)
        object_entity_words.append(j_word)

    out_dataset = pd.DataFrame({
        'id':dataset['id'], 
        'sentence':dataset['sentence'],
        'subject_entity_idxs':subject_entity_idxs,
        'object_entity_idxs':object_entity_idxs,
        'subject_entity_type':subject_entity_types,
        'object_entity_type':object_entity_types,
        'subject_entity_word':subject_entity_words,
        'object_entity_word':object_entity_words,
        'label':dataset['label'],
    })
    return out_dataset

# main
def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    processed_sents = []
    concat_entity = []
    for sentence, (s_start, s_end), (o_start, o_end), s_type, o_type, s_word, e_word in zip(
        dataset['sentence'], 
        dataset['subject_entity_idxs'], dataset['object_entity_idxs'], 
        dataset['subject_entity_type'], dataset['object_entity_type'], 
        dataset['subject_entity_word'], dataset['object_entity_word']
        ):
        temp = make_type_to_token(s_type) + s_word + tokenizer.sep_token + make_type_to_token(o_type) + e_word
        concat_entity.append(temp)
        # concat_entity.append('<s>' + temp + '</s>') # for gpt, bart
        # temp = insert_special_token_in_sentence(
        #     sentence, s_start, s_end, o_start, o_end, 
        #     SpecialTokens.SBJ_START, SpecialTokens.SBJ_END, SpecialTokens.OBJ_START, SpecialTokens.OBJ_END
        # )
        temp = sentence.replace(s_word[1:-1], SpecialTokens.SBJ_START+s_word+SpecialTokens.SBJ_END)
        temp = temp.replace(e_word[1:-1], SpecialTokens.OBJ_START+e_word+SpecialTokens.OBJ_END)
        processed_sents.append(temp)
        # processed_sents.append(temp + '</s>') # for gpt, bart

    print('Sample Tokenized Sentence')
    print(tokenizer.tokenize(concat_entity[0], processed_sents[0], add_special_tokens=True))
    # print(self.tokenizer.tokenize(processed_sents[0])) # for xlm
    print('s_word:',dataset['subject_entity_word'][0])
    print('j_word:',dataset['object_entity_word'][0])
    print('-------------------------------------------------------------------------------')
    tokenized_sentences = tokenizer(
        concat_entity,
        processed_sents,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    # tokenized_sentences.pop('token_type_ids') # for bart
    print(tokenized_sentences.keys())
    return tokenized_sentences


# for aux
# def tokenized_dataset(dataset, tokenizer):
#     """ tokenizer에 따라 sentence를 tokenizing 합니다."""
#     processed_sents = []
#     concat_entity = []
#     type_ids = []
#     for sentence, (s_start, s_end), (o_start, o_end), s_type, o_type, s_word, e_word in zip(
#         dataset['sentence'], 
#         dataset['subject_entity_idxs'], dataset['object_entity_idxs'], 
#         dataset['subject_entity_type'], dataset['object_entity_type'], 
#         dataset['subject_entity_word'], dataset['object_entity_word']
#         ):
#         temp = make_type_to_token(s_type) + s_word + tokenizer.sep_token + make_type_to_token(o_type) + e_word
#         concat_entity.append(temp)
#         # concat_entity.append('<s>' + temp + '</s>') # for gpt, bart
#         # temp = insert_special_token_in_sentence(
#         #     sentence, s_start, s_end, o_start, o_end, 
#         #     SpecialTokens.SBJ_START, SpecialTokens.SBJ_END, SpecialTokens.OBJ_START, SpecialTokens.OBJ_END
#         # )
#         temp = sentence.replace(s_word[1:-1], SpecialTokens.SBJ_START+s_word+SpecialTokens.SBJ_END)
#         temp = temp.replace(e_word[1:-1], SpecialTokens.OBJ_START+e_word+SpecialTokens.OBJ_END)
#         processed_sents.append(temp)
#         type_id = [0, TYPEDICT[s_type], 1, TYPEDICT[o_type]] # [sbj, sbj_type, obj, obj_type]
#         type_ids.append(type_id)
#         # processed_sents.append(temp + '</s>') # for gpt, bart

#     print('Sample Tokenized Sentence')
#     print(tokenizer.tokenize(concat_entity[0], processed_sents[0], add_special_tokens=True))
#     # print(self.tokenizer.tokenize(processed_sents[0])) # for xlm
#     print('s_word:',dataset['subject_entity_word'][0])
#     print('j_word:',dataset['object_entity_word'][0])
#     print('-------------------------------------------------------------------------------')
#     tokenized_sentences = tokenizer(
#         concat_entity,
#         processed_sents,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=256,
#         add_special_tokens=True,
#     )
#     # tokenized_sentences.pop('token_type_ids') # for bart
#     tokenized_sentences['type_ids'] = torch.LongTensor(type_ids)
#     print(tokenized_sentences.keys())
#     return tokenized_sentences




    
def insert_special_token_in_sentence(sent, s_start, s_end, o_start, o_end, s_start_token, s_end_token, o_start_token, o_end_token):
    list_sent = list(sent)
    if s_start < o_start:
        list_sent.insert(s_start, s_start_token)
        list_sent.insert(s_end+2, s_end_token)
        list_sent.insert(o_start+2, o_start_token)
        list_sent.insert(o_end+4, o_end_token)
    else:
        list_sent.insert(o_start, o_start_token)
        list_sent.insert(o_end+2, o_end_token)
        list_sent.insert(s_start+2, s_start_token)
        list_sent.insert(s_end+4, s_end_token)

    inserted_sent = "".join(list_sent)
    return inserted_sent

def make_type_to_token(token_type):
    token_type = '[' + token_type + ']'
    return token_type


def return_dataloader(args, tokenizer):
    train_dataset = load_data(args.train_path)
    dev_dataset = load_data(args.dev_path)
    test_dataset = load_data(args.test_path)

    train_dataset, dev_dataset, test_dataset = apply_preprocess(
        [train_dataset, dev_dataset, test_dataset]
    )

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)
    test_label = []

    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)

    train_dataset = RE_Dataset(tokenized_train, train_label)
    val_dataset = RE_Dataset(tokenized_dev, dev_label)
    test_dataset = RE_Dataset(tokenized_test, test_label)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        # sampler=sampler, 
        num_workers=args.num_workers
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    return {
        'train':train_dataloader, 
        'val':val_dataloader, 
        'test':test_dataloader,
    }