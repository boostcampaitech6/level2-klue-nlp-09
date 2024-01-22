from googletrans import Translator
import pandas as pd 
import tqdm
import re

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)
# string 전체 출력
pd.options.display.max_colwidth = 2000

class Google_Translator:
    def __init__(self):
        self.translator = Translator()
        self.result = {'src_text':'', 'src_lang':'', 'tgt_text':'', 'tgt_lang':''}

    def translate(self, text, lang='en'):
        translated = self.translator.translate(text, dest=lang)
        self.result['src_text'] = translated.origin
        self.result['src_lang'] = translated.src 
        self.result['tgt_text'] = translated.text
        self.result['tgt_lang'] = translated.dest 

        return self.result

    def translate_file(self, file_path, lang='en'):
        with open(file_path, 'r') as f:
            text = f.read()
        return self.translate(text, lang)

    def back_translate(self, sentence, lang1='ko', lang2='en'):
        return self.translate(self.translate(sentence,lang2)['tgt_text'],lang1)['tgt_text']


def seperate(x):
    """
    entity seperate function
    """
    entity_dict = eval(x)
    return entity_dict['word'].strip(),int(entity_dict['start_idx']),int(entity_dict['end_idx']),entity_dict['type'].strip()

def replace_sub_obj(x):
    """
    [sub]&[obj] token to entity word
    """
    pattern1 = re.compile(r'[(\[]?sub[)\]]?', re.IGNORECASE)
    pattern2 = re.compile(r'[(\[]?obj[)\]]?', re.IGNORECASE)

    result = pattern1.sub(x.subject_word, x.sentence)
    result = pattern2.sub(x.object_word, result)
    return result


if __name__=="__main__":
    
    # 데이터 불러오기
    train = pd.read_csv("/data/ephemeral/home/level2_klue/data/train.csv")

    # entity 분리
    train['subject_word'], train['subject_start_idx'], train['subject_end_idx'], train['subject_type'] = zip(*train['subject_entity'].apply(seperate))
    train['object_word'], train['object_start_idx'], train['object_end_idx'], train['object_type'] = zip(*train['object_entity'].apply(seperate))

    # 토큰화
    train_token = train.copy()
    train_token['sentence'] = train.apply(lambda x: x.sentence.replace(x.subject_word,f"[sub]").replace(x.object_word,f"[obj]"),axis=1)
    
    # translator 객체 생성
    translator = Google_Translator()

    # 번역
    back_train = train_token.copy()
    fail_ids = []

    for i, row in tqdm.tqdm_notebook(train_token.iterrows()):
        try:
            back_train.loc[i,'sentence'] = translator.back_translate(row.sentence)
            if i%400 == 0:
                print(f"Original sentence :\n{row.sentence}")
                print(f"Back Translated sentence :\n{back_train.loc[i,'sentence']}")
        except:
            print("FAIL : ",row.sentence)
            fail_ids.append(row['id'])

    back_train = back_train[~back_train['id'].isin(fail_ids)]
    back_train.to_csv("/data/ephemeral/home/level2_klue/data/back.csv",index=False)

    # 토큰 변환
    back_train['subject exist'] = back_train['sentence'].apply(lambda x: True if 'sub' in x.lower() else False)
    back_train['object exist'] = back_train['sentence'].apply(lambda x: True if 'ob' in x.lower() else False)

    df = back_train[back_train['subject exist'] & back_train['object exist']]
    df['sentence'] = df.apply(replace_sub_obj,axis=1)
    df = df.drop(columns=['subject exist','object exist'])

    # train에 BT data 합치기
    df_train = pd.concat([train,df])
    
    df.to_csv("/data/ephemeral/home/level2_klue/KSY/data/back.csv")
    df_train.to_csv("/data/ephemeral/home/level2_klue/KSY/data/train_back.csv",index=False)