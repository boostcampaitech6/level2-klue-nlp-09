# 🏝 멤버 구성 및 역할

| [전현욱](https://github.com/gusdnr122997) | [곽수연](https://github.com/suyeonKwak) | [김가영](https://github.com/garongkim) | [김신우](https://github.com/kimsw9703) | [안윤주](https://github.com/nyunzoo) |
| --- | --- | --- | --- | --- |
| <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/0a2cc555-e3fc-4fb1-9c05-4c99038603b3)" width="140px" height="140px" title="Hyunwook Jeon" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/d500e824-f86d-4e72-ba59-a21337e6b5a3)" width="140px" height="140px" title="Suyeon Kwak" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/0fb3496e-d789-4368-bbac-784aeac06c89)" width="140px" height="140px" title="Gayoung Kim" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/77b3a062-9199-4d87-8f6e-70ecf42a1df3)" width="140px" height="140px" title="Shinwoo Kim" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/f3b42c80-7b82-4fa1-923f-0f11945570e6)" width="140px" height="140px" title="Yunju An" /> |
- **전현욱**
    - 팀 리더, Ensemble 구현, torch 모델 구현, 단일 모델 학습
- **곽수연**
    - 데이터 전처리 및 증강, 단일 모델 학습
- **김가영**
    - Entity Tagging 실험, Prompt 실험, 단일 모델 학습
- **김신우**
    - Rule-based 모델 구현, Entity Tagging 실험, 단일 모델 학습
- **안윤주**
    - PM, 데이터 전처리 및 증강, 단일 모델 학습

# 🍍 프로젝트 기간

2024.01.03 10:00 ~ 2024.01.18 19:00

![image](https://github.com/boostcampaitech6/level2-klue-nlp-09/assets/81287077/d52733b3-4f59-48ea-a30a-9a4e14209357)


# 🍌 프로젝트 소개

- 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 NLP Task 로, 비구조적인 자연어 문장에서 구조적인 triple 을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있다. 
- 본 프로젝트는 주어진 데이터셋을 바탕으로 문장 내 두 단어의 관계를 30 개의 관계 Label 에 대한 예측 확률을 추론하는 모델을 만드는 것에 목적을 둔다.

# 🥥 프로젝트 구조

- Train Data : 32,470개
- Test Data : 7,765개

## 데이터셋 구조

| Column | 설명 |
| --- | --- |
| id | 샘플 순서 ID |
| sentence | 관계 추출을 위한 단어들을 포함한 문장 |
| subject_entity | Subject Entity 에 대한 정보(단어, 시작 인덱스, 끝 인덱스, 타입) |
| object_entity | Object Entity 에 대한 정보(단어, 시작 인덱스, 끝 인덱스, 타입) |
| label | 두 Entity 사이의 관계 (30 개의 Label) |
| source | 샘플 출처 |

## Label Class 기준

![image](https://github.com/boostcampaitech6/level2-klue-nlp-09/assets/81287077/a0fdac21-c136-4700-9ab9-e13a5541508f)


## 평가 지표
- **micro F1 score** : no_relation class 를 제외한 f1 score
- area under the precision-recall curve (AUPRC) : 불균형 데이터에 대한 precision-recall score

# 🤿 사용 모델

- klue/roberta-large
- monologg/koelectra-base-v3-discriminator
- BM-K/KoDiffCSE-RoBERTa
- nlpotato/roberta_large-ssm_wiki_e2-origin_added_korquad_e5
- xlm-roberta-large
- soddokayo/klue-roberta-large-klue-ner
- sdadas/xlm-roberta-large-twitter
- severinsimmler/xlm-roberta-longformer-large-16384

# 👒 폴더 구조

```bash
.
├── EDA.ipynb
├── README.md
├── Wrap-up Report.pdf
├── data_aug
│   ├── back_translation.py
│   ├── data_augmenation_EDA.ipynb
│   ├── kogpt3_test.py
│   └── kullm_test.py
├── entity_tagging
│   ├── Prompt.py
│   └── typed_entity_punct.py
├── huggingface_trainer
│   ├── inference.py
│   ├── load_data.py
│   └── train.py
├── rule_based
│   ├── inference.py
│   ├── load_data.py
│   └── train.py
├── soft_vote.ipynb
├── soft_vote.py
├── torch-train
│   ├── data_handling.py
│   ├── data_handling_tagging.py
│   ├── inference.py
│   ├── modeling.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
└── train_validation_split.ipynb
```

# 🍸 Leaderboard

|  | micro F1-score | AUPRC |
| --- | --- | --- |
| Public | 76.3116 | 81.1209 |
| Private | 74.0375 | 81.1955 |
