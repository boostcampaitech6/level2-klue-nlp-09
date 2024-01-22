# 🏝 멤버 구성 및 역할

| [전현욱](https://github.com/gusdnr122997) | [곽수연](https://github.com/suyeonKwak) | [김가영](https://github.com/garongkim) | [김신우](https://github.com/kimsw9703) | [안윤주](https://github.com/nyunzoo) |
| --- | --- | --- | --- | --- |
| <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/0a2cc555-e3fc-4fb1-9c05-4c99038603b3)" width="140px" height="140px" title="Hyunwook Jeon" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/d500e824-f86d-4e72-ba59-a21337e6b5a3)" width="140px" height="140px" title="Suyeon Kwak" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/0fb3496e-d789-4368-bbac-784aeac06c89)" width="140px" height="140px" title="Gayoung Kim" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/77b3a062-9199-4d87-8f6e-70ecf42a1df3)" width="140px" height="140px" title="Shinwoo Kim" /> | <img src="https://github.com/boostcampaitech6/level1-semantictextsimilarity-nlp-01/assets/81287077/f3b42c80-7b82-4fa1-923f-0f11945570e6)" width="140px" height="140px" title="Yunju An" /> |
- **전현욱**
    - 팀 리더, ensemble 구현, 단일 모델 학습
- **곽수연**
    - Weighted Sampler 구현, 단일 모델 학습
- **김가영**
    - Loss function 실험, 단일 모델 학습
- **김신우**
    - 복합 모델 실험, K-Fold 구현, 단일 모델 학습
- **안윤주**
    - 데이터 전처리 및 증강, 단일 모델 학습

# 🍍 프로젝트 기간

2023.12.11 10:00 ~ 2023.12.21 19:00

# 🍌 프로젝트 소개

- STS(Semantic Text Similarity)란 두 텍스트가 얼마나 유사한지 판단하는 NLP Task로, 일반적으로 두 개의 문장을 입력하고 이러한 문장 쌍이 얼마나 의미적으로 서로 얼마나 유사한지를 판단하는 과제이다.
- 본 프로젝트는 주어진 데이터셋을 바탕으로 0과 5사이의 유사도 점수를 예측하는 모델을 만드는 것에 목적을 둔다.

# 🥥 프로젝트 구조

- Train Data : 9,324개
- Test Data : 1,100개
- Dev Data : 550개

## 데이터셋 구조

| Column | 설명 |
| --- | --- |
| id | 문장 고유 ID. 데이터의 이름, 버전, train/dev/test |
| source | 문장의 출처 - petition(국민청원), NSMC(네이버 영화), slack(업스테이지) |
| sentence1 | 문장 쌍의 첫번째 문장 |
| sentence2 | 문장 쌍의 두번째 문장 |
| label | 문장 쌍에 대한 유사도 (0~5, 소수점 첫번째 자리까지 표시) |
| binary-label | label이 2.5 이하인 경우는 0, 나머지는 1 |

## Label 점수 기준

| label | 설명 |
| --- | --- |
| 5 | 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함 |
| 4 | 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음 |
| 3 | 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음 |
| 2 | 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함 |
| 1 | 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음 |
| 0 | 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음 |

## 평가 지표
- **피어슨 상관 계수 PCC(Pearson Correlation Coefficient)** : 두 변수 X와 Y간의 선형 상관 관계를 계량화한 수치
- 정답을 정확하게 예측하는 것보다, 높은 값은 확실히 높게, 낮은 값은 확실히 낮게 전체적인 경향을 잘 예측하는 것이 중요하게 작용

# 🤿 사용 모델

- klue/roberta-small
- klue/roberta-large
- rurupang/roberta-base-finetuned-sts
- monologg/koelectra-base-v3-discriminator
- BM-K/KoDiffCSE-RoBERTa
- snunlp/KR-ELECTRA-discriminator

# 👒 폴더 구조

```bash
.
├── Readme.md
├── wrapup-report.pdf
└── code
    ├── KSW
    │   └── train_kfold.py
    ├── KSY
    │   ├── train
    │   │   ├── train_kfold_WRS.py
    │   │   ├── train_koelectra.py
    │   │   ├── train_test_aug.py
    │   │   ├── train_test_label.py
    │   │   ├── train_test_WeightedMSE.py
    │   │   └── train_test_WRS.py
    │   └── utils
    │       ├── data_augmentation.py
    │       ├── ensemble.py
    │       └── inference_koelectra.py
    ├── KGY
    │   ├── loss_functions.py
    │   ├── source_tagging.py
    │   └── trainMSE.py
    ├── AYJ
    │   ├── model_test_fin.py
    │   ├── model_test_fin2.py
    │   ├── inference.py
    │   ├── <soon update>
    │   └── <soon update>
    ├── JHW
    │   ├── back_translate.py
    │   ├── ensemble.py
    │   └── make_train_uniform.py
    └── final
        ├── data
        ├── fine-tuned
        ├── output
        ├── back_translate.py
        ├── ensemble.py
        ├── make_train_uniform.py
        ├── inference.py
        └── train.py
```

# 🍸 Leaderboard

|  | pearson |
| --- | --- |
| Public | 0.9218 |
| Private | 0.9311 |
