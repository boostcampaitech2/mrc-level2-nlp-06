<h1 align="center">
BoostCamp AI Tech - [NLP] ODQA
</h1>

 ###### tags: `MRC` `ODQA`
<p align="center">
    <a href="https://boostcamp.connect.or.kr/program_ai.html">
        <img src="https://img.shields.io/badge/BoostCamp-P--Stage-bronze?logo=Naver&logoColor=white"/></a> &nbsp
    </a>
    <a href="https://github.com/KLUE-benchmark/KLUE">
        <img src="https://img.shields.io/badge/Dataset-KLUE--MRC-critical?logo=GitHub&logoColor=white"/></a> &nbsp
    </a>
    <a href="https://huggingface.co/klue/roberta-large">
        <img src="https://img.shields.io/badge/KLUE-roberta--large-yellow"/></a> &nbsp
    </a>
    <a href="https://stackoverflow.com/questions/52229059/em-score-in-squad-challenge">
        <img src="https://img.shields.io/badge/Score (EM)-64.170-bronze"/></a> &nbsp
    </a>
    <a href="https://stackoverflow.com/questions/52229059/em-score-in-squad-challenge">
        <img src="https://img.shields.io/badge/Score (F1 score)-74.690
-bronze"/></a> &nbsp
    </a>
</p>


## Table of Contents
1. [Project Overview](#1.-Project-Overview)
2. [Usage](#2.-Usage)
    - [Train](#Train)
    - [Eval](#Evaluation)
    - [Inference](#Inference)
    - [Inference through Elastic Search](#Inference-through-Elastic-Search)   
3. [Experiment](#3.-Experiment)
4. [Result](#4.-Result)
5. [Things to know](#5.-Things-to-know)
6. [Contributors](#6.-Contributors)

<br/>

## 1. Project Overview 
#### Goals
 - Retriever Task와 Reader Task를 구성하고 통합하여, 질문을 던졌을 때 답변을 해주는 ODQA 시스템 개발
- Retriever
    - 방대한 Open Domain Dataset에서 질의에 알맞은 지문을 찾아오는 Task
- Machine Reading Comprehension(MRC) 
    - 지문이 주어진 상황에서 질의에 대해 응답하는 기계 독해 Task   
- Open-Domain Question Answering(ODQA)
    - Retriever 와 MRC Task를 결합한 시스템

#### Dataset
아래는 제공하는 데이터셋의 분포를 보여준다.
![데이터셋 분포](https://i.imgur.com/UMmlsAh.png)


데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 구성

#### Evaluation Metrics 
- EM(Exact Match)
    - 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어진다.
    - 답이 하나가 아닌 경우는 하나라도 일치하면 정답으로 간주한다.
    ![](https://i.imgur.com/KmzSmep.png)

- F1 score
    ![](https://i.imgur.com/O6lWzA9.png)


#### Directory Structure
```python
.
mrc-level2-nlp-06
|-- arguments.py -> 학습 관련 설정
|-- dense_retrieval.py -> retriever 모델 학습
|-- inference.py
|-- elastic_inference.py
|-- install
|   `-- install_requirements.sh
|-- retrieval.py
|-- train.py -> reader 모델 학습
|-- trainer_qa.py
|-- utils_qa.py
|-- utils
|   |-- init_wandb.py
|   |-- postprocess.py
|   |-- preprocess.py
|   |-- trainer_qa.py
|   |-- utils_dpr.py
|   `-- utils_qa.py
`-- wandb_arguments.py -> wandb 설정


```
data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능하다. 


<br/>

## 2. Usage
### Install Requirements
```
bash install_requirements.sh
```
### Train
 - Custom Hyperparameter : `arguments.py` 참고
 - Using roberta Model : tokenizer 사용시 아래 함수의 옵션을 수정     
     - tokenizer의 return_token_type_ids=False로 설정

```
# Retriever 학습 예시 # 내부에서 변경
python dense_retriver.py 
```
```
# Reader 학습 예시 (train_dataset 사용) 
python train.py --output_dir ./models/train_dataset --do_train
```

### Evaluation

 - MRC 모델의 평가 : `--do_eval`사용
 - 위의 예시에 `--do_eval` 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있다.

```
# mrc 모델 평가 (train_dataset 사용)
python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
```

### Inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행한다.

* 학습한 모델의 test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행한다. 
* 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(--do_eval)를 진행한다.

```
# ODQA 실행 (test_dataset 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장되고 아니면 단순히 출력된다.
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict
```
- DataTrainingArguments

```
dpr:bool - 학습된 dpr encoder와 bm25를 정한 비율로 결합해서 inference
dpr_negative:bool - dense_retriver.py에 있는 코드를 통해 학습된 모델로 inference
```

### Inference through Elastic Search
* 학습한 모델의 test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다. 위와 과정은 동일하고 실행 파일만 변경하면 됩니다. (elastic search를 통한 retrieval 구성이 포함되어 있는 파일입니다.)
* elastic search 사용을 위해 현재 버전으로 저장되어 있는 폴더 내에 user_dic을 만들고 안에 stopwords.txt를 넣어야 한다.
 ```
# ODQA 실행 (test_dataset 사용)
python elastic_inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict
```
- Data Training Arguments

```
elastic_score: bool = field(
        default = True,
        metadata={"help": "whether to score to top_k of each query"}
    )

# elastic search로 context만 뽑아내고 싶다면 False
# score도 함께 뽑아내고 싶다면 True 

```

### How to submit

`inference.py` 파일을 위 예시처럼 `--do_predict` 으로 실행하면 `--output_dir` 위치에 `predictions.json` 이라는 파일이 생성되고 해당 파일을 제출한다.

<br/>

## 3. Experiment

#### Data Preprocessing
- 동일한 문장의 데이터가 42개, 동일하면서도 라벨이 달랐던 데이터가 5개 존재했고 둘 중 올바른 라벨로 수정하였다.
- `preprocess_wiki_documents(contexts)`
    - 들어온 contexts 중에 한글이 포함되지 않거나, 한글이 포함되었지만 code로 구성된 context, 그리고 특정 글자를 포함하는 contexts를 제거하였고, 남은 contexts 에서 특정 문자, 또는 특정 문자를 포함하는 문장을 제거하였다.
    - `use_wiki_preprocessing=True` 인 경우 해당 함수를 사용할 수 있다.

#### Retriver
 - DPR retriever
    - TF-IDF , BM25 기반 Negative Sample: wiki-context와 mrc-context를 내적하는 방법으로 접근했다.
    - In-Batch: query와 positive sentence로 이루어진 미니 배치 내부에서 다른 example들 사이에서 내적을 통한 유사도에서 가장 높은 유사도에 해당하는 example으로 prediction한다.
    - 단일 DPR을 활용하기 보단 BM25의 점수를 선형결합해서 단일 모델 중 가증 큰 성능을 보였다. 
    - BM25 단일모델 EM :  55 -> BM25+DPR EM : 61.240


 - BM25 retriever
     - Okapi bm25 github의 라이브러리를 상속해서 구현했다.
     - bm25, bm25L, bm25Plus, 다양한 토크나이저 실험 중 성능이 가장 좋은 bm25와 klue/bert-base 토크나이저를 사용하여 비교했다.
     - retrieval accuracy: top 1: 0.5875,   top 10: 0.9041,   top 20: 0.9208


 - Elasticsearch retriever
     - Okapi bm25의 성능이 task에서 높은 것을 파악하고 빠르게 동작시키기 위해 사용했다.
     - python 내에서 돌아가도록 Elastic search의 index setting을 구성했다.
     - stopwords filter와 pororo 라이브러리 기반의 tokenizing을 통해 context를 분석했다.


#### Reader
- KLUE/Roberta-large
    - 다른 기학습 가중치를 활용해보았으나 KLUE 데이터셋을 기반으로 전이 학습된 모델이 평균적으로 더 높은 성능을 보였다.


<br/>

## 4. Result
#### Retriver 결과
![Retriver 결과](https://i.imgur.com/9FiZuBU.png)

#### MRC 결과(train)
- roberta-large, bert-base, xlm-roberta-large
![](https://i.imgur.com/SIxlX1I.png)



#### ODQA 결과(LB 점수)
- Ensemble
    - 최고성능 : Hard Voting
        ![](https://i.imgur.com/lHrzmgQ.png)
    - Elastic : Hard Voting
        ![](https://i.imgur.com/rGVGmQ5.png)

- 성능이 향상된 방법
    - 단일 모델 최고성능 : Elastic Search
        ![](https://i.imgur.com/TVS0ah4.png)
    - Preprocess
        ![](https://i.imgur.com/nFSHQFP.png)
        ![](https://i.imgur.com/GVtFZ26.png)
    - BM25-Plus topk20
        ![](https://i.imgur.com/fpHguzM.png)
    - BM25 topk20
        ![](https://i.imgur.com/e6y4vMP.png)
- Baseline
    - topk20 TF-IDF
        ![](https://i.imgur.com/Y4nq651.png)
    - topk10 TF-IDF
        ![](https://i.imgur.com/8HBDPcJ.png)







<br/>

## 5. Things to know

1. `train.py`에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있다.
실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 된다.
**만약 sparse retrieval 관련 코드를 수정 시 존재하는 파일이 load되지 않도록 두 파일을 지우고 다시 실행한다**
2. 모델의 경우 `--overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않는다. 

3. ./outputs/ 폴더 또한 `--overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않는다.

<br/>

## 6. Contributors
나요한_T2073 : https://github.com/nudago
백재형_T2102 : https://github.com/BaekTree
송민재_T2116 : https://github.com/Jjackson-dev
이호영_T2177 : https://github.com/hylee-250
정찬미_T2207 : https://github.com/ChanMiJung
한진_T2237 : https://github.com/wlsl8135
홍석진_T2243 : https://github.com/HongCu

