import os
import json
import pandas as pd
import kss
import logging
import os
import sys
import re
import pickle
import numpy as np
from tqdm import tqdm
from pororo import Pororo
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma
from utils_qa import postprocess_qa_predictions, check_no_error
from datasets import load_metric, load_from_disk, Value, Features, Dataset, DatasetDict, Sequence
from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch
from typing import Callable, List, Dict, NoReturn, Tuple

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval
from arguments import (
    ModelArguments,
    DataTrainingArguments,
)


logger = logging.getLogger(__name__)

def main():
    pororo_tokenizer = Pororo(task='tokenize', lang='ko', model = "mecab.bpe64k.ko")
    # mecab = Mecab()의 성능 -> pororo보다 안좋다. 그냥 pororo 쓰자
    
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)
    
    training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info("Training/evaluation parameters %s", training_args)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    topk = data_args.top_k_retrieval

    # 데이터셋 생성
    es = elastic_setting(index_name = "pororo_tokenize")
    
    print('make a new dataset for elastic search !')
    dataset = load_from_disk('/opt/ml/data/test_dataset')

    # score도 구하고 싶으면 score = True
    datasets = elastic_search_retrieval(topk, es, dataset, data_args.elastic_score)
    if not os.path.isfile("/opt/ml/data/elastic_score.bin"):
        bin_save("/opt/ml/data/elastic.bin", datasets)

    print('run mrc ---- !')
    run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)
    print('finish mrc !')


def bin_save(save_path, data_set):
    file = open(save_path, "wb")
    pickle.dump(data_set, file)
    file.close()
    return None

def pororo_tokenizer(data):
    pororo_tokenizer = Pororo(task='tokenize', lang='ko', model = "mecab.bpe64k.ko")
    return pororo_tokenizer(data)

def wiki_preprocess(data):
            text = data["text"]
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r"\\n", " ", text)
            text = re.sub(r'\\n\\n', ' ', text)
            text = re.sub(r'\n\n', " ", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r'#', ' ', text)
            data["text"] = text
            return data

def dataset_preprocess():
    data_path = '/opt/ml/data/'
    context_path = "mod_wiki.json"

    if not os.path.isfile("/opt/ml/data/mod_wiki.json") :
        with open("/opt/ml/data/wikipedia_documents.json", "r") as f:
            wiki = json.load(f)
        wiki_dict = dict()
        for ids in range(len(wiki)):
            # 인덱스 번호가 string type
            wiki_dict[str(ids)] = wiki_preprocess(wiki[str(ids)])

        with open('/opt/ml/data/mod_wiki.json', 'w', encoding='utf-8') as mf:
            json.dump(wiki_dict, mf, indent="\t", ensure_ascii=False)

    with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
        wiki = json.load(f)
    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print(f"Lengths of unique contexts : {len(contexts)}")

    # train_data = load_from_disk("/opt/ml/data/train_dataset")["train"]
    # val_data = load_from_disk("/opt/ml/data/train_dataset")["validation"]

    # qa_records = [{"example_id" : train_data[i]["id"],"document_title" : train_data[i]["title"],"question_text" : train_data[i]["question"],"answer" : train_data[i]["answers"]} for i in range(len(train_data))]
    wiki_articles = [{"document_text" : contexts[i]} for i in range(len(contexts))]

    return wiki_articles


def populate_index(es_obj, index_name, evidence_corpus):
    for i, rec in enumerate(tqdm(evidence_corpus)):
        try:
            index_status = es_obj.index(index=index_name, id=i, body=rec)
        except:
            print(f'Unable to load document {i}.')
   
    n_records = es_obj.count(index=index_name)['count']
    print(f'Succesfully loaded {n_records} into {index_name}')

    return


def elastic_setting(index_name = "pororo_tokenize"):
    es_server = Popen(['/opt/ml/elasticsearch-7.9.2/bin/elasticsearch'],
                    stdout=PIPE, stderr=STDOUT,
                    preexec_fn=lambda: os.setuid(1)
                    )

    INDEX_NAME = index_name      
    # index_setting은 상황에 맞게 세팅
    # stopwords는 text파일 형태로 따로 저장되어 있음 ('/opt/ml/elasticsearch-7.9.2/config/user_dic') server의 경로의 config안에)
    INDEX_SETTINGS = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "nori_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer",
                            "decompound_mode": "mixed",
                            "filter" : ["stopwords"]
                        }
                    },
                    "filter":{
                        "stopwords": {
                            "type" : "stop",
                            "stopwords_path" : "user_dic/stopwords.txt"
                        }
                    }
                    
                }
            },
            "mappings": {
                "dynamic": "true", 
                "properties": {
                    "document_text": {"type": "text", "analyzer": "nori_analyzer"}
                    }
                }
            }

    es = Elasticsearch('localhost:9200')
    wiki_articles = dataset_preprocess()
    if not es.indices.exists(index_name):
        es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS) # 인덱스 생성
        '''
        elasticsearch.exceptions.RequestError: 
        RequestError(400, 'validation_exception', 'Validation Failed: 1: this action would add [2] total shards,
        but this cluster currently has [1000]/[1000] maximum shards open;')
        
        if this error occured, you have to delete es.indices.delete(index="Index that you want to delete index name")
        if you want to check indices in your ES to know what indices have to be deleted, this code helps you [es.indices.get_alias().keys()]

        '''
        populate_index(es_obj=es, index_name=index_name, evidence_corpus=wiki_articles) # 인덱스 내에 값 채우기
    return es


# elastic search topk 입력 후 tokenize 진행해서 prediction용 형태로 바꾸기
def elastic_search_retrieval(topk, es, dataset, score=True) :
    print(f'start {topk} document')
        
    dataset = dataset['validation']
    dataset_context = []
    dataset_question = dataset['question']
    dataset_id = dataset['id']
    
    for question in tqdm(dataset_question) :
        query = {
            'query': {
                'match': {
                    'document_text': question
                    }
                }
            }
        
        # topk만큼 score 높은 순위로 뽑아주고 그 context tokenization
        topk_docs = es.search(index='pororo_tokenize', body=query, size=topk)['hits']['hits']
        document_list = []
        for doc in topk_docs :
            temp_doc = doc['_source']['document_text']
            tokenized_doc = pororo_tokenizer(temp_doc)
            # tokenized_doc = Mecab().morphs(temp_doc)

            modified_doc = ''.join([word for word in tokenized_doc])
            # pororo tokenization 형태는 _로 나타난다. 그래서 pororo 사용할 때는 밑의 코드 추가
            modified_doc = modified_doc.replace('▁', ' ')

            document_list.append(modified_doc)
        modified_context = ' '.join(document_list)
        dataset_context.append(modified_context)

    

    # 이거 위치는 수정할 필요가 있을 듯
    if score == True: 
        dataset_score = []
        print(f'start score documents -- {topk} document')
        for i, question in tqdm(enumerate(dataset_question)) :
            query = {
            'query': {
                'match': {
                    'document_text': question
                    }
                }
            }

            docs = es.search(index='pororo_tokenize',body=query,size=topk)['hits']['hits']
            dataset_score.append([doc['_score'] for doc in docs])

    if score == True : 
        df = pd.DataFrame({'question' : dataset_question, 'id' : dataset_id, 'context' : dataset_context, 'scores' : dataset_score})

        f = Features({'context': Sequence(feature = Value(dtype='string', id=None), length = -1, id = None),
                    'id': Value(dtype='string', id=None),
                    'question': Value(dtype='string', id=None),
                    'scores' : Sequence(feature=Value(dtype='float64', id = None), length = -1, id = None)
                })
    
        score_datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})
        return score_datasets

    else : 
        df = pd.DataFrame({'id' : dataset_id, 'question' : dataset_question, 'context' : dataset_context})

        f = Features(
            {
        "context": Value(dtype="string", id=None),
        "id": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
            }
        )
        datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})

    return datasets
    




def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model):

    # only for eval or predict
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # check if there is an error
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature Creation
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: we match the start logits and end logits to answers in the original context.
    # 기본    
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset= None,
        eval_dataset=eval_dataset,
        eval_examples=datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - will create predictions.json
    if training_args.do_predict:
        predictions = trainer.predict(test_dataset=eval_dataset,
                                        test_examples=datasets['validation'])

        # predictions.json is already saved when we call postprocess_qa_predictions(). so there is no need to further use predictions.
        print("No metric can be presented because there is no correct answer given. Job done!")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


def run_mrc_score(data_args, training_args, model_args, datasets, tokenizer, model, topk=20):


    # only for eval or predict
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # check if there is an error
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    
    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        context_length = [len(cs) for cs in examples[context_column_name]]
        cumulative = [sum(context_length[:k]) for k, _ in enumerate(context_length)]
        question = [q for q, l in zip(examples[question_column_name],context_length) for _ in range(l)]
        context  = [c for cs in examples[context_column_name] for c in cs]
        
        tokenized_examples = tokenizer(
            question if pad_on_right else context,
            context if pad_on_right else question,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        tokenized_examples['ctx_rank'] = []
        tokenized_examples['scores'] = []
        
        on = 0
        cumulative.append(sum(context_length))
        # doc score 추가!
        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            while cumulative[on+1] <= sample_mapping[i] :
                on += 1
            sample_index = on # sample_mapping[i] // topk
            rank_index = sample_mapping[i] - cumulative[on] # sample_mapping[i] % topk
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples['ctx_rank'].append(rank_index)
            tokenized_examples['scores'].append(examples["scores"][sample_index][rank_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature Creation
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:

        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset= None,
        eval_dataset=eval_dataset,
        eval_examples=datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - will create predictions.json
    if training_args.do_predict:
        predictions = trainer.predict(test_dataset=eval_dataset,
                                        test_examples=datasets['validation'])

        # predictions.json is already saved when we call postprocess_qa_predictions(). so there is no need to further use predictions.
        print("No metric can be presented because there is no correct answer given. Job done!")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

if __name__ == "__main__":
    main()
