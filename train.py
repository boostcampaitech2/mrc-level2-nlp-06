import logging
import os
import sys
from utils.preprocess import prepare_datasets_with_setting
from typing import List, Callable, NoReturn, NewType, Any
import dataclasses
from datasets import load_metric, load_from_disk, Dataset, DatasetDict, Features, Value, Sequence
from utils.postprocess import post_processing_fuction_with_setting
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from utils.utils_qa import postprocess_qa_predictions, check_no_error
from utils.trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)
from wandb_arguments import WandBArguments

import wandb
wandb.login()
logger = logging.getLogger(__name__)

from datetime import datetime
from pytz import timezone
from utils.init_wandb import wandb_args_init

from bert_lstm import BERT_LSTM, BERT_QA

def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments,WandBArguments)
    )
    model_args, data_args, training_args,wandb_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)
    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)
    training_args.report_to = ["wandb"]    
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")    
    wandb_args= wandb_args_init(wandb_args, model_args)
    wandb.init(project=wandb_args.project,
                entity=wandb_args.entity,
                name=wandb_args.name,
                tags=wandb_args.tags,
                group=wandb_args.group,
                notes=wandb_args.notes)
    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    # preprocess trainset
    # trainset = datasets['train']
    # evalset = datasets['validation']
    # import pandas as pd
    # def create_df(dataset):
    #     __index_level_0__ = dataset['__index_level_0__']
    #     title = dataset['title']
    #     context = dataset['context']
    #     question = dataset['question']
    #     id = dataset['id']
    #     document_id = dataset['document_id']
    #     answers = dataset['answers']
    #     df = pd.DataFrame({
    #                         "__index_level_0__":__index_level_0__,
    #                         "title" : title, 
    #                         "context" : context,
    #                         "question" : question,
    #                         "id": id,
    #                         "document_id":document_id,
    #                         "answers":answers
    #                     })
    #     return df
    # df = create_df(trainset)
    # import re
    # def preprocess(text):
    #     text = re.sub(r'\n', ' ', text)
    #     text = re.sub(r"\\n", " ", text)
    #     text = re.sub(r'\\n\\n', ' ', text)
    #     text = re.sub(r'\n\n', " ", text)
    #     text = re.sub(r"\s+", " ", text)
    #     text = re.sub(r'#', ' ', text)
    #     return text
    # df['context'] = df['context'].map(preprocess)
    # f = Features(
    #     {
    #         "answers": Sequence(
    #             feature={
    #                 "text": Value(dtype="string", id=None),
    #                 "answer_start": Value(dtype="int32", id=None),
    #             },
    #             length=-1,
    #             id=None,
    #         ),
    #         "context": Value(dtype="string", id=None),
    #         "id": Value(dtype="string", id=None),
    #         "question": Value(dtype="string", id=None),
    #     }
    # )
    # datasets = DatasetDict({"train": Dataset.from_pandas(df, features=evalset.features), "validation":evalset})

    # print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    # print(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    model = BERT_LSTM() # 사용하려는 backbone 모델과 tokenizer 동일하게 유지해야 합니다. 현재 상태에서 argparser에 넣을 모델 이름을 backbone 모델로 주면 됩니다
    # print(model.parameters) # qa_output (1024, 2)

    # print(
    #     type(training_args),
    #     type(model_args),
    #     type(datasets),
    #     type(tokenizer),
    #     type(model),
    # )
    training_args.label_names = ["start_positions", "end_positions"]
    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:
    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    # Train preprocessing / 전처리를 진행합니다.
    dataset_list, answer_column_name = prepare_datasets_with_setting(tokenizer, datasets, training_args, data_args, max_seq_length)
    
    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
    metric = load_metric("squad")

    # prepare wandb table
    evalset = datasets['validation']
    eval_dict = {}
    for idx, data in enumerate(evalset):
        eval_dict[data['id']] = idx
    
    # wandb table
    def show_wanbd_table(p: EvalPrediction):
        table_data = []
        for idx, pred in enumerate(p.predictions):
            label = p.label_ids[idx]
            # p.prediction {'id': 'mrc-0-001960', 'prediction_text': '스위스 신문인 슈바이츠 암 존탁'},
            # p.label_id {'id': 'mrc-0-003083', 'answers': {'answer_start': [247], 'text': ['미나미 지로']}}

            # pred['prediction_cands] = [(predicted_text, score), (predicted_text, score), ...]
            table_data.append( [pred['id'], label['answers']['text'][0],\
                                pred['prediction_cands'][0][0], pred['prediction_cands'][0][1],\
                                pred['prediction_cands'][1][0], pred['prediction_cands'][1][1],\
                                pred['prediction_cands'][2][0], pred['prediction_cands'][2][1],\
                                label['answers']['answer_start'],\
                                evalset[eval_dict[pred['id']]]['context'], evalset[eval_dict[pred['id']]]['question']
                                ] )
        columns = ["id", "label_answer", "1st_pred_answer","1st_score", "2nd_pred_answer","2nd_score", "3rd_pred_answer","3rd_score","label_answer_start",  "context", "question"]
        ans_table = wandb.Table(data = table_data, columns = columns)
        wandb.log({"answer":ans_table})

    def compute_metrics(p: EvalPrediction):
        show_wanbd_table(p)
        for idx, pred in enumerate(p.predictions):
            del pred['prediction_cands']

        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    post_processing_function = post_processing_fuction_with_setting(data_args, datasets["validation"], answer_column_name)
    # Trainer 초기화
    train_dataset = dataset_list[0] if training_args.do_train else None
    eval_dataset = dataset_list[1] if training_args.do_eval else None
    trainer = QuestionAnsweringTrainer( 
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        if model_args.resume:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        else:
            train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    wandb.finish()

if __name__ == "__main__":
    main()
    #  python train.py --output_dir ./outputs/roberta-large-512 --do_train --do_eval --num_train_epochs 30 --model_name_or_path klue/bert-base --eval_steps 100 --evaluation_strategy steps --overwrite_output_dir --save_total_limit 3 --load_best_model_at_end