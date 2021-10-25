import logging
import os
import sys

from typing import List, Callable, NoReturn, NewType, Any
import dataclasses
from datasets import load_metric, load_from_disk, Dataset, DatasetDict

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

from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)
from wandb_arguments import WandBArguments
logger = logging.getLogger(__name__)
from datetime import datetime
from pytz import timezone
parser = HfArgumentParser(( DataTrainingArguments))
data_args = parser.parse_args_into_dataclasses()[0]
datasets = load_from_disk(data_args.dataset_name)

# question_token_dict ={
#     'who' : ['누구', '인물은?', '이는?', '여자는?', '남자는?', '사람은?', '정체는?', '선수는?', '작곡가는?', '경쟁자는?', '사람들은?', '당사자는?', '선임자는?', '근무지는?', '이름은?'],        
#     'why' : ['왜', '까닭은?', '이유는?', '때문'],
#     'where' : ['어디', '출신', '지역은?', '장소는?', '곳은?', '국가는?', '곳은?', '나라는?', '학교는?', '도시는?', '대학은?', '원산지는?'], 
#     'what' : ['것은?', '무엇인가?'], 
#     'how': ['어떻', '방식은', '어떠한'],
#     'when': ['시기는', '몇 년', '해는','언제부터','언제까지', '날짜', '몇 월', '때 는', '날은?', '달은?', '기간은?', '어느', '연도는?', '년도는?', '시간은?'],  
# }
# count_question = {
#     'who' : 0,
#     'why' : 0,
#     'where' : 0,
#     'what' : 0,
#     'how' : 0,
#     'when' : 0,    
# }
# class Question :
#     def __init__(self, question) -> None:
#        self.question = question
#        self.tags=[]
#     def get_question(self):
#         return self.question
#     def set_tag(self, tag) :
#         self.tags.append(tag)
#     def get_tags(self):
#         return self.tags
#     def get_tags_len(self):
#         return len(self.tags)
    
        
    
# question_token = ['who', 'why', 'where', 'what', 'how', 'when']
# question_list = [] 
# for dataset in datasets['train']:
#     temp = Question(dataset['question'])
#     for token in question_token :         
#         if any(token in dataset['question'] for token in question_token_dict[token]) :            
#             temp.set_tag(token)
#     question_list.append(temp)            
# for question in question_list:
#     if question.get_tags_len() ==0:
#         count_question['what']+=1
#     else : 
#         for token in question_token :         
#             if token in question.get_tags():
#                 count_question[token]+=1
        
    

# print(count_question)


#################################### EDA
'''
이 코딩대로 수행한결과, 멀티 태그가 붙은 갯수는 10개
총 질의 count =3952
{'who': 868, 'why': 83, 'where': 484, 'what': 2025, 'how': 30, 'when': 477}
'''
count=0
for dataset in datasets['train'] : 
    if dataset['answers']['text'][0] in dataset['context']:
        continue
    count+=1
print(count)