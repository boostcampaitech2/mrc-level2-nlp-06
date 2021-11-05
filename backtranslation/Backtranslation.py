import logging
import os
import sys
import time
from typing import List, Callable, NoReturn, NewType, Any
import dataclasses
from datasets import load_metric, load_from_disk, Dataset, DatasetDict
from transformers import (
    HfArgumentParser,
    set_seed,
)

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)
parser = HfArgumentParser(( DataTrainingArguments))
data_args = parser.parse_args_into_dataclasses()[0]
datasets = load_from_disk(data_args.dataset_name)

#Augment use pororo
# from pororo import Pororo
# trans = Pororo(task='mt', lang='multi')
#Augment use papago
# import requests
# def trans(text, src_lang, tgt_lang):
#     client_id = "0sY8o1kKKQFuLCiSB89_" # <-- client_id 
#     client_secret = "CZak0D5fRH" # <-- client_secret

#     data = {'text' : text,
#             'source' : src_lang,
#             'target': tgt_lang}

#     url = "https://openapi.naver.com/v1/papago/n2mt"

#     header = {"X-Naver-Client-Id":client_id,
#               "X-Naver-Client-Secret":client_secret}
    
#     response = requests.post(url, headers=header, data=data)
#     rescode = response.status_code

#     if(rescode==200):
#         send_data = response.json()
#         trans_data = (send_data['message']['result']['translatedText'])
#         return trans_data
#     else:
#         print("Error Code:" , rescode)

def augment(text: str, src_lang: str, tgt_lang: (list or str), module) -> str:
    if type(tgt_lang) == str:
        tgt_lang = [tgt_lang]
    
    for i in range(len(tgt_lang)):
        if i == 0:
            mid_text = module(text, tgt_lang[0])['translatedText']
        else:
            mid_text = module(mid_text, tgt_lang[i])['translatedText']    
    aug_text = module(mid_text, src_lang)['translatedText']
    return aug_text

def augment_combination(text: str, src_lang: str, tgt_lang_comb: list, module) -> list:
    return [augment(text, src_lang, tgt_lang, module) for tgt_lang in tgt_lang_comb]    
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./googlekey/key.json"
from google.cloud import translate_v2 as tr

client = tr.Client()

def augment_question(origin_question) :     
    return augment_combination(origin_question, 'ko', [['en'], ['ja'], ['zh-CN'], ['ja', 'en']], client.translate)


