import os
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from retrieval import *

def get_wiki_bm25(datasets):
    k1=1.5
    b=0.75
    epsilon=0.25
    data_path = "../"
    with open("../data/wikipedia_documents.json", "r", encoding="utf-8") as f:
            wiki = json.load(f)

    contexts = list(
                dict.fromkeys([v["text"] for v in wiki.values()])
            ) 
    model_checkpoint = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenize_fn = tokenizer.tokenize

    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, 
        is_bm25=True
    )
    
    retriever.get_sparse_embedding()
    df = retriever.retrieve(
                datasets["train"], topk=20
            )
