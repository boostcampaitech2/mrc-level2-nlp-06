import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from pprint import pprint
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from elasticsearch import Elasticsearch, helpers

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)

from typing import List, Tuple, NoReturn, Any, Optional, Union



import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
    HfArgumentParser,
)

from datasets import load_dataset

from transformers import AutoTokenizer
from wandb_arguments import WandBArguments

import logging
logging.basicConfig(filename='wiki_night.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

import wandb
wandb.login()


from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

DPR_PATH = "../dpr_output"
WIKI_EMB_PATH = "wiki_dense_emb"
ALL_EMB_PATH = "all_dense_emb"
WIKI_SET_DIR = "../data/wikipedia_documents.json"
from contextlib import contextmanager

try:
    es.transport.close()
except:
    pass
es = Elasticsearch()

INDEX_NAME = "toy_index"

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)
    


class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
      
    def forward(self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output


# Ground Truth와 똑같거나 비슷한 Passage가 출력됐나요? 역시 Pretrained는 강력하군요.
# ### ❓ Dense Retrieval 코드를 class로 합쳐봅시다.
# 현재 구현된 모델은 굉장히 단순히 짜여졌습니다. 이미 Pretrain된 BERT 모델을 활용했는데, 다른 PLM은 어떨까요? 이를 다시 시도해보자니 코드가 너무 셀로 흩어져있어서 재사용이 어렵네요. 우선 클래스화를 진햏한 후에, `BERT`가 아닌 다른 PLM도 활용해봅시다.

# ❗ _Hint_   
# Scratch부터 짜는게 많이 어려우신가요?
# 
# 만들어야하는 기능들의 파이프라인을 나열하고 하나씩 모듈화하는 습관을 들이는 것이 좋습니다. 순서대로 생각해볼까요?
# 
# 1. Setup   
#     Naive한 `Dataset`을 받아서 이를 In-Batch Negative를 활용한 후 Dataloader로 변경해주는 코드가 있어야겠죠? 클래스 내에서 활용할 수 있도록 속성(attribute)으로 만들어줍시다. 이 코드를 위에서 활용한 `train` 함수에서 조금 차용해볼까요?
# 2. PLM을 주어진 Passage 와 In-batch negative 기법을 활용해서 훈련합니다.   
#     이는 위에서 만든 `train` 함수를 약간 응용해서 재활용합시다.
# 3. 훈련한 PLM 을 통해 Query 를 Transform 합니다.
# 4. 내적을 통해 유사도를 구하고, 내림차순을 통해 유사한 Passage 를 Retrieval 합니다.
# 
# 위 4단계를 하나씩 함수로 구현한 후 Class로 합치면 훨씬 쉬울 겁니다!

# In[66]:


# 코드가 많아보이지만 주석이 더 많지롱
import tqdm.notebook as tq
# for i in tq.tqdm(...):




"""
Use Case
1. make instance
   use
        dpr = DenseRetrieval(
                args (Huggingface Arguments):
                dataset (datasets.Dataset):
                num_neg (int):
                tokenizer (Callable):
                p_encoder (torch.nn.Module):
                q_encoder (torhc.nn.Module):
                data_path: Optional[str] = "../data/",
                context_path: Optional[str] = "wikipedia_documents.json",
             )

2. train
    use
        dpr.training_encoders(is_save_p_enc = True, is_save_q_enc = True) -> None
        
    desription
        train then save them -> to use them in validatoin and inference.
        
3. embedding wiki
    use
        dpr.embedding_wiki(use_pre_encoding = "last")

    description
        embedding wiki data with encoder and decoder.
        No need to train all the time. Once trained, just bring them and create new ret, then embed!
        use_pre_encoding: use last pre_encoding version.
        

4. get relevant doc
    use
        dpr.get_relevant_doc(query:str, k:int) -> return doc_scores, doc_indices
        dpr.get_relevant_doc_bulk(queriesList[str], k:int) -> return doc_scores, doc_indices
    decription
        same as sparse retrieval. need to call embedding wiki in advance. 
5. evaluate
    use
        dpr.evaluate(context = "eval")
    description
        evaluate the performance. may chooise validation set or wiki.


Class DenseRetrival
feature:
    __init__
    training_encoders
    embedding_wiki
    get_relevant_doc
    get_relevant_doc_bulk

    prepare_dataset:
    prepare_in_batch_negative: put negative inputs in batch. Do this before creating dataset.

    encoding_passage(): used in embedding_wiki. Do we need this anyway?
    encoding_query(): used in get_relevant_doc. need to do this to calculate sim between embedding.


use case1: retrieval(p_enc, q_enc) -> ret.get_relevant_doc(...). assert both not None.
use case2: ret(p_enc, q_enc) -> ret.trainig_encoders. assert both not None.
"""


class DenseRetrieval:
    def __init__(self,
        args = None,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        num_neg=2,
        tokenizer = None,
        dataset = None, # dataset to train and validation. must have qeustion and passage pair.
        evalset = None,
        p_encoder = None,
        q_encoder = None,
        prediction_only = False,
        pre_encode_psg = None
    ):
        """
        Arguments:
            args (Huggingface Arguments):
                세팅과 학습에 필요한 설정값을 받습니다.
            dataset (datasets.Dataset):
                Huggingface의 Dataset을 받아옵니다.
            num_neg (int):
                In-batch negative 수행시 사용할 negative sample의 수를 받아옵니다.
            tokenizer (Callable):
                Tokenize할 함수를 받아옵니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab
            p_encoder (torch.nn.Module):
                Passage를 Dense Representation으로 임베딩시킬 모델입니다.
            q_encoder (torhc.nn.Module):
                Query를 Dense Representation으로 임베딩시킬 모델입니다.

        Summary:
            학습과 추론에 필요한 객체들을 받아서 속성으로 저장합니다.
            객체가 instantiate될 때 in-batch negative가 생긴 데이터를 만들도록 함수를 수행합니다.
        """
        if pre_encode_psg:
            self.emb_file = pre_encode_psg
        self.num_neg = num_neg
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        # print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))


        self.tokenizer = tokenizer


        # train 함수에 보내도 되는 것들
        self.args = args
        self.dataset = dataset
        self.evalset = evalset
        if p_encoder:
            self.p_encoder = p_encoder.to(args.device)
        self.q_encoder = q_encoder.to(args.device)
        self.dataloader = None
        if not prediction_only:
            self.prepare_in_batch_negative(num_neg=num_neg)

        self.p_emb = None

    def prepare_in_batch_negative(self,
        dataset=None,
        num_neg=2,
        tokenizer=None
    ):
        """
        Arguments:
            dataset (datasets.Dataset, default=None):
                Huggingface의 Dataset을 받아오면,
                in-batch negative를 추가해서 Dataloader를 만들어주세요.
            num_neg (int, default=2):
                In-batch negative 수행시 사용할 negative sample의 수를 받아옵니다.
            tokenizer (Callable, default=None):
                Tokenize할 함수를 받아옵니다.
                별도로 받아오지 않으면 속성으로 저장된 Tokenizer를 불러올 수 있게 짜주세요.

        Note:
            모든 Arguments는 사실 이 클래스의 속성으로 보관되어 있기 때문에
            별도로 Argument를 직접 받지 않아도 수행할 수 있게 만들어주세요.
        """
        if dataset is None:
            dataset = self.dataset
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        # given dataset: read data. huggingface dataset. input: dataset
        # output: 6개의 tensor in tuple in one example.
        # corpus = list(set(dataset['context']))
        corpus =dataset['context'] # query <-> context 1:1 대응 관계
        # data_size = len(corpus)
        # create negative of num_neg for each example
        # p_with_neg = []
        # corpus = np.array(corpus)
        # for c in corpus:
        #     while True:
        #         idx = np.random.choice(len(corpus), num_neg)
        #         if c not in corpus[idx]:
        #             p_with_neg.append(c)
        #             p_with_neg.extend(corpus[idx])
        #             break
        # p_woth_neg # (batch * 3)        
        q = dataset['question'] # (batch)

        p_seg = tokenizer(corpus,                                     padding = "max_length",                                    return_tensors = "pt",                                    truncation = True
                                    )
        q_seg = tokenizer(q, 
                                    padding = "max_length",\
                                    return_tensors = "pt",\
                                    truncation = True
                                    )
        # p_seg # (batch * 3, max_len)
        # q_seg # (batch, max_len)
        # 배치 크기가 달라서 dataloader에 못넣을 듯. batch을 맞춰줘야 한다.
        # max_len = p_seg['input_ids'].size(-1) # max_len 구하기 위해서
        # p_seg['input_ids'] = p_seg['input_ids'].view(-1, num_neg + 1, max_len)
        # p_seg['attention_mask'] = p_seg['attention_mask'].view(-1, num_neg + 1, max_len) # (-1, num_neg + 1, -1) -> only one dim can be inferred.
        # p_seg['token_type_ids'] = p_seg['token_type_ids'].view(-1, num_neg + 1, max_len)
        # print(p_seg['input_ids'].size())
        # print(q_seg['input_ids'].size())
        dataset = TensorDataset(p_seg['input_ids'], p_seg['attention_mask'],   p_seg['token_type_ids'],                                q_seg['input_ids'], q_seg['attention_mask'],   q_seg['token_type_ids'])
        self.dataloader = DataLoader(dataset, batch_size=self.args.per_device_train_batch_size)

    def train(self,
        logger = None,
        args=None,
        partial = None,
        eval_set = ["eval", "tr_and_ev", "wiki"]
    ):
        """
        Summary:
            train을 합니다. 위에 과제에서 이용한 코드를 활용합시다.
            encoder들과 dataloader가 속성으로 저장되어있는 점에 유의해주세요.
        """
        def log(log_msg):
            if logger:
                logger.log(log_msg)

        if isinstance(eval_set, str):
            eval_set = [eval_set]
        assert "tr" in eval_set  or "eval" in eval_set or "tr_and_ev" in eval_set or "wiki" in eval_set, f"You put {eval_set}. at least one of eval, tr_and_ev, wiki must  be in eval_set argument."

        # pass
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder

        if args == None:
            args = self.args
        num_neg = self.num_neg
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(self.dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )
        # wandb.watch(self.p_encoder, log_freq=1)
        # wandb.watch(self.q_encoder, log_freq=1)
        min_loss = float("inf")
        topk_list = [1,5,10,20, 30, 50]
        best_acc = [float('-inf')] * len(topk_list)
        for e in range(self.args.num_train_epochs):
            batch_size = self.args.per_device_train_batch_size
            print(f"epoch {e} of {self.args.num_train_epochs}")
            logging.info(f"epoch {e} of {self.args.num_train_epochs}")
            epoch_loss = 0
            for batch in tqdm(self.dataloader):
                self.p_encoder.train()
                self.q_encoder.train()

                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()
                
                # targets = torch.zeros(batch_size).to(args.device, dtype=torch.int64)
                # p_input = {
                #     "input_ids":batch[0].view(batch_size * (num_neg + 1), -1).to(args.device), 
                #     "attention_mask":batch[1].view(batch_size * (num_neg + 1), -1).to(args.device), 
                #     "token_type_ids":batch[2].view(batch_size * (num_neg + 1), -1).to(args.device)
                # } # b * 3, max_len
                p_input = {
                    "input_ids":batch[0].to(args.device), 
                    "attention_mask":batch[1].to(args.device), 
                    "token_type_ids":batch[2].to(args.device)
                }# b, max_len
                q_input = {
                    "input_ids":batch[3].to(args.device), 
                    "attention_mask":batch[4].to(args.device), 
                    "token_type_ids":batch[5].to(args.device)
                }# b, max_len

                # 마지막 배치에 batch size보다 작으면 에러 나는것을 방지. 마지막 한방울까지 모델에 먹입시다!!!
                num_batch_size = args.per_device_train_batch_size
                if len(batch[0]) < args.per_device_train_batch_size:
                    num_batch_size = len(batch[0])
            
                del batch

                targets = torch.arange(0, num_batch_size).to(args.device, dtype=torch.int64)#.long()
                # pasage : (m_b, num_neg, max_len) -> (m_b * num_neg, max_len) -> bert -> (m_b * num_neg, max_len, emb) -> cls -> (m_b * num_neg, emb) -> (m_b, num_neg, emb)
                # query: (m_b, max_len) -> bert -> (m_b, max_len, emb) -> cls -> (m_b, emb) -> view -> (m_b, 1, emb) -> transpose ->  (m_b, emb, 1)
                # bmm: (m_b, num_neg, 1) -> squeeze -> (m_b, num_neg) -> softmax -> (m_b, num_neg)
                # loss: target은 항상 0이다. 그리고 맨 위에 정답 passage가 있음. 

                # print(f"p_input size: {p_input['input_ids'].size()}")
                # print(f"q_input size: {q_input['input_ids'].size()}")

                p_emb = self.p_encoder(**p_input) # in hidden state, b, 3, max_len, emb
                q_emb = self.q_encoder(**q_input)# in hidden state, b, max_len, emb
                # print("original passage bert output size : ", p_output.size())
                # p_emb = p_output.view(batch_size, num_neg + 1, -1)#[1] # b, 3, emb
                # print(p_emb.size())
                # q_emb = q_output#[1] # b, emb
                # q_emb = q_emb.view(batch_size, -1, 1) # b, emb, 1
                # print(q_emb.size())
                # sim_scores = torch.bmm(p_emb, q_emb).squeeze() # b, 3, 1 -> b, 3
                # print(f"p_emb.T shape = {p_emb.T.size()}")
                # print(f"q_emb shape = {q_emb.size()}")
                sim_scores = torch.matmul(q_emb, p_emb.T )
                # print(sim_scores.size())
                sim_scores = F.log_softmax(sim_scores, dim = 1)

                loss = F.nll_loss(sim_scores, targets)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

                self.q_encoder.zero_grad()
                self.p_encoder.zero_grad()

                del p_input, q_input
                torch.cuda.empty_cache()

            epoch_loss = loss.item() / len(self.dataloader)
            print(f"{e} epoch loss:{epoch_loss}")
            logging.info(f"{e} epoch loss:{epoch_loss}")
            log({"train_loss": epoch_loss, "epoch" : e})
            if min_loss > epoch_loss:
                min_loss = epoch_loss
                print(f"best model in train loss update acc {epoch_loss}")
                logging.info(f"best model in train loss update acc {epoch_loss}")
                torch.save(self.p_encoder.state_dict(), "../dpr_output/best_p_enc_model_train_loss.pt")
                torch.save(self.q_encoder.state_dict(), "../dpr_output/best_q_enc_model_train_loss.pt")

            # evaluation
            tmp_context = "tr"
            if tmp_context in eval_set:
                new_acc = [float("-inf")] * len(topk_list)
                new_acc = self.evaluate(search_context = tmp_context, topk = topk_list, new_acc = new_acc, partial = partial)
                for acc_idx, n_a in enumerate(new_acc):
                    log({f"{tmp_context}_accuracy_top{topk_list[acc_idx]}": n_a})
                    print(f"{tmp_context}_accuracy_top{topk_list[acc_idx]}: {n_a}")
                    logging.info(f"{tmp_context}_accuracy_top{topk_list[acc_idx]}: {n_a}")                    

            tmp_context = "eval"
            if tmp_context in eval_set:
                new_acc = [float("-inf")] * len(topk_list)
                new_acc = self.evaluate(search_context = tmp_context, topk = topk_list, new_acc = new_acc, partial = partial)
                for acc_idx, n_a in enumerate(new_acc):
                    log({f"{tmp_context}_accuracy_top{topk_list[acc_idx]}": n_a})
                    print(f"{tmp_context}_accuracy_top{topk_list[acc_idx]}: {n_a}")
                    logging.info(f"{tmp_context}_accuracy_top{topk_list[acc_idx]}: {n_a}")

            tmp_context = "tr_and_ev"
            if tmp_context in eval_set:
                new_acc = [float("-inf")] * len(topk_list)
                new_acc = self.evaluate(search_context = tmp_context, topk = topk_list, new_acc = new_acc, partial = partial)
                for acc_idx, n_a in enumerate(new_acc):
                    log({f"{tmp_context}_accuracy_top{topk_list[acc_idx]}": n_a})
                    print(f"{tmp_context}_accuracy_top{topk_list[acc_idx]}: {n_a}")
                    logging.info(f"{tmp_context}_accuracy_top{topk_list[acc_idx]}: {n_a}")

            tmp_context = "wiki"
            if tmp_context in eval_set:
                new_acc = [float("-inf")] * len(topk_list)
                new_acc = self.evaluate(search_context = tmp_context, topk = topk_list, new_acc = new_acc, partial = partial)

                for acc_idx, n_a in enumerate(new_acc):
                    logging.info(f"{tmp_context}_accuracy_top{topk_list[acc_idx]}: {n_a}")
                    print(f"{tmp_context}_accuracy_top{topk_list[acc_idx]}: {n_a}")
                    log({f"{tmp_context}_accuracy_top{topk_list[acc_idx]}": n_a})
                    if best_acc[acc_idx] < n_a:
                        best_acc[acc_idx] = n_a
                        print(f"best model in acc of top{topk_list[acc_idx]}_acc_{n_a} update")
                        logging.info(f"best model in acc of top{topk_list[acc_idx]}_acc_{n_a} update")
                        torch.save(self.p_encoder.state_dict(), f"../dpr_output/best_p_enc_model_top{topk_list[acc_idx]}.pt")
                        torch.save(self.q_encoder.state_dict(), f"../dpr_output/best_q_enc_model_top{topk_list[acc_idx]}.pt")   
                        torch.save(self.p_emb, f'../dpr_output/{tmp_context}_dense_emb/passage_embedding_top{topk_list[acc_idx]}.pt')

            
        print("train finishied.")
        print("model saved.")


    def loading_embedding(self, path, emb_kind):
        print(f"Loading {emb_kind} embedding...")
        if emb_kind != "wiki":
            raise NotImplementedError()
        # emb_list = []
        # files = os.listdir(path) # dir is your directory path
        # num_of_file = len(files)
        # print(num_of_file)
        emb = torch.load(path +"/"+ self.emb_file)
        print(f"{emb_kind} embedding loaded.")
        return emb
        # for i in range(num_of_file): 
        #     emb = torch.load(path + "/passage_embedding" + str(i) + "out_of_" + str(num_of_file) + ".pt")
        #     emb_list.append( emb )
        # return torch.cat(emb_list)

    def load_wikiset(self):
        with open(WIKI_SET_DIR, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        wikiset = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )
        return wikiset

    def get_dense_embedding(self, search_context = "wiki", custom = None, partial = None, force_encode = False):
        # raise NotImplementedError()
        ## in evaluate should not use pre encoded embeddingm, but embed new!
        if custom:
            print(f"Embedding custom...")
            corpus = self.evalset['context']
            self.p_emb = self.encoding_passage(custom)
            print(f"Embedding custom finished.")
            return


        assert search_context in ["tr", "eval", "tr_and_ev", "wiki"], f"you put {search_context} context must be either eval, all(train and eval set) or wiki."
        if search_context == "wiki":
            wiki_path = os.path.join(DPR_PATH, WIKI_EMB_PATH)
            files = os.listdir(wiki_path) # dir is your directory path
            num_of_file = len(files)
            if not force_encode and os.path.isdir(wiki_path) and num_of_file > 0: # when emb exists
                self.p_emb = self.loading_embedding(wiki_path, search_context)
                return
            else: # when need to embd new
                print("loading wiki data...")
                corpus = self.load_wikiset()
                print("loaded wiki set.")

        elif search_context == "tr":
            corpus = self.dataset['context']

        elif search_context == "tr_and_ev":
            all_path = os.path.join(DPR_PATH, ALL_EMB_PATH)
            files = os.listdir(all_path) # dir is your directory path
            num_of_file = len(files)
            if not force_encode and os.path.isdir(all_path) and num_of_file > 0: # when emb exists
                self.p_emb = self.loading_embedding(all_path, search_context)
                return
            else: # when need to embd new
                corpus = self.dataset['context'] + self.evalset['context']

        else: # eval only
            corpus = self.evalset['context']
        
        if partial:
            assert isinstance(partial, int) and ( 0 < partial and partial < len(corpus) ), "partial must be type int and 0 < partial < len(target corpus)."
            corpus = corpus[:partial]
            print(f"partial size :{len(corpus)} ")
        print(f"Embedding {search_context}...")
        self.p_emb = self.encoding_passage(corpus)
        print(f"Embedding {search_context} finished.")
        

    def encoding_question(self, queries):

        # 학습을 하고 나서 instance가 살아있다면 q_encoder을 다시 줄 필요가 없음.
        # q_encoder 모델을 불러와서 바로 인코딩 하고 싶을 때, 해당 모델을 생성하고 리트리버 껍데기만 만들고 나서 q_encoder 넣고 실행하면 된다.
        q_encoder = self.q_encoder
        assert q_encoder != None, "q_encoder is None. make sure you put q_encoder as input OR this retriever instance has been trained."
        print("encoding question....")
        with torch.no_grad():
            q_encoder.eval()

            # if input is single example, type casting to list for huggingface tokenizer type match
            if isinstance(queries, str):
                queries = [queries]

            q = self.tokenizer(queries, 
                            padding = "max_length", 
                            truncation = True, 
                            return_tensors = "pt"
                          ).to(self.args.device)

            q_seg = {'input_ids' : q['input_ids'], # (q_batch, max_len)
                       'attention_mask': q['attention_mask'],
                       'token_type_ids' :q['token_type_ids']
            }

            dataset = TensorDataset(q_seg['input_ids'], q_seg['attention_mask'],   q_seg['token_type_ids'])
            q_loader = DataLoader(dataset, batch_size=self.args.per_device_eval_batch_size)
            emb_list = []
            for idx, psg_batch in tqdm(enumerate(q_loader)):
                q_seg = {'input_ids' : psg_batch[0].to(self.args.device), # (psg_batch, max_len)
                       'attention_mask': psg_batch[1].to(self.args.device),
                       'token_type_ids' :psg_batch[2].to(self.args.device)
                }
                q_emb = q_encoder(**q_seg).to("cpu") # (psg_batch, emb)
                emb_list.append(q_emb)    
                torch.cuda.empty_cache()
            q_emb = torch.cat(emb_list)
            
        return q_emb

    def encoding_passage(self, corpus):

        p_encoder = self.p_encoder
        assert p_encoder != None, "p_encoder is None. make sure you put p_encoder as input OR this retriever instance has been trained."
        with torch.no_grad():
            p_encoder.eval()

            # if input is single example, type casting to list for huggingface tokenizer type match
            if isinstance(corpus, str):
                corpus = [corpus]
            print("number of corpus to encode: ", len(corpus))
            p = self.tokenizer(corpus, 
                            padding = "max_length", 
                            truncation = True, 
                            return_tensors = "pt"
                          ).to(self.args.device)
            p_seg = {'input_ids' : p['input_ids'], # (psg_batch, max_len)
                       'attention_mask': p['attention_mask'],
                       'token_type_ids' :p['token_type_ids']
            }

            # max_len = p_seg['input_ids'].size(-1) # max_len 구하기 위해서
            # p_seg['input_ids'] = p_seg['input_ids'].view(-1, num_neg + 1, max_len)
            # p_seg['attention_mask'] = p_seg['attention_mask'].view(-1, num_neg + 1, max_len) # (-1, num_neg + 1, -1) -> only one dim can be inferred.
            # p_seg['token_type_ids'] = p_seg['token_type_ids'].view(-1, num_neg + 1, max_len)

            dataset = TensorDataset(p_seg['input_ids'], p_seg['attention_mask'],   p_seg['token_type_ids'])
            psg_loader = DataLoader(dataset, batch_size=self.args.per_device_eval_batch_size)
            emb_list = []
            for idx, psg_batch in tqdm(enumerate(psg_loader)):
                p_seg = {'input_ids' : psg_batch[0].to(self.args.device), # (psg_batch, max_len)
                       'attention_mask': psg_batch[1].to(self.args.device),
                       'token_type_ids' :psg_batch[2].to(self.args.device)
                }
                p_emb = p_encoder(**p_seg).to("cpu") # (psg_batch, emb)
                emb_list.append(p_emb)    
                torch.cuda.empty_cache()
            p_emb = torch.cat(emb_list)
            print("passage embeding done!")
            print("passage embedding saved.")
        return p_emb

    def get_relevant_doc(self,
        queries,
        k=1,
        args=None,
        corpus = None,
    ):
        """
        Arguments:
            query (str)
                문자열로 주어진 질문입니다.
            k (int, default=1)
                상위 몇 개의 유사한 passage를 뽑을 것인지 결정합니다.
            args (Huggingface Arguments, default=None)
                Configuration을 필요한 경우 넣어줍니다.
                만약 None이 들어오면 self.args를 쓰도록 짜면 좋을 것 같습니다.

        Summary:
            1. query를 받아서 embedding을 하고
            2. 전체 passage와의 유사도를 구한 후
            3. 상위 k개의 문서 index를 반환합니다.
        """
        assert self.p_emb is not None, "embed passage first."

        # pass
        p_emb = self.encoding_passage(corpus) # (psg_batch, emb)
        q_emb = self.encoding_question(queries) # (1, emb)
    
        with torch.no_grad():
            p_emb = p_emb.T # dim match for dot product # (emb, psg_batch)

            # print(torch.matmul(q_emb, p_emb))
            sim_score = torch.matmul(q_emb, p_emb).squeeze() # (1, psg_batch) -> (psg_batch)
            index = torch.argsort(sim_score, descending=True) # 
            # index = index.squeeze()
            # print(index)
            # print(sim_score[index])
            index = np.array(index)
            corpus = np.array(corpus)
            # print(corpus[index])
        torch.cuda.empty_cache()

        return index[:k]

    def get_relevant_doc_bulk_index_score(self,
        queries,
        args=None,
    ):
        """
        Arguments:
            query (List[str])
                문자열로 주어진 질문입니다.
            k (int, default=1)
                상위 몇 개의 유사한 passage를 뽑을 것인지 결정합니다.
            args (Huggingface Arguments, default=None)
                Configuration을 필요한 경우 넣어줍니다.
                만약 None이 들어오면 self.args를 쓰도록 짜면 좋을 것 같습니다.

        Summary:
            1. query를 받아서 embedding을 하고
            2. 전체 passage와의 유사도를 구한 후
            3. 상위 k개의 문서 index를 반환합니다.

        Reason to seperate from default get_relevant_doc_bulk: In the vectorized, no need to calculate another k again.
        """
        # pass
        assert self.p_emb is not None, "embed passage first."
        q_emb = self.encoding_question(queries) # (q_batch, emb)
        with torch.no_grad():
            p_emb = self.p_emb.T # dim match for dot product # (emb, psg_batch)

            # print(torch.matmul(q_emb, p_emb))
            sim_score = torch.matmul(q_emb, p_emb) # (q_batch, psg_batch) #.squeeze()
            index = torch.argsort(sim_score, descending=True)
            index = index.squeeze() # (psg_batch)
            # print(index)
            # print(sim_score[index])
            index = np.array(index)
            sim_score = sim_score.numpy()
            # score = sim_score[index]
            score = np.take(sim_score, index)
        torch.cuda.empty_cache()

        return score, index
    def get_relevant_doc_bulk(self,
        queries,
        k=1,
        args=None,
    ):
        """
        Arguments:
            query (List[str])
                문자열로 주어진 질문입니다.
            k (int, default=1)
                상위 몇 개의 유사한 passage를 뽑을 것인지 결정합니다.
            args (Huggingface Arguments, default=None)
                Configuration을 필요한 경우 넣어줍니다.
                만약 None이 들어오면 self.args를 쓰도록 짜면 좋을 것 같습니다.

        Summary:
            1. query를 받아서 embedding을 하고
            2. 전체 passage와의 유사도를 구한 후
            3. 상위 k개의 문서 index를 반환합니다.
        """
        # pass
        assert self.p_emb is not None, "embed passage first."

        score, sim_index = self.get_relevant_doc_bulk_index_score(queries)

        return score[:,:k], sim_index[:,:k]


    def evaluate(self, search_context = "wiki", topk = 1, new_acc = None, partial = None):
        assert isinstance(topk, int) or all(isinstance(k, int) for k in topk ), "topk type must be int or List[int]"
        print(f"evaluation of {search_context} start...")
        # 1. 240개의 query를 240개에서 찾기: 위에서 진행. top 20에서 85? 85?% 을 맞춤.
        ## query: validation, corpus : validation 
        # topk = 20
        # query = self.evalset['question']
        # ground_truth = self.evalset['context']
        # cand_corpus = self.evalset['context']
        # self.get_dense_embedding(search_context = "eval")

        # why 1 fail?
        # try with fixed size in traningset
        # num_range = 3000
        # query = self.dataset["train"]['question'][:num_range]
        # ground_truth = self.dataset["train"]['context'][:num_range]
        # cand_corpus = self.dataset["train"]['context'][:num_range]
        # self.get_dense_embedding(search_context = "eval", custom = cand_corpus)

        # 2. 240개의 query를 4000개에서 찾기:
        ## query: validation, corpus: train + validation
        # query = self.evalset['question']
        # ground_truth = self.evalset['context']
        # cand_corpus = self.evalset['context'] + self.dataset["train"]['context']
        # self.get_dense_embedding(search_context = "tr_and_ev")

        # 3. 4000개의 query을 4000개에서 찾기:
        # 4. 240개의 query를 60000개에서 찾기: testset에 train의 question이 없다면 가장 유사한 경우
        # 5. 4000개의 query을 60000에서 찾기: 


        ## defalut
        if search_context =="tr_and_ev":
            query = self.dataset['question'] + self.evalset['question']
            ground_truth = self.dataset['context'] + self.evalset['context']
        elif search_context =="tr":
            query = self.dataset['question']
            ground_truth = self.dataset['context']
        else: # when search_context is either all or wiki, use validation set
            query = self.evalset['question']
            ground_truth = self.evalset['context']
        ## 만약 test 의 query에 train 혹은 validation의 질문이 동일하게 포함되어 있다면????
        ## 그러면 validation에서도 val의 query만 하는 것이 아니라... train 것도 다 포함해야 하지 않을까?
        ## val 의 query만 쓰는 이유가, 처음 보는 query와 passage을 잘 찾아내는 generalization 성능을 알기 위해서인데...

        if search_context == "wiki":
            cand_corpus = self.load_wikiset()
        else:
            cand_corpus = ground_truth

        if partial:
            assert isinstance(partial, int) and ( 0 < partial and partial < len(cand_corpus) ) , "partial must be type int, range(1, len(search corpus)"
            query = query[:partial]
            cand_corpus = cand_corpus[:partial]
            ground_truth = ground_truth[:partial]

        self.get_dense_embedding(search_context = search_context, partial = partial, force_encode = True)

        if isinstance(topk, int):
            k_list = [topk]
        else: # already checked if type is either int or List[int]
            k_list = topk
        
        score, results = self.get_relevant_doc_bulk_index_score(queries=query)
        # print(results.shape)
        for i, k in enumerate(k_list):
            sym_index = results[:,:k]
            # print(sym_index.shape)

            num_of_val = len(ground_truth)
            num_correct = 0
            for j, top_indexes in enumerate(sym_index):
                for idx in top_indexes:
                    if cand_corpus[idx][:10] == ground_truth[j][:10]:
                        # print(cand_corpus[idx][:10])
                        # print(ground_truth[i][:10])
                        num_correct += 1
                        # print("[match correct]:",cand_corpus[idx][:10])
                        break
                    # print(cand_corpus[idx][:10])
            eval_acc = num_correct / num_of_val
            # print(f"{search_context}_top{k} Accuracy: {eval_acc}. | num_correct is {num_correct} out of {num_of_val}. ")

            if new_acc:
                new_acc[i] = eval_acc
        # num_correct = 0
        # for i, top_indexes in enumerate(results):
        #     print(f"for {i} th question, ground truth is [{ground_truth[i][:10]}]")
        #     for idx in top_indexes:
        #         cand_psg.append(valid_corpus[idx][:100])
        #         print(f"top {i+1} passage is [{valid_corpus[idx][:10]}]")
        #         if valid_corpus[idx][:100] == ground_truth[i][:100]:
        #             num_correct += 1
        #             print("[CORRECT!]")
        #             break
        #     print()
        # print(num_correct)
        # validify(dataset, retriever, topk = 50)

        return new_acc
    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

def encode_passage(pretrained_p_enc, passage_set):

    pass


def load_encoder():
    p_encoder_loaded = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    p_encoder_loaded.load_state_dict(torch.load("./dpr_output/p_enc_model.pt"))
    q_encoder_loaded = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder_loaded.load_state_dict(torch.load("./dpr_output/q_enc_model.pt"))

def load_embedding():
    pass

def loaded_encoder_embedding_wiki_test(pretrained_p_enc, pretrained_q_enc):
    # wiki data encoding
    # load한 인코더 디코더 값이 저장하기 전과 같은지 확인.
    retriever_loaded = DenseRetrieval(
        args=args,
        dataset=None,#dataset,
        num_neg=2,
        tokenizer=tokenizer,
        p_encoder=pretrained_p_enc,
        q_encoder=pretrained_q_enc
    )
    import time
    start = time.time()

    # 지금 코드에서는 사실 validify 함수에 dataset만 잘 넣어주면 될것 같다. 
    # 
    with open("../data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    print("loaded wiki set")
    wikiset = list(
        dict.fromkeys([v["text"] for v in wiki.values()])
    )  # set 은 매번 순서가 바뀌므로
    # len(wikiset[:3000])
    validify(dataset, retriever_loaded, topk = 20000, add_context=wikiset, use_pre_encode = True)

    end = time.time()
    print(int(end - start))

def train(pretrained_psg_enc = None, pretrained_q_enc = None):
 
    # parser = HfArgumentParser(
    #     (ModelArguments, DataTrainingArguments, TrainingArguments,WandBArguments)
    # )
    # model_args, data_args, training_args,wandb_args = parser.parse_args_into_dataclasses()
    parser = HfArgumentParser(
        (WandBArguments)
    )
    wandb_args = parser.parse_args_into_dataclasses()[0]
    wandb_args.tags = list(wandb_args.tags)
    
    # print(f"model is from {model_args.model_name_or_path}")
    # print(f"data is from {data_args.dataset_name}")
    if not wandb_args.group :
        wandb_args.group = model_args.model_name_or_path

    if not wandb_args.tags: 
        # wandb_args.tags = [wandb_args.author,model_args.model_name_or_path]
        wandb_args.tags = [wandb_args.author]
    else:
        wandb_args.tags.append(wandb_args.author)
        # wandb_args.tags.append(model_args.model_name_or_path)
    
    # if model_args.model_name_or_path == "./models/train_dataset/" :
    #     wandb_args.group = "eval"
    #     wandb_args.tags = ["eval"]

    if not wandb_args.name:
        wandb_args.name = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    wandb_args.name = wandb_args.author +"/"+wandb_args.name
    print(f"Create {wandb_args.name} chart in wandB...")
    print(f"WandB {wandb_args.entity}/{wandb_args.project} Project Group and tages [{wandb_args.group},{wandb_args.tags}]")

    wandb.init(project=wandb_args.project,
                entity=wandb_args.entity,
                name=wandb_args.name,
                tags=wandb_args.tags,
                group=wandb_args.group,
                notes=wandb_args.notes)


    set_seed(42) # magic number :)


    print ("PyTorch version:[%s]."% (torch.__version__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("device:[%s]."%(device))
    model_checkpoint = "klue/bert-base"
    # model_checkpoint = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=412, # use only to encode passage and query. 
        num_train_epochs=100,
        weight_decay=0.01
    )
    wandb.config.update(args)
    # test
    p_encoder_loaded = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    # p_encoder_loaded.load_state_dict(torch.load("../dpr_output/best_p_enc_model_train_loss.pt"))
    q_encoder_loaded = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    # q_encoder_loaded.load_state_dict(torch.load("../dpr_output/best_p_enc_model_train_loss.pt"))
    dataset = load_from_disk("../data/train_dataset")

    trainset = dataset['train'] # for debug
    evalset = dataset['validation']
    retriever_loaded = DenseRetrieval(
        args=args,
        dataset=trainset,
        evalset=evalset,
        # num_neg=4,
        tokenizer=tokenizer,
        p_encoder=p_encoder_loaded,
        q_encoder=q_encoder_loaded
    )
    # retriever_loaded.evaluate(search_context = "wiki", topk = [1,5,20,50])
    retriever_loaded.train(logger = wandb, partial = None, eval_set = ["tr", "eval", "tr_and_ev"])#,"eval", "tr_and_ev"])#"wiki", "eval", "tr_and_ev"]) # partial for debug. validation set(no matter the val set is wiki or eval, etc... partial them to this number only.)
    # w = retriever_loaded.load_wikiset(partial = 100)
    # print(len(w))

# def evaluate():
def predict():
    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=512, # use only to encode passage and query. 
        num_train_epochs=30,
        weight_decay=0.01
    )
    model_checkpoint = "klue/bert-base"
    # p_encoder_loaded = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder_loaded = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    # p_encoder_loaded.load_state_dict(torch.load("../dpr_output/best_p_enc_model_train_loss.pt"))
    q_encoder_loaded.load_state_dict(torch.load("../dpr_output/best_p_enc_model_top1.pt"))
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    dataset = load_from_disk("../data/train_dataset")

    trainset = dataset['train'] # for debug
    evalset = dataset['validation']

    retriever_loaded = DenseRetrieval(
        prediction_only=True,
        args=args,
        dataset=None,
        evalset=None,
        # num_neg=4,
        tokenizer=tokenizer,
        p_encoder=None,#p_encoder_loaded,
        q_encoder=q_encoder_loaded,
        pre_encode_psg = "passage_embedding_top1.pt"
    )
    retriever_loaded.get_dense_embedding()
    # corpus = evalset['question']
    # print(retriever_loaded.retrieve(evalset, topk = 1))

if __name__ == "__main__":
    train()