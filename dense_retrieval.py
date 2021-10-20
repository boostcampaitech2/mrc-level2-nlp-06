import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from typing import List, Tuple, NoReturn, Any, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
from datasets import Dataset, load_from_disk, concatenate_datasets

from retrieval import SparseRetrieval, timer


class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
    super(BertEncoder, self).__init__(config)

    self.bert = BertModel(config)
    self.init_weights()
      
  def forward(self, input_ids, 
              attention_mask=None, token_type_ids=None): 
  
      outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
      pooled_output = outputs[1]
      return pooled_output

class DenseRetrieval(SparseRetrieval):
    """ SparseRetreival을 활용해, 메소드를 DenseRetrieval에 맞춰 오버라이딩
        기존에서 p_embedding, contexts, tfidfv를 가져옵니다.
    """
    def __init__(self, tokenize_fn, data_path, context_path, dataset_path):
        super().__init__(tokenize_fn, data_path, context_path)
        self.org_dataset = load_from_disk(dataset_path)
        self.full_ds = concatenate_datasets([
                self.org_dataset["train"].flatten_indices(),
                self.org_dataset["validation"].flatten_indices(),
                ])
        self.get_sparse_embedding() 
        self.num_neg = 3
        self.p_with_neg = []
        self.p_encoder = None
        self.q_encoder = None
        self.dense_p_embedding = []
    
    def get_resverse_topk_similarity(self, qeury_vec, k):
        """
        Arguments: 
            queries (List): 하나의 Query를 받습니다.
            k (Optional[int]): 1 하위 몇개의 Passage를 반환할지 정합니다.
        Note:
            !주의사항! Sparse클래스와 달리 하위 k개의 Passage를 반환합니다!
        """
        result = qeury_vec * self.p_embedding.T
        result = result.toarray()

        doc_scores3 = np.partition(result, k)[:, :k][:, ::-1]
        ind = np.argsort(doc_scores3, axis=-1)[:, :]# ::-1]
        doc_scores3 = np.sort(doc_scores3, axis=-1)[:, :]# ::-1]
        doc_indices3 = np.argpartition(result, k)[:, :k][:, :]# ::-1]
        r, c = ind.shape
        ind = ind + np.tile(np.arange(r).reshape(-1, 1), (1, c)) * c
        doc_indices3 = doc_indices3.ravel()[ind].reshape(r, c)

        return doc_scores3, doc_indices3

    def make_train_data(self, tokenizer):
        """ Note: Dense Embedding학습을 하기 위한 데이터셋을 만듭니다. """
        print("make_train_data...")
        corpus = np.array(self.contexts)
        query_vec = self.tfidfv.transform(self.full_ds['context'])
        doc_scores, doc_indices = self.get_resverse_topk_similarity(query_vec, self.num_neg)
        neg_idxs = doc_indices

        for idx, c in enumerate(tqdm(self.full_ds['context'])):
            p_neg = corpus[neg_idxs[idx]]
            self.p_with_neg.append(c)
            self.p_with_neg.extend(p_neg)

        print('[Positive context]')
        print(self.p_with_neg[4], '\n')
        print('[Negative context]')
        print(self.p_with_neg[5], '\n', self.p_with_neg[6])

        q_seqs = tokenizer(self.full_ds['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(self.p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, self.num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, self.num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, self.num_neg+1, max_len)

        print(p_seqs['input_ids'].size())  #(num_example, pos + neg, max_len)
        train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                                q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])
        return train_dataset

    def init_model(self, model_checkpoint):
        """ Encoder 모델을 생성해 줍니다."""
        print("init_model...")
        self.p_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
        self.q_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()

    def train(self, args, dataset):
        """ p_encoder, q_encoder를 학습시켜 줍니다. """
        print("training...")
        # Dataloader
        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
                {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0
        
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()
        
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for epoch, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            for step, batch in enumerate(epoch_iterator):
                self.q_encoder.train()
                self.p_encoder.train()
                
                targets = torch.zeros(args.per_device_train_batch_size).long()
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                    targets = targets.cuda()

                p_inputs = {'input_ids': batch[0].view(
                                                args.per_device_train_batch_size*(self.num_neg+1), -1),
                            'attention_mask': batch[1].view(
                                                args.per_device_train_batch_size*(self.num_neg+1), -1),
                            'token_type_ids': batch[2].view(
                                                args.per_device_train_batch_size*(self.num_neg+1), -1)
                            }
                
                q_inputs = {'input_ids': batch[3],
                            'attention_mask': batch[4],
                            'token_type_ids': batch[5]}
                
                p_outputs = self.p_encoder(**p_inputs)  #(batch_size*(self.num_neg+1), emb_dim)
                q_outputs = self.q_encoder(**q_inputs)  #(batch_size*, emb_dim)

                # Calculate similarity score & loss
                p_outputs = p_outputs.view(args.per_device_train_batch_size, -1, self.num_neg+1)
                q_outputs = q_outputs.view(args.per_device_train_batch_size, 1, -1)

                sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, self.num_neg+1)
                sim_scores = sim_scores.view(args.per_device_train_batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)
                #print(loss)

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.q_encoder.zero_grad()
                self.p_encoder.zero_grad()
                global_step += 1
                torch.cuda.empty_cache()

            torch.save(self.p_encoder.state_dict(), f"./outputs/p_encoder_{epoch}.pth")
            torch.save(self.q_encoder.state_dict(), f"./outputs/q_encoder_{epoch}.pth")
        return self.p_encoder, self.q_encoder

    def load_model(self, model_checkpoint, p_path, q_path):
        """ 학습이 완료된 p, q_encoder를 불러옵니다."""
        self.p_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
        self.q_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
        self.p_encoder.load_state_dict(torch.load(p_path))
        self.q_encoder.load_state_dict(torch.load(q_path))
        print("load_model finished...")
    

    def get_dense_embedding(self):
        """ p_encoder를 활용해 전체 문서에 대해 embedding 벡터를 계산합니다. 30분 소요"""
        p_embs = []
        with torch.no_grad():
            self.p_encoder.eval()
            for p in tqdm(self.contexts):
                p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                p_emb = self.p_encoder(**p).to('cpu').numpy()
                p_embs.append(p_emb)
            self.dense_p_embedding = torch.Tensor(p_embs).squeeze()
        print("get_dense_embedding finished...")


    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """ Arguments: 
            query (str): 하나의 Query를 받습니다.
            k (Optional[int]): 1 상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            메소드 오버라이딩. p,q_encoder를 계산하여, 최종 스코어와, idx를 반환해줍니다.
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        # 1. q encoder 이용 dense_q_embedding 생성
        with torch.no_grad():
            self.q_encoder.eval()
            q_seqs_val = tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            q_emb = self.q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)
        print('p_embs.size:', self.dense_p_embedding.size(), 'q_emb.size: ', q_emb.size())

        # 2. 생성된 embedding에 dot product를 수행 => Document들의 similarity ranking을 구함
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.dense_p_embedding, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        print('dot_prod_scores: ', dot_prod_scores)
        print('rank: ',rank)
        return dot_prod_scores[0], rank

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        """ 메소드 오버라이딩. Dataset형태로 queries가 들어오는 경우 수행 구현 필요"""
        return super().get_relevant_doc_bulk(queries, k=k)


if __name__=="__main__":
    data_path  = "../data/"
    dataset_path = "../data/train_dataset"
    context_path = "wikipedia_documents.json"
    model_checkpoint = "klue/bert-base"

    org_dataset = load_from_disk(dataset_path)
    full_ds = concatenate_datasets([
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ])

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,use_fast=False,)
    dense_retriever = DenseRetrieval(tokenize_fn=tokenizer.tokenize, data_path = data_path, context_path = context_path, dataset_path=dataset_path)

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
    )
    ## 학습과정 ##
    #train_dataset = detnse_retriever.make_train_data(tokenizer)
    #dense_retriever.init_model(model_checkpoint)
    #dense_retriever.train(args, train_dataset)

    ## 추론과정 ##
    dense_retriever.load_model(model_checkpoint, "outputs/p_encoder_0.pt", "outputs/q_encoder_0.pt")
    dense_retriever.get_dense_embedding()
    df = dense_retriever.retrieve(full_ds[0]['question'], topk=3)
    print(df)
    