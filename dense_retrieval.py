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
        arguments: train_data: 기존 wiki데이터가 아닌 특정데이터를 활용할때 추가
    """
    def __init__(self, tokenize_fn, data_path, context_path, dataset_path, tokenizer, train_data):
        super().__init__(tokenize_fn, data_path, context_path)
        self.org_dataset = load_from_disk(dataset_path)
        self.train_data = train_data
        self.num_neg = 2
        self.p_with_neg = []
        self.p_encoder = None
        self.q_encoder = None
        self.dense_p_embedding = []
        self.tokenizer = tokenizer
        self.get_sparse_embedding() 
    
    def get_topk_similarity(self, qeury_vec, k):
        result = qeury_vec * self.p_embedding.T
        result = result.toarray()

        doc_scores3 = np.partition(result, -k)[:, -k:][:, ::-1]
        ind = np.argsort(doc_scores3, axis=-1)[:, ::-1]
        doc_scores3 = np.sort(doc_scores3, axis=-1)[:, ::-1]
        doc_indices3 = np.argpartition(result, -k)[:, -k:][:, ::-1]
        r, c = ind.shape
        ind = ind + np.tile(np.arange(r).reshape(-1, 1), (1, c)) * c
        doc_indices3 = doc_indices3.ravel()[ind].reshape(r, c)

        return doc_scores3, doc_indices3

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
        query_vec = self.tfidfv.transform(self.train_data['context'])
        doc_scores, doc_indices = self.get_topk_similarity(query_vec, self.num_neg*10)
        neg_idxs = []
        for idx, ind in enumerate(tqdm(doc_indices)): # 4000
            neg_idx = []
            for i in range(2, len(ind)): # 2~20 find negative
                if not self.contexts[ind[i]][:200] in self.train_data['context'][idx]:
                    neg_idx.append(ind[i])
                if len(neg_idx)==self.num_neg: break
            neg_idxs.append(neg_idx)
        
        print(neg_idxs)
        for idx, c in enumerate(tqdm(self.train_data['context'])):
            p_neg = corpus[neg_idxs[idx]]
            self.p_with_neg.append(c)
            self.p_with_neg.extend(p_neg)

        print(self.train_data['question'][0])
        print('[Positive context]')
        print(self.p_with_neg[0], '\n')
        print('[Negative context]')
        print(self.p_with_neg[1], '\n', self.p_with_neg[2])

        q_seqs = tokenizer(self.train_data['question'], padding="max_length", truncation=True, return_tensors='pt')
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
                p_outputs = torch.transpose(p_outputs.view(args.per_device_train_batch_size, self.num_neg+1, -1), 1, 2)
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

            torch.save(self.p_encoder.state_dict(), f"./outputs/p_encoder_{epoch}.pt")
            torch.save(self.q_encoder.state_dict(), f"./outputs/q_encoder_{epoch}.pt")
        return self.p_encoder, self.q_encoder

    def load_model(self, model_checkpoint, p_path, q_path):
        """ 학습이 완료된 p, q_encoder를 불러옵니다."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        self.p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(device)
        self.q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(device)
        self.p_encoder.load_state_dict(torch.load(p_path))
        self.q_encoder.load_state_dict(torch.load(q_path))
        print("load_model finished...")
    
    def get_dense_embedding(self):
        """ p_encoder를 활용해 전체 문서에 대해 embedding 벡터를 계산합니다. 12분 소요"""
        dataloader = DataLoader(self.train_data['context'], batch_size=4, drop_last=True)
        p_embs = []
        with torch.no_grad():
            self.p_encoder.eval()
            for step, batch in enumerate(tqdm(dataloader)):
                    batch = self.tokenizer(batch, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                    p_emb = self.p_encoder(**batch)
                    p_emb = p_emb.to('cpu').numpy()
                    p_embs.append(p_emb)
            self.dense_p_embedding = torch.Tensor(p_embs).reshape(-1,768)
        torch.cuda.empty_cache()
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
            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            q_emb = self.q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)

        # 2. 생성된 embedding에 dot product를 수행 => Document들의 similarity ranking을 구함
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.dense_p_embedding, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        #print('dot_prod_scores: ', dot_prod_scores)
        #print('rank: ',rank)
        torch.cuda.empty_cache()
        return dot_prod_scores[0], rank

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        """ 메소드 오버라이딩. Dataset형태로 queries가 들어오는 경우 수행 구현 필요"""
        dataloader = DataLoader(queries, batch_size=4)
        result = []
        with torch.no_grad():
            self.q_encoder.eval()
            for batch in tqdm(dataloader):    
                q_seqs_val = self.tokenizer(batch, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                q_emb = self.q_encoder(**q_seqs_val).to('cpu')            
                res = torch.matmul(q_emb, torch.transpose(self.dense_p_embedding, 0, 1))#.numpy() # 32, 56000
                result.append(res.tolist()) # [batch_size, 32, 56000]

        if not isinstance(result, np.ndarray):
                result = np.array(result)#.toarray()
        result = result.reshape((-1, self.dense_p_embedding.size(0)))

        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        torch.cuda.empty_cache()
        return doc_scores, doc_indices

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None or self.bm25 is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."
        
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.train_data['context'][doc_indices[i]])

            return (doc_scores, [self.train_data['context'][doc_indices[i]] for i in range(topk)])

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
                        [self.train_data['context'][pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

        # using parallel search, it tears the datsets to single example, which is dict type with keys
        elif isinstance(query_or_dataset, dict):
            # check for error in cases of wrong approach
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc(
                    query_or_dataset["question"], k=topk
                )
            key1 = list(query_or_dataset.keys())[0]
            assert isinstance(query_or_dataset[key1], str), "dict value is not str. Need to check if it might be a list. If so,\
                                                            it looks like; title:[..., ...]. This may cause serious malfunctioning."
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": query_or_dataset["question"],
                "id": query_or_dataset["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context_id": doc_indices,
                "context": " ".join(
                    [self.contexts[pid] for pid in doc_indices]
                ),
            }
            total.append(tmp)

            if "context" in query_or_dataset.keys() and "answers" in query_or_dataset.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = query_or_dataset["context"]
                tmp["answers"] = query_or_dataset["answers"]

            cqas = pd.DataFrame(total)
            return cqas
        else: # Added this branch because parallel processing increases the risk of malfunctioning. 
            raise Exception('The input is neither str, dataset, nor dict.')
        

    def topk_experiment(self, topK_list, dataset):
        """ MRC데이터에 대한 성능을 검증합니다. retrieve를 통한 결과 + acc측정"""
        result_dict = {}
        for topK in tqdm(topK_list):
            result_retriever = self.retrieve(dataset, topk=topK)
            correct = 0
            for index in tqdm(range(len(result_retriever)), desc="topk_experiment"):
                if  result_retriever['original_context'][index][:200] in result_retriever['context'][index]:
                    correct += 1
            result_dict[topK] = correct/len(result_retriever)
        return result_dict

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
    dense_retriever = DenseRetrieval(tokenize_fn=tokenizer.tokenize, data_path = data_path, 
                                    context_path = context_path, dataset_path=dataset_path, 
                                    tokenizer=tokenizer, train_data=org_dataset["validation"])

    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    ## 학습과정 ##
    train_dataset = dense_retriever.make_train_data(tokenizer)
    dense_retriever.init_model(model_checkpoint)
    dense_retriever.train(args, train_dataset)

    ## 추론준비 ##
    dense_retriever.load_model(model_checkpoint, "outputs/p_encoder_4.pt", "outputs/q_encoder_4.pt")
    dense_retriever.get_dense_embedding()

    ## 추론 ##
    for i in range(10):
        df = dense_retriever.retrieve(org_dataset['validation'][i]['question'], topk=3)
        print(df)

    ## topk 출력 ##
    topK_list = [1,10,20]
    result = dense_retriever.topk_experiment(topK_list, org_dataset["validation"])
    print(result)