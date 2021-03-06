import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd
from torch.nn.functional import softmax
from pathos.multiprocessing import ProcessingPool as Pool

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union
from transformers import AutoTokenizer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

import rank_bm25
from utils.utils_dpr import get_dpr_score
from utils.preprocess import wiki_preprocess

from utils.preprocess import preprocess_wiki_documents

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# for multi processing :(
retriever = None 
def par_search(queries, topk):
    # pool.map may put only one argument. We need two arguments: datasets and topk.
    def wrapper(query): 
        rel_doc = retriever.get_relevant_doc(query, k = topk)
        return rel_doc
    pool = Pool()

    pool.restart() 

    rel_docs_score_indices = pool.map(wrapper, queries)
    pool.close()
    pool.join()

    doc_scores = []
    doc_indices = []
    for s,idx in rel_docs_score_indices:
        doc_scores.append( s )
        doc_indices.append( idx )

    return doc_scores, doc_indices



class MyBm25(rank_bm25.BM25Okapi): # must do like this. Doing "from rank_bm25 import BM250kapi"  
                                   # and inherit BM250kapi directly, cannot save pickle.
                                   # See https://stackoverflow.com/questions/1412787/picklingerror-cant-pickle-class-decimal-decimal-its-not-the-same-object
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
            super().__init__(corpus, tokenizer=tokenizer, k1=k1, b=b, epsilon=epsilon)    
    
    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)

        top_n_idx = np.argsort(scores)[::-1][:n]
        doc_score = scores[top_n_idx]
        
        return doc_score, top_n_idx


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        is_bm25 = False,
        use_wiki_preprocessing = False,
        k1=1.5, b=0.75, epsilon=0.25,
        q_encoder = None,
        p_encoder = None
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

            is_bm25:
                유사도 랭킹을 bm25로 할것인지 결정합니다.

            use_wiki_preprocessing:
                wiki documents를 전처리할지 결정합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """
        self.tokenize_fn = tokenize_fn
        self.data_path = data_path
        
        # wiki data 전처리한 파일이 없으면 만들기
        if context_path == 'mod_wiki.json':
            if not os.path.isfile("/opt/ml/data/mod_wiki.json") :
                with open("/opt/ml/data/wikipedia_documents.json", "r") as f:
                    wiki = json.load(f)
                wiki_dict = dict()
                for ids in range(len(wiki)):
                    # 인덱스 번호가 string type
                    wiki_dict[str(ids)] = wiki_preprocess(wiki[str(ids)])

                with open('/opt/ml/data/mod_wiki.json', 'w', encoding='utf-8') as mf:
                    json.dump(wiki_dict, mf, indent="\t", ensure_ascii=False)
        
        with open(os.path.join(self.data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)


        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique wiki contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # wiki 전처리
        if use_wiki_preprocessing:
            self.contexts = preprocess_wiki_documents(self.contexts)

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=self.tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )
        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

        # Transform by bm25
        self.bm25 = None
        self.is_bm25 = is_bm25
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        #encoder for dpr
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder


    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        if not self.is_bm25: # tfidf
            pickle_name = f"sparse_embedding.bin"
            tfidfv_name = f"tfidv.bin"
            emd_path = os.path.join(self.data_path, pickle_name)
            tfidfv_path = os.path.join(self.data_path, tfidfv_name)

            if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
                with open(emd_path, "rb") as file:
                    self.p_embedding = pickle.load(file)
                with open(tfidfv_path, "rb") as file:
                    self.tfidfv = pickle.load(file)
                print("Embedding pickle load.")
            else:
                print("Build passage embedding")
                self.p_embedding = self.tfidfv.fit_transform(self.contexts)
                print(self.p_embedding.shape)
                with open(emd_path, "wb") as file:
                    pickle.dump(self.p_embedding, file)
                with open(tfidfv_path, "wb") as file:
                    pickle.dump(self.tfidfv, file)
                print("Embedding pickle saved.")

        else: # bm25
            bm25_name = f"bm25.bin"
            bm25_path = os.path.join(self.data_path, bm25_name)
            if os.path.isfile(bm25_path):
                with open(bm25_path, "rb") as file:
                    self.bm25 = pickle.load(file)
                print("Embedding bm25 pickle load.")
            else:
                print("Building bm25... It may take 1 minute and 30 seconds...")
                # bm25 must tokenizer first 
                # because it runs pool inside and this cuases unexpected result.
                tokenized_corpus = []
                for c in tqdm(self.contexts):
                    tokenized_corpus.append(self.tokenize_fn(c))
                self.bm25 = MyBm25(tokenized_corpus, k1 = self.k1, b = self.b, epsilon=self.epsilon)
                with open(bm25_path, "wb") as file:
                    pickle.dump(self.bm25, file)
                print("bm25 pickle saved.")

        

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve_dpr(self, dataset, topk: Optional[int] = 20):
        print("dpr mode!")
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        dpr_score = get_dpr_score(dataset['question'], self.contexts, tokenizer,self.p_encoder, self.q_encoder)

        bm25_score = []
        for query in dataset['question']:
            tok_q = self.tokenize_fn(query)
            bm25_score.append(self.bm25.get_scores(tok_q))
        bm25_score = torch.tensor(np.array(bm25_score))
        dpr_score = softmax(dpr_score,dim=1)
        bm25_score = softmax(bm25_score,dim=1)

        total_score = []
        for idx in range(len(dataset['question'])):
            total_score.append((dpr_score[idx]*0.2+bm25_score[idx]).tolist())
        total_score = torch.tensor(np.array(total_score))
        ranks = torch.argsort(total_score, dim=1, descending=True).squeeze()
        context_list = []
        for index in range(len(ranks)):
            k_list = []
            for i in range(topk):
                k_list.append(self.contexts[ranks[index][i]])
            context_list.append(k_list)
                
        total = []
        for idx, example in enumerate(
            tqdm(dataset, desc="Sparse retrieval: ")
        ):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context_id": ranks[idx][:topk],
                "context": " ".join(
                    context_list[idx]
                ),
            }
            # if "context" in example.keys() and "answers" in example.keys():
            #     # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
            #     tmp["original_context"] = example["context"]
            #     tmp["answers"] = example["answers"]
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

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

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        if not self.is_bm25:
            with timer("transform"):
                query_vec = self.tfidfv.transform([query])
            assert (
                np.sum(query_vec) != 0
            ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            with timer("query ex search"):
                result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()

            sorted_result = np.argsort(result.squeeze())[::-1]
            doc_score = result.squeeze()[sorted_result].tolist()[:k]
            doc_indices = sorted_result.tolist()[:k]
            return doc_score, doc_indices
        else: #bm25
            try:
                # query가 한개의 string이 아닐 때 에러가 나요.
                tok_q = self.tokenize_fn(query)
            except:
                raise Exception("While processing bm25 with parallel serach, input is expected to be a single query, but somethong went wrong. Find this error in get_relevant_doc in retrieval.py")
            doc_score, doc_indices = self.bm25.get_top_n(tok_q, self.contexts, n = k)
            return doc_score, doc_indices
        
    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        if not self.is_bm25:
            query_vec = self.tfidfv.transform(queries)
            assert (
                np.sum(query_vec) != 0
            ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            result = query_vec * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            doc_scores = []
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_scores.append(result[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
            return doc_scores, doc_indices
        else: #bm25
            doc_scores = []
            doc_indices = []

            # parallel search
            # 하나로 쪼개서 안에 들어가서 각각 토크나이즈를 한다.
            global retriever
            retriever = self
            doc_scores, doc_indices = par_search(queries, k)

            return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

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
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
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

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")
    parser.add_argument("--use_wiki_preprocessing", metavar=False, type=bool, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
        is_bm25=True
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
