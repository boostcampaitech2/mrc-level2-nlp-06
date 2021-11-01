import rank_bm25
import numpy as np

def get_top_n(corpus_size, get_scores_func, query, documents, n=5):
    assert corpus_size == len(documents), "The documents given don't match the index corpus!"

    scores = get_scores_func(query)

    top_n_idx = np.argsort(scores)[::-1][:n]
    doc_score = scores[top_n_idx]
    
    return doc_score, top_n_idx


class OurBm25(rank_bm25.BM25Okapi): # must do like this. Doing "from rank_bm25 import BM250kapi"  
                                   # and inherit BM250kapi directly, cannot save pickle.
                                   # See https://stackoverflow.com/questions/1412787/picklingerror-cant-pickle-class-decimal-decimal-its-not-the-same-object
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25, delta=0.5):
            super().__init__(corpus, tokenizer=tokenizer, k1=k1, b=b, epsilon=epsilon)    
    
    def get_top_n(self, query, documents, n=5):
        return get_top_n(self.corpus_size, self.get_scores, query, documents, n=n)

class OurBm25L(rank_bm25.BM25L): # must do like this. Doing "from rank_bm25 import BM250kapi"  
                                   # and inherit BM250kapi directly, cannot save pickle.
                                   # See https://stackoverflow.com/questions/1412787/picklingerror-cant-pickle-class-decimal-decimal-its-not-the-same-object
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25, delta=0.5):
            super().__init__(corpus, tokenizer=tokenizer, k1=k1, b=b, delta = delta )     
    
    def get_top_n(self, query, documents, n=5):
        return get_top_n(self.corpus_size, self.get_scores, query, documents, n=n)

class OurBm25Plus(rank_bm25.BM25Plus): # must do like this. Doing "from rank_bm25 import BM250kapi"  
                                   # and inherit BM250kapi directly, cannot save pickle.
                                   # See https://stackoverflow.com/questions/1412787/picklingerror-cant-pickle-class-decimal-decimal-its-not-the-same-object
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25, delta=1):
            super().__init__(corpus, tokenizer=tokenizer, k1=k1, b=b, delta = delta )     
    
    def get_top_n(self, query, documents, n=5):
        return get_top_n(self.corpus_size, self.get_scores, query, documents, n=n)