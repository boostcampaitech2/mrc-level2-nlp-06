#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
pip install datasets==1.5.0
pip install transformers==4.5.0
pip install tqdm==4.41.1
pip install pandas==1.1.4
pip install scikit-learn==0.24.1
pip install konlpy==0.5.2
pip install wandb
pip install rank-bm25
pip install pathos
# faiss install (if you want to)
pip install faiss-gpu
pip install elasticsearch==7.6.0
pip install pororo==0.4.2
pip install kss==3.2.0
