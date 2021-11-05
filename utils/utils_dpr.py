from transformers import BertConfig
from transformers import is_torch_available, PreTrainedTokenizerFast, TrainingArguments,BertPreTrainedModel,BertModel
import torch
import os

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


def get_dpr_score(query, contexts, tokenizer, p_encoder_path, q_encoder_path):
    p_encoder = BertEncoder.from_pretrained("klue/bert-base")
    q_encoder = BertEncoder.from_pretrained("klue/bert-base")

    p_encoder.load_state_dict(torch.load(p_encoder_path))
    q_encoder.load_state_dict(torch.load(q_encoder_path))
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    with torch.no_grad():
        p_encoder.eval()
        q_encoder.eval()
        q_seqs_val = tokenizer(query, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        q_emb = q_encoder(**q_seqs_val).to('cpu') #(num_query, emb_dim)
        p_embs = []
        for p in contexts:
            p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            p_emb = p_encoder(**p).to('cpu').numpy()
            p_embs.append(p_emb)

        p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
    return dot_prod_scores