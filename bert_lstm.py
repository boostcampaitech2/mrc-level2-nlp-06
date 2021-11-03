from transformers import BertModel, BertTokenizerFast
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from torch import nn

class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          ### New layers:
          self.lstm = nn.LSTM(self.bert.config.hidden_size, 256, batch_first=True,bidirectional=True, num_layers=10)
        #   self.linear = nn.Linear(256*2, <number_of_classes>)
          
    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(
               ids, 
               attention_mask=mask)

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings
          hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
          linear_output = self.linear(hidden.view(-1,256*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification

          return linear_output

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = CustomBERTModel()


class BERT_QA(nn.Module):
    def __init__(self):
        super(BERT_QA, self).__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")    
        
        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2) # * 2 concat bidirection lstm hidden state

    def forward(        
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # return_dict = False
            )

        sequence_output = outputs.last_hidden_state

        # print(sequence_output.shape)#torch.Size([8, 384, 768])



        # 이하 huggingface 코드 복붙. 허페! 허페!
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # print(start_logits.shape)#torch.Size([8, 384])
        # print(start_positions.shape)#torch.Size([8])

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # if not return_dict:
        if False:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




class BERT_LSTM(nn.Module):
    def __init__(self):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")

        
        ### New layers:
        self.HIDDEN_DIM = 256

        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.HIDDEN_DIM, batch_first=True,bidirectional=True)
        self.qa_outputs = nn.Linear(self.HIDDEN_DIM * 2, 2) # * 2 concat bidirection lstm hidden state

        #   self.linear = nn.Linear(HIDDEN_DIM*2, <number_of_classes>)
          

    def forward(        
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        
        # sequence_output, pooled_output 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # return_dict = False
            )
        sequence_output = outputs.last_hidden_state
        """
        File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/rnn.py", line 564, in forward
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        AttributeError: 'str' object has no attribute 'size'
        
        solution
        # https://stackoverflow.com/questions/65132144/bertmodel-transformers-outputs-string-instead-of-tensor
        return_dict = False from huggingface new vertion.
        
        근데 아래서 어차피 또 사용해서 그냥 return dict = True로 두고, 그냥 내가 빼서 쓰기로 했다.

        """
        # print(sequence_output.shape) # torch.Size([8, 384, 768])
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings
        # print(lstm_output.shape)
        # print(h.shape)
        # h = h.view(-1, self.HIDDEN_DIM * 2)# bidirectional lstm hidden state to single node
        # print("lstm_output.shape:", lstm_output.shape)

        # 이하 huggingface 코드 복붙. 허페! 허페!
        logits = self.qa_outputs(lstm_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # print(start_logits.shape)
        # print(start_positions.shape)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        """
        ValueError: Expected input batch_size (2) to match target batch_size (8).
        -> trainer 자체의 에러이다.
        여기랑 뭔상관이지... 혹시...
        # if not return_dict:?
        # trainer api에 들어가는 config을 최대한 맞춰봐야겠다.
        # 별거 없는디...?
        # loss을 밖에서 구하나? 근데 그러면 여기에서는 뭘 한거임? 
        # 실제 trainer api에 들어가는 모델도 auto model이 전부라서 이 코드랑 동일할텐데?
        # 아니다. 내가 에러 코드를 잘못 읽었음. trainer은 여기 이전의 외부이고 ,
        # 실제로 여기 내부에서 발생한 에러가 맞다.
        # 여기 안의 nll loss이다. 즉 위의 cross entropy에서 발생함.

        print(start_logits.shape)
        print(start_positions.shape)
        torch.Size([2, 8])
        torch.Size([8])
        
        2가 있는 의미는 분명... bidirectional LSTM에서 나온게 아닐까? 
        view으로 해결!
        """
        
        if not return_dict:
        # if False:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



# class BertForQuestionAnswering(BertPreTrainedModel):


#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertModel(config, add_pooling_layer=False)
#         self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)


#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         start_positions=None,
#         end_positions=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for position (index) of the start of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
#             sequence are not taken into account for computing the loss.
#         end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
#             Labels for position (index) of the end of the labelled span for computing the token classification loss.
#             Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
#             sequence are not taken into account for computing the loss.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]







#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         total_loss = None
#         if start_positions is not None and end_positions is not None:
#             # If we are on multi-GPU, split add a dimension
#             if len(start_positions.size()) > 1:
#                 start_positions = start_positions.squeeze(-1)
#             if len(end_positions.size()) > 1:
#                 end_positions = end_positions.squeeze(-1)
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             ignored_index = start_logits.size(1)
#             start_positions = start_positions.clamp(0, ignored_index)
#             end_positions = end_positions.clamp(0, ignored_index)

#             loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
#             start_loss = loss_fct(start_logits, start_positions)
#             end_loss = loss_fct(end_logits, end_positions)
#             total_loss = (start_loss + end_loss) / 2

#         if not return_dict:
#             output = (start_logits, end_logits) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output

#         return QuestionAnsweringModelOutput(
#             loss=total_loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )