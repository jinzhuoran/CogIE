"""
@Author: jinzhuan
@File: bert_ws.py
@Desc: 
"""
import torch
import torch.nn as nn
from transformers import BertModel
from cogie.models.base.base_function import BaseFunction
from cogie.modules.decoder.crf import ConditionalRandomField, allowed_transitions


class Bert4WSFunction(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            batch=None,
    ):
        input_ids, attention_mask, segment_ids, label_ids, label_masks = batch
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        sequence_output = self.dropout(sequence_output)
        output = self.classifier(sequence_output)
        return output

    def predict(
            self,
            batch=None,
    ):
        input_ids, attention_mask, segment_ids, label_ids, label_masks = batch
        output = self.forward(batch)
        prediction, ans_score = self.crf.viterbi_decode(output, label_masks)
        batch_size, max_len, feat_dim = output.shape
        valid_len = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        for i in range(batch_size):
            seq_len = 0
            for j in range(max_len):
                if label_masks[i][j].item() == 1:
                    seq_len += 1
                else:
                    valid_len[i] = seq_len
                    break
        return prediction, valid_len

    def loss(
            self,
            batch=None,
            loss_function=None,
    ):
        input_ids, attention_mask, segment_ids, label_ids, label_masks = batch
        output = self.forward(batch)
        loss = self.crf(output, label_ids, label_masks).mean()
        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_ids, attention_mask, segment_ids, label_ids, label_masks = batch
        output = self.forward(batch)
        prediction, ans_score = self.crf.viterbi_decode(output, label_masks)
        batch_size, max_len, feat_dim = output.shape
        valid_len = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        for i in range(batch_size):
            seq_len = 0
            for j in range(max_len):
                if label_masks[i][j].item() == 1:
                    seq_len += 1
                else:
                    valid_len[i] = seq_len
                    break
        metrics.evaluate(prediction, label_ids, valid_len)


class Bert4WS(Bert4WSFunction, nn.Module):
    def __init__(
            self,
            vocabulary,
            embedding_size=768,
            hidden_dropout_prob=0.1,
            bert_model='hfl/chinese-roberta-wwm-ext',
            device=torch.device("cuda"),
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.embedding_size = embedding_size
        self.label_size = len(self.vocabulary)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.device = device
        self.bert = BertModel.from_pretrained(bert_model)
        self.classifier = nn.Linear(self.embedding_size, self.label_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.crf = ConditionalRandomField(self.label_size, include_start_end_trans=True,
                                          allowed_transitions=allowed_transitions(vocabulary, include_start_end=True))


class Bert4WSParallel(nn.DataParallel, Bert4WSFunction):

    def __init__(self, module, device_ids):
        nn.DataParallel.__init__(self, module=module, device_ids=device_ids)
        self.vocabulary = self.module.vocabulary
        self.label_size = self.module.label_size
        self.device = self.module.device
        self.crf = self.module.crf
