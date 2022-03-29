"""
@Author: jinzhuan
@File: bert_re_entity.py
@Desc: 
"""
import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as f


class Bert4ReEntity(nn.Module):
    def __init__(self, label_size, bert_model):
        super().__init__()
        self.label_size = label_size
        self.hidden_size = 768 * 2
        self.bert = BertModel.from_pretrained(bert_model)
        self.linear = nn.Linear(self.hidden_size, self.label_size)

    def forward(
            self,
            batch=None
    ):
        token, att_mask, pos1, pos2, label_ids = batch
        hidden, _ = self.bert(token, attention_mask=att_mask)
        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        x = self.linear(x)
        return x

    def predict(
            self,
            batch=None,
    ):
        x = self.forward(batch)
        prediction_labels = torch.argmax(f.log_softmax(x, dim=1), dim=1)
        return prediction_labels

    def loss(
            self,
            batch=None,
            loss_function=None,
    ):
        token, att_mask, pos1, pos2, label_ids = batch
        x = self.forward(batch)
        loss = loss_function(x, label_ids)
        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        token, att_mask, pos1, pos2, label_ids = batch
        x = self.forward(batch)
        prediction_labels = torch.argmax(f.log_softmax(x, dim=1), dim=1)
        metrics.evaluate(prediction_labels, label_ids)
