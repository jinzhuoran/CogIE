"""
@Author: jinzhuan
@File: bert_et.py
@Desc: 
"""
import torch
import torch.nn as nn
from transformers import BertModel
from cogie.models import BaseModule


class Bert4Et(BaseModule):
    def __init__(
            self,
            class_size,
            embedding_size=768,
            hidden_dropout_prob=0.1,
            bert_model='bert-base-cased',
            device=torch.device("cuda"),
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.class_size = class_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.device = device
        self.bert = BertModel.from_pretrained(bert_model)
        self.classifier = nn.Linear(self.embedding_size, self.class_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            batch=None,
    ):
        input_ids, attention_mask, start_pos, end_pos, label_ids = batch
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        ave_output = torch.zeros(batch_size, feat_dim, dtype=torch.float, device=self.device)
        for i in range(batch_size):
            ave_output[i] = torch.mean(sequence_output[i][start_pos[i][0].item():end_pos[i][0].item()], dim=0)
        output = self.classifier(ave_output)
        output = self.sigmoid(output)
        return output

    def predict(
            self,
            batch=None,
    ):
        output = self.forward(batch)
        output = output > 0.5
        output = output.long()
        return output

    def loss(
            self,
            batch=None,
            loss_function=None,
    ):
        input_ids, attention_mask, start_pos, end_pos, label_ids = batch
        output = self.forward(batch)
        loss = loss_function(output.view(-1), label_ids.view(-1))
        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_ids, attention_mask, start_pos, end_pos, label_ids = batch
        output = self.forward(batch)
        output = output > 0.5
        output = output.long()
        metrics.evaluate(output, label_ids)


class Bert4EtParallel(nn.DataParallel):

    def __init__(self, module, device_ids):
        super().__init__(module=module, device_ids=device_ids)
        self.class_size = self.module.class_size

    def loss(
            self,
            batch=None,
            loss_function=None,
    ):
        input_ids, attention_mask, start_pos, end_pos, label_ids = batch
        output = self.forward(batch)
        loss = loss_function(output.view(-1), label_ids.view(-1).float())
        return loss

    def predict(
            self,
            batch=None,
    ):
        output = self.forward(batch)
        output = output > 0.5
        output = output.long()
        return output

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_ids, attention_mask, start_pos, end_pos, label_ids = batch
        output = self.forward(batch)
        output = output > 0.5
        output = output.long()
        metrics.evaluate(output, label_ids)
