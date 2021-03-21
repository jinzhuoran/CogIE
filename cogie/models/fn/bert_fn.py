"""
@Author: jinzhuan
@File: bert_fn.py
@Desc: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel
from cogie.models import BaseFunction


class Bert4FnFunction(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            batch=None,
    ):
        input_ids, attention_mask, word_pos, label_ids = batch
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        ave_output = torch.zeros(batch_size, feat_dim, dtype=torch.float, device=self.device)
        for i in range(batch_size):
            ave_output[i] = torch.mean(sequence_output[i][word_pos[i][0].item():word_pos[i][1].item()], dim=0)
        ave_output = self.dropout(ave_output)
        output = self.classifier(ave_output)
        return output

    def predict(
            self,
            batch=None,
    ):
        output = self.forward(batch)
        prediction = torch.argmax(f.log_softmax(output, dim=1), dim=1)
        return prediction

    def loss(
            self,
            batch=None,
            loss_function=None,
    ):
        input_ids, attention_mask, word_pos, label_ids = batch
        output = self.forward(batch)
        loss = loss_function(output.view(-1, self.label_size), label_ids.view(-1))
        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_ids, attention_mask, word_pos, label_ids = batch
        output = self.forward(batch)
        prediction = torch.argmax(f.log_softmax(output, dim=1), dim=1)
        metrics.evaluate(prediction, label_ids)


class Bert4Fn(Bert4FnFunction, nn.Module):
    def __init__(
            self,
            label_size,
            embedding_size=768,
            hidden_dropout_prob=0.1,
            bert_model='bert-base-cased',
            device=torch.device("cuda"),
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.label_size = label_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.device = device
        self.bert = BertModel.from_pretrained(bert_model)
        self.classifier = nn.Linear(self.embedding_size, self.label_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


class Bert4FnParallel(nn.DataParallel, Bert4FnFunction):

    def __init__(self, module, device_ids):
        nn.DataParallel.__init__(self, module=module, device_ids=device_ids)
        self.label_size = self.module.label_size
        self.device = self.module.device
