"""
@Author: jinzhuan
@File: bert_et_context.py
@Desc: 
"""
import torch
import torch.nn as nn
from transformers import BertModel
from cogie.models import BaseFunction


class Bert4EtWithContextFunction(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            batch=None,
    ):
        input_id, attention_mask, segment_id, head_index, label_id, start, end = batch
        sequence_output = self.bert(input_ids=input_id, attention_mask=attention_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        for i in range(batch_size):
            sequence_output[i] = torch.index_select(sequence_output[i], 0, head_index[i])

        mention_output = torch.zeros(batch_size, feat_dim, dtype=torch.float, device=self.device)
        for i in range(batch_size):
            mention_output[i] = torch.mean(sequence_output[i][start[i].item():end[i].item()], dim=0)

        if self.use_context:
            left_output = torch.zeros(batch_size, feat_dim, dtype=torch.float, device=self.device)
            for i in range(batch_size):
                left_output[i] = torch.mean(sequence_output[i][0:start[i].item()], dim=0)

            right_output = torch.zeros(batch_size, feat_dim, dtype=torch.float, device=self.device)
            for i in range(batch_size):
                right_len = 0
                for j in range(max_len):
                    if head_index[i][j].item() != 0:
                        right_len += 1
                right_output[i] = torch.mean(sequence_output[i][end[i].item():right_len], dim=0)

            mention_output = torch.cat((left_output, mention_output, right_output), dim=1)

        output = self.classifier(mention_output)

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
        input_id, attention_mask, segment_id, head_index, label_id, start, end = batch
        output = self.forward(batch)
        loss = loss_function(output.view(-1), label_id.view(-1).float())
        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_id, attention_mask, segment_id, head_index, label_id, start, end = batch
        output = self.forward(batch)
        output = self.sigmoid(output)
        # output = torch.clamp(output, 0, 1)
        # output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        output = output > 0.5
        output = output.long()
        metrics.evaluate(output, label_id)


class Bert4EtWithContext(Bert4EtWithContextFunction, nn.Module):
    def __init__(
            self,
            label_size,
            embedding_size=768,
            hidden_dropout_prob=0.1,
            bert_model='bert-base-cased',
            device=torch.device("cuda"),
            use_context=False,

    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.label_size = label_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.device = device
        self.use_context = use_context

        self.bert = BertModel.from_pretrained(bert_model)
        if self.use_context:
            self.classifier = nn.Linear(self.embedding_size * 3, self.label_size)
        else:
            self.classifier = nn.Linear(self.embedding_size, self.label_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.sigmoid = nn.Sigmoid()


class Bert4EtWithContextParallel(nn.DataParallel, Bert4EtWithContextFunction):

    def __init__(self, module, device_ids):
        nn.DataParallel.__init__(self, module=module, device_ids=device_ids)
        self.label_size = self.module.label_size
        self.device = self.module.device
        self.sigmoid = self.module.sigmoid
