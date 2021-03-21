"""
@Author: jinzhuan
@File: bert_frame.py
@Desc: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel
from cogie.models import BaseFunction


class Bert4FrameFunction(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            batch=None,
    ):
        input_ids, attention_mask, head_indexes, frame_id, element_id, label_mask = self.get_batch(batch)
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        head_indexes = torch.LongTensor(head_indexes).to(self.device)
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        batch_size, max_len, feat_dim = sequence_output.shape
        for i in range(batch_size):
            sequence_output[i] = torch.index_select(sequence_output[i], 0, head_indexes[i])
        sequence_output = self.dropout(sequence_output)
        output = self.classifier(sequence_output)
        return output

    def predict(
            self,
            batch=None,
    ):
        input_ids, attention_mask, head_indexes, frame_id, element_id, label_mask = self.get_batch(batch)
        label_mask = torch.LongTensor(label_mask).to(self.device)
        output = self.forward(batch)
        prediction = torch.argmax(f.log_softmax(output, dim=2), dim=2)
        batch_size, max_len, feat_dim = output.shape
        valid_len = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        for i in range(batch_size):
            seq_len = 0
            for j in range(max_len):
                if label_mask[i][j].item() == 1:
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
        input_ids, attention_mask, head_indexes, frame_id, element_id, label_mask = self.get_batch(batch)
        frame_id = torch.LongTensor(frame_id).to(self.device)
        label_mask = torch.LongTensor(label_mask).to(self.device)
        output = self.forward(batch)
        active_loss = label_mask.view(-1) == 1
        active_output = output.view(-1, self.label_size)[active_loss]
        active_labels = frame_id.view(-1)[active_loss]
        loss = loss_function(active_output, active_labels)
        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_ids, attention_mask, head_indexes, frame_id, element_id, label_mask = self.get_batch(batch)
        frame_id = torch.LongTensor(frame_id).to(self.device)
        label_mask = torch.LongTensor(label_mask).to(self.device)
        output = self.forward(batch)
        prediction = torch.argmax(f.log_softmax(output, dim=2), dim=2)
        batch_size, max_len, feat_dim = output.shape
        valid_len = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        for i in range(batch_size):
            seq_len = 0
            for j in range(max_len):
                if label_mask[i][j].item() == 1:
                    seq_len += 1
                else:
                    valid_len[i] = seq_len
                    break
        metrics.evaluate(prediction, frame_id, valid_len)

    def get_batch(self, batch):
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            head_indexes = batch['head_indexes']
            frame_id = batch['frame_id']
            element_id = batch['element_id']
            label_mask = batch['label_mask']
            return input_ids, attention_mask, head_indexes, frame_id, element_id, label_mask
        else:
            return batch


class Bert4Frame(Bert4FrameFunction, nn.Module):
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


class Bert4FrameParallel(nn.DataParallel, Bert4FrameFunction):

    def __init__(self, module, device_ids):
        nn.DataParallel.__init__(self, module=module, device_ids=device_ids)
        self.label_size = self.module.label_size
        self.device = self.module.device
