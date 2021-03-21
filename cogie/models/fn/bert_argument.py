"""
@Author: jinzhuan
@File: bert_argument.py
@Desc: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel
from cogie.models import BaseFunction


class Bert4ArgumentFunction(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            batch=None,
    ):
        input_ids, attention_mask, segment_id, head_indexes, label_ids, label_mask, frame, pos = batch
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        batch_size, max_len, feat_dim = sequence_output.shape
        output = torch.zeros(batch_size, max_len, feat_dim * 3, dtype=torch.float, device=self.device)

        for i in range(batch_size):
            sequence_output[i] = torch.index_select(sequence_output[i], 0, head_indexes[i])
        for i in range(batch_size):
            for j in range(max_len):
                if pos[i].item() == j:
                    frame_emb = self.class_embedding[frame[i].item()]
                else:
                    frame_emb = self.class_embedding[0]
                output[i][j] = torch.cat(
                    (sequence_output[i][j], self.pos_embedding[self.to_pos[j - pos[i].item()]], frame_emb),
                    dim=0)
        output = self.dropout(output)
        output = self.classifier(output)
        return output

    def predict(
            self,
            batch=None,
    ):
        input_ids, attention_mask, segment_id, head_indexes, label_ids, label_mask, frame, pos = batch
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
        input_ids, attention_mask, segment_id, head_indexes, label_ids, label_mask, frame, pos = batch
        output = self.forward(batch)
        active_loss = label_mask.view(-1) == 1
        active_output = output.view(-1, self.label_size)[active_loss]
        active_labels = label_ids.view(-1)[active_loss]
        loss = loss_function(active_output, active_labels)
        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_ids, attention_mask, segment_id, head_indexes, label_ids, label_mask, frame, pos = batch
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
        metrics.evaluate(prediction, label_ids, valid_len)


class Bert4Argument(Bert4ArgumentFunction, nn.Module):
    def __init__(
            self,
            label_size,
            embedding_size=768,
            hidden_dropout_prob=0.1,
            bert_model='bert-base-cased',
            device=torch.device("cuda"),
            frame_vocabulary=None
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.label_size = label_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.device = device
        self.bert = BertModel.from_pretrained(bert_model)
        self.classifier = nn.Linear(self.embedding_size * 3, self.label_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.frame_vocabulary = frame_vocabulary
        self.pos_embedding = torch.nn.Parameter(torch.FloatTensor(256 * 2 + 1, self.embedding_size))
        self.class_embedding = torch.nn.Parameter(torch.FloatTensor(label_size + 1, self.embedding_size))
        self.to_pos = {}
        for i in range(-256, 257):
            self.to_pos[i] = i + 256


class Bert4ArgumentParallel(nn.DataParallel, Bert4ArgumentFunction):

    def __init__(self, module, device_ids):
        nn.DataParallel.__init__(self, module=module, device_ids=device_ids)
        self.label_size = self.module.label_size
        self.device = self.module.device
