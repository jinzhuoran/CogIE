"""
@Author: jinzhuan
@File: bert_crf.py
@Desc: 
"""
import torch
import torch.nn as nn
from transformers import BertModel
from cogie.models import BaseFunction
import torch.nn.functional as F
from torchcrf import CRF


class BertCRFFunction(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            batch=None,
    ):
        input_ids, attention_mask, segment_ids, head_indexes, label_ids, label_masks = batch
        label_masks = label_masks.type(torch.uint8)
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        sequence_output = self.dropout(sequence_output)
        batch_size, max_len, feat_dim = sequence_output.shape
        for i in range(batch_size):
            sequence_output[i] = torch.index_select(sequence_output[i], 0, head_indexes[i])
        output = self.classifier(sequence_output)
        output = F.log_softmax(output, dim=-1)
        return output

    def predict(
            self,
            batch=None,
    ):
        input_ids, attention_mask, segment_ids, head_indexes, label_ids, label_masks = batch
        label_masks = label_masks.type(torch.uint8)
        output = self.forward(batch)
        prediction = self.crf.decode(output, label_masks)
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
        input_ids, attention_mask, segment_ids, head_indexes, label_ids, label_masks = batch
        label_masks = label_masks.type(torch.uint8)
        output = self.forward(batch)
        loss = self.crf(output, label_ids, label_masks, reduction='token_mean')
        return -1 * loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_ids, attention_mask, segment_ids, head_indexes, label_ids, label_masks = batch
        prediction, valid_len = self.predict(batch)
        for pre in prediction:
            pre += (256 - len(pre)) * [0]
        prediction = torch.tensor(prediction, dtype=torch.long, device=self.device)
        metrics.evaluate(prediction, label_ids, valid_len)


class BertCRF(BertCRFFunction, nn.Module):
    def __init__(
            self,
            vocabulary,
            embedding_size=768,
            hidden_dropout_prob=0.5,
            bert_model='bert-base-cased',
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
        self.crf = CRF(self.label_size, batch_first=True)


class BertCRFParallel(nn.DataParallel, BertCRFFunction):

    def __init__(self, module, device_ids):
        nn.DataParallel.__init__(self, module=module, device_ids=device_ids)
        self.label_size = self.module.label_size
        self.device = self.module.device
        self.crf = self.module.crf
