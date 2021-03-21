"""
@Author: jinzhuan
@File: bert_re.py
@Desc: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel
from cogie.models import BaseFunction
import random


class Bert4ReFunction(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            batch=None,
    ):
        input_ids, attention_mask, head_indexes, entity_mentions, relation_mentions, entity_mentions_mask, relation_mentions_mask = batch
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        for i in range(batch_size):
            sequence_output[i] = torch.index_select(sequence_output[i], 0, head_indexes[i])

        sequence_output = self.dropout(sequence_output)
        valid_entity_mentions = []
        valid_relation_mentions = []
        for idx in range(batch_size):
            length = 0
            for i in range(max_len):
                if entity_mentions_mask[idx][i].item() == 1:
                    length += 1
                else:
                    break
            valid_entity_mentions.append(entity_mentions[idx][:length])

            length = 0
            for i in range(max_len):
                if relation_mentions_mask[idx][i].item() == 1:
                    length += 1
                else:
                    break
            valid_relation_mentions.append(relation_mentions[idx][:length])
        relation_predictions = []
        relation_prediction_pairs = []
        relation_golden_pairs = []
        if self.training:
            for idx in range(batch_size):
                for i in range(len(valid_relation_mentions[idx])):
                    relation_prediction_pairs.append([idx] + valid_relation_mentions[idx][i].tolist()[:-1])
                    mentions_object = sequence_output[idx][
                                      valid_relation_mentions[idx][i][0].item():valid_relation_mentions[idx][i][1].item()]
                    mentions_object = torch.mean(mentions_object, dim=0)
                    mentions_subject = sequence_output[idx][
                                       valid_relation_mentions[idx][i][2].item():valid_relation_mentions[idx][i][
                                           3].item()]
                    mentions_subject = torch.mean(mentions_subject, dim=0)
                    object_subject = torch.cat([mentions_object, mentions_subject], 0)
                    relation_predictions.append(object_subject)
            for idx in range(batch_size):
                for i in range(len(valid_entity_mentions[idx])):
                    for j in range(len(valid_entity_mentions[idx])):
                        if i == j:
                            continue
                        elif [idx] + valid_entity_mentions[idx][i].tolist() \
                                + valid_entity_mentions[idx][j].tolist() in relation_prediction_pairs:
                            continue
                        elif random.random() > self.sample_prob:
                            continue
                        else:
                            mentions_object = sequence_output[idx][
                                              valid_entity_mentions[idx][i][0].item():valid_entity_mentions[idx][i][
                                                  1].item()]
                            mentions_object = torch.mean(mentions_object, dim=0)
                            mentions_subject = sequence_output[idx][
                                               valid_entity_mentions[idx][j][0].item():valid_entity_mentions[idx][j][
                                                   1].item()]
                            mentions_subject = torch.mean(mentions_subject, dim=0)
                            object_subject = torch.cat([mentions_object, mentions_subject], 0)
                            relation_predictions.append(object_subject)
                            relation_prediction_pairs.append(
                                [idx] + valid_entity_mentions[idx][i].tolist() + valid_entity_mentions[idx][j].tolist())
        else:
            for idx in range(batch_size):
                for i in range(len(valid_entity_mentions[idx])):
                    for j in range(len(valid_entity_mentions[idx])):
                        if i == j:
                            continue
                        else:
                            mentions_object = sequence_output[idx][
                                              valid_entity_mentions[idx][i][0].item():valid_entity_mentions[idx][i][
                                                  1].item()]
                            mentions_object = torch.mean(mentions_object, dim=0)
                            mentions_subject = sequence_output[idx][
                                               valid_entity_mentions[idx][j][0].item():valid_entity_mentions[idx][j][
                                                   1].item()]
                            mentions_subject = torch.mean(mentions_subject, dim=0)
                            object_subject = torch.cat([mentions_object, mentions_subject], 0)
                            relation_predictions.append(object_subject)
                            relation_prediction_pairs.append(
                                [idx] + valid_entity_mentions[idx][i].tolist() + valid_entity_mentions[idx][j].tolist())
        for idx in range(batch_size):
            for i in range(len(valid_relation_mentions[idx])):
                relation_golden_pairs.append([idx] + valid_relation_mentions[idx][i].tolist())
        if len(relation_prediction_pairs) == 0:
            relation_predictions = torch.tensor([], dtype=torch.long, device=self.device)
        else:
            relation_predictions = torch.stack(tuple(relation_predictions), dim=0)
        relation_prediction_pairs = torch.LongTensor(relation_prediction_pairs).to(self.device)
        relation_golden_pairs = torch.LongTensor(relation_golden_pairs).to(self.device)
        return relation_predictions, relation_prediction_pairs, relation_golden_pairs

    def predict(
            self,
            batch=None,
    ):
        relation_predictions, relation_prediction_pairs, relation_golden_pairs = self.forward(batch)
        relation_prediction_pairs = relation_prediction_pairs.cpu().numpy().tolist()
        output = self.classifier(relation_predictions)
        prediction_labels = torch.argmax(f.log_softmax(output, dim=1), dim=1)
        pred_tuple = []
        for i in range(len(relation_prediction_pairs)):
            if self.vocabulary.to_index('<unk>') != prediction_labels[i].item():
                pred_tuple.append(tuple(relation_prediction_pairs[i] + [prediction_labels[i].item()]))
                continue
        return pred_tuple

    def loss(
            self,
            batch=None,
            loss_function=None,
    ):
        input_ids, attention_mask, head_indexes, entity_mentions, relation_mentions, entity_mentions_mask, relation_mentions_mask = batch
        relation_predictions, relation_prediction_pairs, relation_golden_pairs = self.forward(batch)
        relation_prediction_pairs = relation_prediction_pairs.cpu().numpy().tolist()
        relation_golden_pairs = relation_golden_pairs.cpu().numpy().tolist()
        output = self.classifier(relation_predictions)
        prediction_labels = torch.argmax(f.log_softmax(output, dim=1), dim=1)
        active_labels = []
        for prediction_pair in relation_prediction_pairs:
            label = self.vocabulary.to_index('<unk>')
            for golden_pair in relation_golden_pairs:
                if prediction_pair == golden_pair[:-1]:
                    label = golden_pair[-1]
                    break
            active_labels.append(label)
        active_labels = torch.tensor(active_labels, dtype=torch.long, device=self.device)
        loss = loss_function(output, active_labels)
        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_ids, attention_mask, head_indexes, entity_mentions, relation_mentions, entity_mentions_mask, relation_mentions_mask = batch
        relation_predictions, relation_prediction_pairs, relation_golden_pairs = self.forward(batch)
        relation_prediction_pairs = relation_prediction_pairs.cpu().numpy().tolist()
        relation_golden_pairs = relation_golden_pairs.cpu().numpy().tolist()
        output = self.classifier(relation_predictions)
        prediction_labels = torch.argmax(f.log_softmax(output, dim=1), dim=1)
        pred_tuple = []
        target_tuple = []
        for golden_pair in relation_golden_pairs:
            target_tuple.append(tuple(golden_pair))
        for i in range(len(relation_prediction_pairs)):
            if self.vocabulary.to_index('<unk>') != prediction_labels[i].item():
                pred_tuple.append(tuple(relation_prediction_pairs[i] + [prediction_labels[i].item()]))
                continue
        metrics.evaluate(pred_tuple, target_tuple)


class Bert4Re(Bert4ReFunction, nn.Module):
    def __init__(
            self,
            vocabulary,
            embedding_size=768 * 2,
            hidden_dropout_prob=0.1,
            bert_model='bert-base-cased',
            device=torch.device("cuda"),
            sample_prob=0.2,
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
        self.sample_prob = sample_prob


class Bert4ReParallel(nn.DataParallel, Bert4ReFunction):

    def __init__(self, module, device_ids):
        nn.DataParallel.__init__(self, module=module, device_ids=device_ids)
        self.label_size = self.module.label_size
        self.device = self.module.device
        self.classifier = self.module.classifier
        self.vocabulary = self.module.vocabulary
        self.sample_prob = self.module.sample_prob
