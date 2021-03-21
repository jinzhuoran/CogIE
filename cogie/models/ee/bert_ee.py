"""
@Author: jinzhuan
@File: bert_ee.py
@Desc: 
"""
import torch
import torch.nn as nn
from transformers import BertModel
from cogie.models import BaseFunction


class Bert4EEFunction(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        tokens_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d, words, triggers = self.get_batch(batch)
        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = self.predict_triggers(
            tokens_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d)
        argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = [], [], [], []
        if len(argument_keys) > 0:
            argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = self.predict_arguments(
                argument_hidden, argument_keys, arguments_2d)

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d

    def predict_triggers(self, tokens_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)
        encoded_layers, _ = self.bert(tokens_x_2d)
        x = encoded_layers

        batch_size = tokens_x_2d.shape[0]

        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])

        trigger_logits = self.trigger_classifier(x)
        trigger_hat_2d = trigger_logits.argmax(-1)

        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = x[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers(
                [self.trigger_vocabulary.to_word(trigger) for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = x[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys

    def predict_arguments(self, argument_hidden, argument_keys, arguments_2d):
        argument_hidden = torch.stack(argument_hidden)
        argument_logits = self.argument_classifier(argument_hidden)
        argument_hat_1d = argument_logits.argmax(-1)

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            a_label = self.argument_vocabulary.to_index('<unk>')
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys,
                                                                                 argument_hat_1d.cpu().numpy()):
            if a_label == self.argument_vocabulary.to_index('<unk>'):
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d

    def predict(
            self,
            batch=None
    ):
        tokens_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d, words, triggers = self.get_batch(batch)
        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = self.forward(
            batch)
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        batch_size = tokens_x_2d.shape[0]
        for i in range(batch_size):
            predicted_triggers = find_triggers(
                [self.trigger_vocabulary.to_word(trigger) for trigger in trigger_hat_2d[i].tolist()])
        return argument_hat_2d

    def loss(
            self,
            batch=None,
            loss_function=None,
    ):
        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = self.forward(
            batch)
        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = loss_function(trigger_logits, triggers_y_2d.view(-1))

        if len(argument_keys) > 0:
            argument_loss = loss_function(argument_logits, arguments_y_1d)
            loss = trigger_loss + 2 * argument_loss
        else:
            loss = trigger_loss

        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        tokens_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d, words, triggers = self.get_batch(batch)
        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys, argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = self.forward(
            batch)
        metrics.evaluate(words, triggers, trigger_hat_2d, arguments_2d, argument_hat_2d, argument_keys)

    def get_batch(self, batch):
        if isinstance(batch, dict):
            tokens_x_2d = batch['tokens_x']
            head_indexes_2d = batch['head_indexes']
            triggers_y_2d = batch['triggers_y']
            arguments_2d = batch['arguments']
            words = batch['words']
            triggers = batch['triggers']
            return tokens_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d, words, triggers
        else:
            return batch


class Bert4EE(Bert4EEFunction, nn.Module):
    def __init__(
            self,
            trigger_vocabulary,
            argument_vocabulary,
            embedding_size=768,
            hidden_dropout_prob=0.1,
            bert_model='bert-base-cased',
            device=torch.device("cuda"),
    ):
        super().__init__()
        self.trigger_vocabulary = trigger_vocabulary
        self.argument_vocabulary = argument_vocabulary
        self.trigger_size = len(trigger_vocabulary)
        self.argument_size = len(argument_vocabulary)
        self.embedding_size = embedding_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.device = device
        self.bert_model = bert_model
        self.bert = BertModel.from_pretrained(bert_model)
        self.trigger_classifier = nn.Linear(self.embedding_size, self.trigger_size)
        self.argument_classifier = nn.Linear(self.embedding_size * 2, self.argument_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


class Bert4EEParallel(nn.DataParallel, Bert4EEFunction):

    def __init__(self, module, device_ids):
        nn.DataParallel.__init__(self, module=module, device_ids=device_ids)
        self.trigger_vocabulary = self.module.trigger_vocabulary
        self.argument_vocabulary = self.module.argument_vocabulary
        self.trigger_size = len(self.module.trigger_vocabulary)
        self.argument_size = len(self.module.argument_vocabulary)
        self.device = self.module.device


def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][1]])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]
