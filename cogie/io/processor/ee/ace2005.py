"""
@Author: jinzhuan
@File: ace2005.py
@Desc: 
"""
import os
from ..processor import Processor
from cogie.utils import Vocabulary
from cogie.core import DataTable
from transformers import BertTokenizer


class ACE2005TriggerProcessor(Processor):
    """
    The ace2005 dataset processing follows https://github.com/nlpcl-lab/ace2005-preprocessing
    """

    def __init__(self, label_list=None, path=None, padding=None, unknown=None, bert_model='bert-base-cased',
                 max_length=256):
        super().__init__(label_list, path, padding, unknown, bert_model, max_length)

    def process(self, dataset, path=None):
        datable = DataTable()

        for item in dataset:
            words = item['words']
            triggers = ['O'] * len(words)
            for event_mention in item['golden-event-mentions']:
                for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                    trigger_type = event_mention['event_type']
                    if i == event_mention['trigger']['start']:
                        triggers[i] = 'B-{}'.format(trigger_type)
                    else:
                        triggers[i] = 'I-{}'.format(trigger_type)
            input_id, attention_mask, segment_id, valid_mask, label_id, label_mask = process(words, triggers,
                                                                                             self.tokenizer,
                                                                                             self.vocabulary,
                                                                                             self.max_length)
            datable('input_ids', input_id)
            datable('attention_mask', attention_mask)
            datable('segment_ids', segment_id)
            datable('valid_masks', valid_mask)
            datable('label_ids', label_id)
            datable('label_masks', label_mask)

        if path and os.path.exists(path):
            datable.save_table(path)
        return datable


class ACE2005Processor:
    """
    The ace2005 dataset processing follows https://github.com/nlpcl-lab/ace2005-preprocessing
    """

    def __init__(self, trigger_path=None, argument_path=None, bert_model='bert-base-cased', max_length=256):
        if trigger_path and argument_path:
            self.trigger_vocabulary = Vocabulary.load(trigger_path)
            self.argument_vocabulary = Vocabulary.load(argument_path)
            self.max_length = max_length
            self.bert_model = bert_model
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

    def process(self, dataset, path=None):
        datable = DataTable()
        for item in dataset:
            words = item['words']
            triggers = ['O'] * len(words)
            arguments = {
                'candidates': [
                    # ex. (5, 6, "entity_type_str"), ...
                ],
                'events': {
                    # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                },
            }

            for entity_mention in item['golden-entity-mentions']:
                arguments['candidates'].append(
                    (entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))

            for event_mention in item['golden-event-mentions']:
                for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                    trigger_type = event_mention['event_type']
                    if i == event_mention['trigger']['start']:
                        triggers[i] = 'B-{}'.format(trigger_type)
                    else:
                        triggers[i] = 'I-{}'.format(trigger_type)

                event_key = (
                    event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])
                arguments['events'][event_key] = []
                for argument in event_mention['arguments']:
                    role = argument['role']
                    if role.startswith('Time'):
                        role = role.split('-')[0]
                    arguments['events'][event_key].append(
                        (argument['start'], argument['end'], self.argument_vocabulary.to_index(role)))

            words = ['[CLS]'] + words + ['[SEP]']
            tokens_x, triggers_y, arguments, head_indexes, words, triggers = \
                process(words, triggers, arguments, self.tokenizer, self.trigger_vocabulary, self.max_length)

            datable('tokens_x', tokens_x)
            datable('triggers_y', triggers_y)
            arguments = {
                'candidates': [
                    # ex. (5, 6, "entity_type_str"), ...
                ],
                'events': {
                    # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                },
            }
            datable('arguments', arguments)
            datable('head_indexes', head_indexes)
            datable('words', words)
            datable('triggers', triggers)

        if path and os.path.exists(path):
            datable.save_table(path)
        return datable


def process(words, triggers, arguments, tokenizer, trigger_vocabulary, max_seq_length):
    # We give credits only to the first piece.
    tokens_x, is_heads = [], []
    for w in words:
        tokens = tokenizer.tokenize(w) if w not in ['[CLS]', '[SEP]'] else [w]
        tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

        if w in ['[CLS]', '[SEP]']:
            is_head = [0]
        else:
            is_head = [1] + [0] * (len(tokens) - 1)

        tokens_x.extend(tokens_xx)
        is_heads.extend(is_head)

    triggers_y = [trigger_vocabulary.to_index(t) for t in triggers]
    head_indexes = []
    for i in range(len(is_heads)):
        if is_heads[i]:
            head_indexes.append(i)
    tokens_x = tokens_x + [0] * (max_seq_length - len(tokens_x))
    head_indexes = head_indexes + [0] * (max_seq_length - len(head_indexes))
    triggers_y = triggers_y + [trigger_vocabulary.to_index('<pad>')] * (max_seq_length - len(triggers_y))

    return tokens_x, triggers_y, arguments, head_indexes, words, triggers
