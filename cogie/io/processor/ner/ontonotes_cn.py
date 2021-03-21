"""
@Author: jinzhuan
@File: ontonotes_cn.py
@Desc: 
"""
from ..processor import Processor
from cogie.core import DataTable


class OntoNotesNERProcessor(Processor):
    def __init__(self, label_list=None, path=None, padding=None, unknown=None, bert_model='hfl/chinese-roberta-wwm-ext',
                 max_length=256):
        super().__init__(label_list, path, padding=padding, unknown=unknown, bert_model=bert_model,
                         max_length=max_length)

    def process(self, dataset):
        datable = DataTable()
        for sentence, label in zip(dataset['sentence'], dataset['label']):
            input_id, attention_mask, segment_id, head_index, label_id, label_mask = process(sentence, label,
                                                                                             self.tokenizer,
                                                                                             self.vocabulary,
                                                                                             self.max_length)
            if len(input_id) <= self.max_length and len(head_index) <= self.max_length and len(
                    label_id) <= self.max_length:
                datable('input_ids', input_id)
                datable('attention_mask', attention_mask)
                datable('segment_ids', segment_id)
                datable('head_indexes', head_index)
                datable('label_ids', label_id)
                datable('label_masks', label_mask)
        return datable


def process(words, labels, tokenizer, vocabulary, max_seq_length):
    input_id = []
    label_id = []
    words = ['[CLS]'] + words + ['[SEP]']
    is_heads = []
    head_index = []

    for word in words:
        token = tokenizer.tokenize(word)
        input_id.extend(token)
        if word in ['[CLS]', '[SEP]']:
            is_head = [0]
        else:
            is_head = [1] + [0] * (len(token) - 1)
        is_heads.extend(is_head)

    for i in range(len(is_heads)):
        if is_heads[i]:
            head_index.append(i)

    input_id = tokenizer.convert_tokens_to_ids(input_id)
    attention_mask = [1] * len(input_id)
    segment_id = [0] * len(input_id)

    for label in labels:
        label_id.append(vocabulary.to_index(label))
    label_mask = [1] * len(label_id)

    input_id += [0 for _ in range(max_seq_length - len(input_id))]
    attention_mask += [0 for _ in range(max_seq_length - len(attention_mask))]
    segment_id += [0 for _ in range(max_seq_length - len(segment_id))]
    head_index += [0 for _ in range(max_seq_length - len(head_index))]
    label_id += [-1 for _ in range(max_seq_length - len(label_id))]
    label_mask += [0 for _ in range(max_seq_length - len(label_mask))]

    return input_id, attention_mask, segment_id, head_index, label_id, label_mask

