"""
@Author: jinzhuan
@File: BBN.py
@Desc: 
"""
from ..processor import Processor
from cogie.core import DataTable


class BBNEtProcessor(Processor):
    def __init__(self, label_list=None, path=None, padding=None, unknown=None,
                 bert_model='bert-base-cased', max_length=256):
        super().__init__(label_list, path, padding, unknown, bert_model, max_length)

    def process(self, dataset):
        cnt = 0
        datable = DataTable()
        for i in range(len(dataset)):
            words, mentions = dataset[i]
            for mention in mentions:
                labels = mention['labels']
                input_id, attention_mask, segment_id, head_index, label_id = \
                    process(words, labels, self.tokenizer, self.vocabulary, self.max_length)
                if len(input_id) <= self.max_length and len(head_index) <= self.max_length:
                    datable('input_id', input_id)
                    datable('attention_mask', attention_mask)
                    datable('segment_id', segment_id)
                    datable('head_index', head_index)
                    datable('label_id', label_id)
                    datable('start', mention['start'])
                    datable('end', mention['end'])
                else:
                    cnt += 1
        print(cnt)
        return datable


def process(words, labels, tokenizer, vocabulary, max_seq_length):
    input_id = []
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
    label_id = [0] * len(vocabulary)
    for label in labels:
        label_id[vocabulary.to_index(label)] = 1

    input_id += [0 for _ in range(max_seq_length - len(input_id))]
    attention_mask += [0 for _ in range(max_seq_length - len(attention_mask))]
    segment_id += [0 for _ in range(max_seq_length - len(segment_id))]
    head_index += [0 for _ in range(max_seq_length - len(head_index))]

    return input_id, attention_mask, segment_id, head_index, label_id
