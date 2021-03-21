"""
@Author: jinzhuan
@File: framenet.py
@Desc: 
"""
from cogie.core import DataTable
from cogie.utils import Vocabulary
from transformers import BertTokenizer


class FrameNetProcessor:
    def __init__(self, frame_path=None, element_path=None, bert_model='bert-base-cased', max_length=256):
        self.frame_vocabulary = Vocabulary.load(frame_path)
        self.element_vocabulary = Vocabulary.load(element_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_length = max_length

    def process(self, dataset):
        datable = DataTable()
        for item in dataset.values():
            sentence = item['sentence']
            frames = item['frames']
            elements = item['elements']
            input_ids, attention_mask, head_indexes, frame_id, element_id, label_mask = process(sentence, frames,
                                                                                                elements,
                                                                                                self.tokenizer,
                                                                                                self.frame_vocabulary,
                                                                                                self.element_vocabulary,
                                                                                                self.max_length)
            datable('input_ids', input_ids)
            datable('attention_mask', attention_mask)
            datable('head_indexes', head_indexes)
            datable('frame_id', frame_id)
            datable('element_id', element_id)
            datable('label_mask', label_mask)
        return datable


def process(sentence, frames, elements, tokenizer, frame_vocabulary, element_vocabulary, max_length):
    input_ids, is_heads = [], []
    sentence = ['[CLS]'] + sentence + ['[SEP]']
    frame_label = ['<unk>'] * len(sentence)
    element_id = []
    for word in sentence:
        token = tokenizer.tokenize(word) if word not in ['[CLS]', '[SEP]'] else [word]
        input_id = tokenizer.convert_tokens_to_ids(token)

        if word in ['[CLS]', '[SEP]']:
            is_head = [0]
        else:
            is_head = [1] + [0] * (len(token) - 1)

        input_ids.extend(input_id)
        is_heads.extend(is_head)

    attention_mask = [1] * len(input_ids)

    for frame, element in zip(frames, elements):
        for i in range(len(frame)):
            if frame[i] != '<unk>':
                frame_label[i] = frame[i]
                element_list = [element_vocabulary.to_index(e) for e in element]
                element_list = element_list + [element_vocabulary.to_index('<pad>')] * (max_length - len(element_list))
                element_label = {tuple([i, i + 1, frame[i]]): element_list}
                element_id.append(element_label)
                break

    frame_id = [frame_vocabulary.to_index(f) for f in frame_label]

    label_mask = [1] * len(frame_id)
    head_indexes = []
    for i in range(len(is_heads)):
        if is_heads[i]:
            head_indexes.append(i)

    input_ids = input_ids + [0] * (max_length - len(input_ids))
    attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
    head_indexes = head_indexes + [0] * (max_length - len(head_indexes))
    frame_id = frame_id + [frame_vocabulary.to_index('<pad>')] * (max_length - len(frame_id))
    label_mask = label_mask + [0] * (max_length - len(label_mask))

    return input_ids, attention_mask, head_indexes, frame_id, element_id, label_mask


def process(sentence, frames, elements, tokenizer, frame_vocabulary, element_vocabulary, max_length):
    input_ids, is_heads = [], []
    sentence = ['[CLS]'] + sentence + ['[SEP]']
    frame_label = ['<unk>'] * len(sentence)
    element_id = []
    for word in sentence:
        token = tokenizer.tokenize(word) if word not in ['[CLS]', '[SEP]'] else [word]
        input_id = tokenizer.convert_tokens_to_ids(token)

        if word in ['[CLS]', '[SEP]']:
            is_head = [0]
        else:
            is_head = [1] + [0] * (len(token) - 1)

        input_ids.extend(input_id)
        is_heads.extend(is_head)

    attention_mask = [1] * len(input_ids)

    for frame, element in zip(frames, elements):
        for i in range(len(frame)):
            if frame[i] != '<unk>':
                frame_label[i] = frame[i]
                element_list = [element_vocabulary.to_index(e) for e in element]
                element_list = element_list + [element_vocabulary.to_index('<pad>')] * (max_length - len(element_list))
                element_label = {tuple([i, i + 1, frame[i]]): element_list}
                element_id.append(element_label)
                break

    frame_id = [frame_vocabulary.to_index(f) for f in frame_label]

    label_mask = [1] * len(frame_id)
    head_indexes = []
    for i in range(len(is_heads)):
        if is_heads[i]:
            head_indexes.append(i)

    input_ids = input_ids + [0] * (max_length - len(input_ids))
    attention_mask = attention_mask + [0] * (max_length - len(attention_mask))
    head_indexes = head_indexes + [0] * (max_length - len(head_indexes))
    frame_id = frame_id + [frame_vocabulary.to_index('<pad>')] * (max_length - len(frame_id))
    label_mask = label_mask + [0] * (max_length - len(label_mask))

    return input_ids, attention_mask, head_indexes, frame_id, element_id, label_mask
