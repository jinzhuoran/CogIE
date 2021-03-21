"""
@Author: jinzhuan
@File: conll2003.py
@Desc: 
"""
from ..processor import Processor
from transformers import BertTokenizer
from cogie.core import DataTable
from tqdm import tqdm
import nltk


class Conll2003NERProcessor(Processor):
    def __init__(self, label_list=None, path=None, padding=None, unknown=None, bert_model='bert-base-cased',
                 max_length=256):
        super().__init__(label_list, path, bert_model=bert_model,
                         max_length=max_length)

    def process(self, dataset):
        datable = DataTable()
        for sentence, label in zip(dataset['sentence'], dataset['label']):
            input_id, attention_mask, segment_id, valid_mask, label_id, label_mask = process(sentence, label,
                                                                                             self.tokenizer,
                                                                                             self.vocabulary,
                                                                                             self.max_length)
            datable('input_ids', input_id)
            datable('attention_mask', attention_mask)
            datable('segment_ids', segment_id)
            datable('valid_masks', valid_mask)
            datable('label_ids', label_id)
            datable('label_masks', label_mask)
        return datable


class TrexNerProcessor(Processor):
    def __init__(self, label_list=None, path=None, max_length=256, padding=None, unknown=None):
        super().__init__(label_list, path, padding, unknown)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.max_length = max_length
        self.blank_padding = True
        self.mask_entity = True

    def process(self, datasets):
        datable = DataTable()
        for dataset in tqdm(datasets, desc="Processing"):
            text = dataset['text']
            entities = dataset['entities']
            sentences_boundaries = dataset['sentences_boundaries']
            for sentences_boundary in sentences_boundaries:
                positions = []
                sentence = text[sentences_boundary[0]:sentences_boundary[1]]
                for entity in entities:
                    if entity['boundaries'][0] >= sentences_boundary[0] and \
                            entity['boundaries'][1] <= sentences_boundary[1] and 'entity' in entity['uri']:
                        positions.append(entity['boundaries'])
                words, labels = get_labels(sentence, positions, text, sentences_boundary)
                input_id, attention_mask, segment_id, valid_mask, label_id, label_mask = process(words, labels,
                                                                                                 self.tokenizer,
                                                                                                 self.vocabulary,
                                                                                                 self.max_length)
                datable('input_ids', input_id)
                datable('attention_masks', attention_mask)
                datable('segment_ids', segment_id)
                datable('valid_mask', valid_mask)
                datable('label_ids', label_id)
                datable('label_masks', label_mask)
        return datable


def get_labels(sentence, positions, text, sentences_boundary):
    words = nltk.word_tokenize(sentence)
    labels = ['O'] * len(words)
    left_length = len(nltk.word_tokenize(text[:sentences_boundary[0]]))
    for position in positions:
        left_words = nltk.word_tokenize(text[:position[0]])
        mention_words = nltk.word_tokenize(text[:position[1]])
        for i in range(len(left_words) - left_length, len(mention_words) - left_length):
            if i == len(left_words) - left_length:
                labels[i] = 'B'
            else:
                labels[i] = 'I'
    return words, labels


def process(words, labels, tokenizer, vocabulary, max_seq_length):
    input_id = []
    valid_mask = []
    label_id = []
    words.insert(0, '[CLS]')
    words.append('[SEP]')
    labels.insert(0, 'O')
    labels.append('O')
    for word in words:
        token = tokenizer.tokenize(word)
        input_id.extend(token)
        for i in range(len(token)):
            valid_mask.append(1 if i == 0 else 0)
    input_id = tokenizer.convert_tokens_to_ids(input_id)
    attention_mask = [1] * len(input_id)
    segment_id = [0] * len(input_id)

    for label in labels:
        label_id.append(vocabulary.to_index(label))
    label_mask = [1] * len(label_id)

    input_id = input_id[0:max_seq_length]
    attention_mask = attention_mask[0:max_seq_length]
    segment_id = segment_id[0:max_seq_length]
    valid_mask = valid_mask[0:max_seq_length]
    label_id = label_id[0:max_seq_length]
    label_mask = label_mask[0:max_seq_length]

    input_id += [0 for _ in range(max_seq_length - len(input_id))]
    attention_mask += [0 for _ in range(max_seq_length - len(attention_mask))]
    segment_id += [0 for _ in range(max_seq_length - len(segment_id))]
    valid_mask += [0 for _ in range(max_seq_length - len(valid_mask))]
    label_id += [-1 for _ in range(max_seq_length - len(label_id))]
    label_mask += [0 for _ in range(max_seq_length - len(label_mask))]

    return input_id, attention_mask, segment_id, valid_mask, label_id, label_mask
