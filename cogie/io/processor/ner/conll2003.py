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
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class Conll2003W2NERProcessor(Processor):
    def __init__(self, label_list=None, path=None, padding=None, unknown=None, bert_model='bert-base-cased',
                 max_length=256):
        super().__init__(label_list, path, bert_model=bert_model,
                         max_length=max_length)
        # self.vocabulary.idx2word = {0: '<pad>', 1: '<suc>', 2: 'b-org', 3: 'b-misc', 4: 'b-per', 5: 'i-per', 6: 'b-loc'}
        # self.vocabulary.word2idx = {'<pad>': 0, '<suc>': 1, 'b-org': 2, 'b-misc': 3, 'b-per': 4, 'i-per': 5, 'b-loc': 6}

        # self.vocabulary.idx2word = {0: '<pad>', 1: '<suc>', 2: 'b-org', 3: 'b-misc', 4: 'b-per', 5: 'i-per', 6: 'b-loc', 7: 'i-org', 8: 'i-misc', 9: 'i-loc'}
        # self.vocabulary.word2idx ={'<pad>': 0, '<suc>': 1, 'b-org': 2, 'b-misc': 3, 'b-per': 4, 'i-per': 5, 'b-loc': 6, 'i-org': 7, 'i-misc': 8, 'i-loc': 9}

    def process(self, dataset):
        datable = DataTable()
        # add your own process code here
        for sentence, label in zip(dataset['sentence'], dataset['label']):
            bert_inputs,attention_masks,\
            grid_labels, grid_mask2d, \
            pieces2word, dist_inputs, \
            sent_length, entity_text = \
                process_w2ner(sentence,label,self.tokenizer,self.vocabulary,self.max_length)

            datable('bert_inputs', bert_inputs)
            datable('attention_masks',attention_masks)
            datable('grid_labels', grid_labels)
            datable('grid_mask2d', grid_mask2d)
            datable('pieces2word', pieces2word)
            datable('dist_inputs', dist_inputs)
            datable('sent_length', sent_length)
            # datable('entity_text', entity_text)
            # 暂时不用text信息 不方便拼接成batch

        return datable


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


def process_w2ner(sentence, labels, tokenizer, vocab, max_seq_length):
    dis2idx = np.zeros((1000), dtype='int64')
    dis2idx[1] = 1
    dis2idx[2:] = 2
    dis2idx[4:] = 3
    dis2idx[8:] = 4
    dis2idx[16:] = 5
    dis2idx[32:] = 6
    dis2idx[64:] = 7
    dis2idx[128:] = 8
    dis2idx[256:] = 9

    tokens = [tokenizer.tokenize(word) for word in sentence]
    pieces = [piece for pieces in tokens for piece in pieces]
    _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
    _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
    _attention_mask = np.ones_like(_bert_inputs)

    length = len(sentence)
    _grid_labels = np.zeros((length, length), dtype=np.int)
    _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
    _dist_inputs = np.zeros((length, length), dtype=np.int)
    _grid_mask2d = np.ones((length, length), dtype=np.bool)

    if tokenizer is not None:
        start = 0
        for i, pieces in enumerate(tokens):
            if len(pieces) == 0:
                continue
            pieces = list(range(start, start + len(pieces)))
            _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
            start += len(pieces)

    for k in range(length):
        _dist_inputs[k, :] += k
        _dist_inputs[:, k] -= k

    for i in range(length):
        for j in range(length):
            if _dist_inputs[i, j] < 0:
                _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
            else:
                _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
    _dist_inputs[_dist_inputs == 0] = 19

    entity_texts = []
    for idx,label in enumerate(labels):
        # label = label.lower()
        # if label == 'o':
        #     label = "<pad>"
        _grid_labels[idx,idx] = vocab.word2idx[label]
        # if vocab.word2idx[label] >= 10:
            # print("????label={}????".format(label))
        entity_texts.append(str(idx)+"-#-"+str(vocab.word2idx[label]))
    entity_texts = set(entity_texts)

    # padding to max_seq_length
    max_tok = max_seq_length
    sent_length = length
    max_pie = max_tok

    padded_bert_inputs = np.zeros(max_pie,dtype=np.long)
    padded_bert_inputs[:_bert_inputs.shape[0]] = _bert_inputs
    padded_attention_mask = np.zeros(max_pie,dtype=np.long)
    padded_attention_mask[:_attention_mask.shape[0]] = _attention_mask

    def fill(data, new_data):
        new_data[:data.shape[0],:data.shape[1]] = data
        return new_data


    dis_mat = np.zeros((max_tok,max_tok),dtype=np.long)
    _dist_inputs = fill(_dist_inputs,dis_mat)
    labels_mat = np.zeros((max_tok,max_pie),dtype=np.long)
    _grid_labels = fill(_grid_labels,labels_mat)
    mask2d_mat = np.zeros((max_tok, max_tok), dtype=np.bool)
    _grid_mask2d = fill(_grid_mask2d, mask2d_mat)
    sub_mat = np.zeros((max_tok, max_pie), dtype=np.bool)
    _pieces2word = fill(_pieces2word, sub_mat)



    return padded_bert_inputs,padded_attention_mask, _grid_labels, _grid_mask2d, _pieces2word, _dist_inputs, sent_length, entity_texts
