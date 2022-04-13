
from ..processor import Processor
from transformers import BertTokenizer
from cogie.core import DataTable
from tqdm import tqdm
import nltk
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class TrexW2NERProcessor(Processor):
    def __init__(self, label_list=None, path=None, padding=None, unknown=None, bert_model='bert-base-cased',
                 max_length=256):
        super().__init__(label_list, path, bert_model=bert_model,unknown=unknown,
                         max_length=max_length)

    def process(self, dataset):
        datable = DataTable()
        # add your own process code here
        for sentence, ner in zip(dataset['sentence'], dataset['ner']):
            bert_inputs,attention_masks,\
            grid_labels, grid_mask2d, \
            pieces2word, dist_inputs, \
            sent_length, entity_text = \
                process_w2ner(sentence,ner,self.tokenizer,self.vocabulary,self.max_length)

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


def process_w2ner(sentence, ner, tokenizer, vocab, max_seq_length):
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

    for entity in ner:
        index = entity["index"]
        for i in range(len(index)):
            if i + 1 >= len(index):
                break
            _grid_labels[index[i], index[i + 1]] = 1
        _grid_labels[index[-1], index[0]] = vocab.word2idx[entity["type"]]
    entity_texts = []
    # for idx,label in enumerate(labels):
    #     # label = label.lower()
    #     # if label == 'o':
    #     #     label = "<pad>"
    #     _grid_labels[idx,idx] = vocab.word2idx[label]
    #     # if vocab.word2idx[label] >= 10:
    #         # print("????label={}????".format(label))
    #     entity_texts.append(str(idx)+"-#-"+str(vocab.word2idx[label]))
    # entity_texts = set(entity_texts)

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
