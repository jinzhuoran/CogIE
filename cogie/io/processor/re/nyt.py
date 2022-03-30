import os
from cogie.utils import load_json
from cogie.core import DataTable
import torch
from transformers import BertTokenizer
from tqdm import tqdm

class NYTREProcessor:
    def __init__(self, path=None, bert_model='bert-base-cased',max_length=128):
        self.path = path
        self.bert_model = bert_model
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.ner_vocabulary=load_json(os.path.join(path,"ner2idx.json"))
        self.rc_vocabulary=load_json(os.path.join(path,"rel2idx.json"))
        self.collate_fn = collater(self.ner_vocabulary, self.rc_vocabulary)


    def process(self,dataset):
        datable = DataTable()
        print("process data...")
        for text, ner_label,rc_label in tqdm(zip(dataset['text'], dataset['ner_label'],dataset['rc_label']),total=len(dataset['text'])):
            words, ner_labels, rc_labels, bert_len= self.process_item(text,
                                                                      ner_label,
                                                                      rc_label)
            datable('words', words)
            datable('ner_labels', ner_labels)
            datable('rc_labels',rc_labels)
            datable('bert_len', bert_len)
        return datable

    def process_item(self,words, ner_labels,rc_labels):
        sent_str = ' '.join(words)
        bert_words = self.tokenizer.tokenize(sent_str)
        bert_len = len(bert_words) + 2
        word_to_bep = self.map_origin_word_to_bert(words)
        ner_labels = self.ner_label_transform(ner_labels, word_to_bep)
        rc_labels = self.rc_label_transform(rc_labels, word_to_bep)

        return (words, ner_labels, rc_labels, bert_len)

    def map_origin_word_to_bert(self, words):
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict

    def ner_label_transform(self, ner_label, word_to_bert):
        new_ner_labels = []

        for i in range(0, len(ner_label)):
            # +1 for [CLS]
            sta = word_to_bert[ner_label[i][0]][0] + 1
            end = word_to_bert[ner_label[i][1]][0] + 1
            new_ner_labels += [sta, end, self.ner_vocabulary[ner_label[i][2]]]

        return new_ner_labels

    def rc_label_transform(self, rc_label, word_to_bert):
        new_rc_labels = []

        for i in range(0, len(rc_label)):
            # +1 for [CLS]
            e1 = word_to_bert[rc_label[i][0]][0] + 1
            e2 = word_to_bert[rc_label[i ][1]][0] + 1
            new_rc_labels += [e1, e2, self.rc_vocabulary[rc_label[i ][2]]]

        return new_rc_labels

class collater():
    def __init__(self, ner2idx, rel2idx):
        self.ner2idx = ner2idx
        self.rel2idx = rel2idx

    def __call__(self, data):
        words = [item[0] for item in data]
        ner_labels = [item[1] for item in data]
        rc_labels = [item[2] for item in data]
        bert_len = [item[3] for item in data]

        batch_size = len(words)

        max_len = max(bert_len)
        ner_labels = [gen_ner_labels(ners, max_len, self.ner2idx) for ners in ner_labels]
        rc_labels = [gen_rc_labels(rcs,max_len, self.rel2idx) for rcs in rc_labels]

        ner_labels = torch.stack(ner_labels, dim=2)
        rc_labels = torch.stack(rc_labels, dim=2)
        mask = mask_to_tensor(bert_len, batch_size)

        return [words,ner_labels,rc_labels,mask]

def gen_ner_labels(ner_list,l, ner2idx):
    labels = torch.FloatTensor(l,l,len(ner2idx)).fill_(0)
    for i in range(0,len(ner_list),3):
        head = ner_list[i]
        tail = ner_list[i+1]
        # n = ner2idx[ner_list[i+2]]
        n=int(ner_list[i+2].item())
        labels[head][tail][n] = 1

    return labels


def gen_rc_labels(rc_list, l, rel2idx):
    labels = torch.FloatTensor(l, l, len(rel2idx)).fill_(0)
    for i in range(0, len(rc_list), 3):
        # e1 = rc_list[i]
        # e2 = rc_list[i + 1]
        # r=rc_list[i + 2]
        # labels[e1][e2][rel2idx[r]] = 1
        e1 = int(rc_list[i].item())
        e2 = int(rc_list[i + 1].item())
        r = int(rc_list[i + 2].item())
        labels[e1][e2][r]= 1

    return labels


def mask_to_tensor(len_list, batch_size):
    token_len = max(len_list)
    tokens = torch.LongTensor(token_len, batch_size).fill_(0)
    for i, s in enumerate(len_list):
        tokens[:s, i] = 1

    return tokens