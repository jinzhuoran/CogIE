"""
@Author: jinzhuan
@File: processor.py
@Desc: 
"""
import os
from cogie.utils import Vocabulary
from transformers import BertTokenizer


class Processor:
    def __init__(self, label_list=None, path=None, padding='<pad>', unknown='<unk>', bert_model='bert-base-cased',
                 max_length=256):
        self.path = path
        self.max_length = max_length
        self.bert_model = bert_model
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        if label_list:
            self.vocabulary = Vocabulary(padding=padding, unknown=unknown)
            self.vocabulary.add_word_lst(label_list)
            self.vocabulary.build_vocab()
            self.save_vocabulary(self.path)
        else:
            self.load_vocabulary(self.path)

    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def get_vocabulary(self):
        return self.vocabulary

    def save_vocabulary(self, path):
        self.vocabulary.save(os.path.join(path, 'vocabulary.txt'))

    def load_vocabulary(self, path):
        self.vocabulary = Vocabulary.load(os.path.join(path, 'vocabulary.txt'))

    def load(self):
        pass
