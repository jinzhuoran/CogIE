"""
@Author: jinzhuan
@File: trex.py
@Desc:
"""
import os
from ..loader import Loader
from cogie.utils import load_json
import nltk
from cogie.core.datable import DataTable
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm



class TrexNerLoader(Loader):
    def __init__(self,debug=False):
        super().__init__()
        self.debug = debug

    def _load(self, path):
        datas = load_json(path)
        # datas = datas[0:int(len(datas)/2000)]
        if self.debug:
            datas = datas[0:100]
        dataset = DataTable()
        for data in tqdm(datas):
            text = data['text']
            entities = data['entities']
            sentences_boundaries = data['sentences_boundaries']
            words_boundaries = data["words_boundaries"]

            prev_length = 0
            sentences = []
            ners = []
            for i, sentences_boundary in enumerate(sentences_boundaries):
                charid2wordid = {}
                sentence = []
                for j, (start, end) in enumerate(words_boundaries):
                    if start >= sentences_boundary[0] and end <= sentences_boundary[1]:
                        if start == sentences_boundary[0]:
                            # print("j={}  prev_length={}".format(j,prev_length))
                            assert j == prev_length
                        charid2wordid = {**charid2wordid, **{key: j - prev_length for key in range(start, end + 1)}}
                        sentence.append(text[start:end])
                prev_length += len(sentence)
                sentences.append(sentence)
                dataset("sentence", sentence)
                ners_one_sentence = []
                for entity in entities:
                    entity_boundary = entity["boundaries"]
                    start, end = entity_boundary
                    if start >= sentences_boundary[0] and end <= sentences_boundary[1]:
                        index = list(set([charid2wordid[charid] for charid in range(start, end)]))
                        for k in index:
                            assert k < len(sentence)
                        ner = {"index": index,
                               "type": "null"}
                        ners_one_sentence.append(ner)
                ners.append(ners_one_sentence)
                dataset("ner", ners_one_sentence)

        return dataset

    def load_all(self, path):
        label_list = ["null"]
        self.label_set = set(label_list)
        train_set = self._load(os.path.join(path, 'train.json'))
        dev_set = self._load(os.path.join(path, 'dev.json'))
        test_set = self._load(os.path.join(path, 'test.json'))
        return [train_set,dev_set,test_set]


def get_mention_position(text, sentence_boundary, entity_boundary):
    left_text = text[sentence_boundary[0]:entity_boundary[0]]
    right_text = text[sentence_boundary[0]:entity_boundary[1]]
    left_length = len(nltk.word_tokenize(left_text))
    right_length = len(nltk.word_tokenize(right_text))
    return [left_length, right_length]


