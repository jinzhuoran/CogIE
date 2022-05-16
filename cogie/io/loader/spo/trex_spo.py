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
from tqdm import tqdm



class TrexSpoLoader(Loader):
    def __init__(self,debug=False):
        super().__init__()
        self.debug = debug

    def _load(self, path):
        datas = load_json(path)
        if self.debug:
            datas = datas[0:100]
        dataset = DataTable()
        for data in tqdm(datas):
            text = data['text']
            sentences_boundaries = data['sentences_boundaries']
            words_boundaries = data["words_boundaries"]
            triples = data["triples"]
            if not triples: # if there is no triples
                continue

            prev_length = 0
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
                triples_one_sentence = []
                for triple in triples:
                    if triple["sentence_id"] != i:
                        continue
                    if triple["subject"] is not None and triple["predicate"] is not None and triple["object"] is not None:
                        subject, predicate, object = triple["subject"], triple["predicate"], triple["object"]
                        if subject["boundaries"] is not None and predicate["boundaries"] is not None and object["boundaries"] is not None:
                            # print(triple)
                            keys = ["subject","predicate","object"]
                            for key in keys:
                                start,end = triple[key]["boundaries"]
                                triple[key]["boundaries"] = sorted(list(set([charid2wordid[charid] for charid in range(start,end)])))
                            triples_one_sentence.append({
                                "subject":triple["subject"]["boundaries"],
                                "predicate":triple["predicate"]["boundaries"],
                                "object":triple["object"]["boundaries"],
                            })
                if not triples_one_sentence:
                    continue

                dataset("sentence", sentence)
                dataset("triple",triples_one_sentence)

        return dataset

    def load_all(self, path):
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


