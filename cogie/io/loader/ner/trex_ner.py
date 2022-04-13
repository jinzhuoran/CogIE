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
    def __init__(self):
        super().__init__()

    def _load(self, path):
        dataset = DataTable()

        def construct(datas,file_name=None,debug=False):
            sentences = []
            ners = []
            if debug:
                datas = datas[0:100]

            for data in tqdm(datas):
                text = data['text']
                entities = data['entities']
                sentences_boundaries = data['sentences_boundaries']
                words_boundaries = data["words_boundaries"]

                # 修正word跨sentence的情况
                indexes = []
                # pos_word = 0
                # pos_sentence = 0
                # while(pos_word < len(words_boundaries) and pos_sentence < len(sentences_boundaries)):
                #     w_start,w_end = words_boundaries[pos_word]
                #     s_start,s_end = sentences_boundaries[pos_sentence]
                #     if(w_start < s_start and w_end <= s_start):
                #         pos_word += 1
                #     elif(w_start < s_start and w_end > s_start):
                #         indexes.append((pos_word,w_start,s_start,w_end))
                #     elif(w_start >= s_start and w_end < s_end):
                #         pos_word += 1
                #     elif(w_start < s_end and w_end > s_end):
                #         pass
                #     else:
                #         raise ValueError("!!!")

                for s_start,s_end in sentences_boundaries:
                    for idx,(w_start,w_end) in enumerate(words_boundaries):
                        if w_start < s_start and s_start < w_end:
                            indexes.append((idx,w_start,s_start,w_end))
                            break
                for step,(idx,w_start,s_start,w_end) in enumerate(indexes):
                    words_boundaries[step+idx] = [w_start,s_start]
                    words_boundaries.insert(step+idx+1,[s_start,w_end])

                prev_length = 0
                sentences = []
                ners = []
                for i,sentences_boundary in enumerate(sentences_boundaries):
                    charid2wordid = {}
                    sentence = []
                    for j,(start,end) in enumerate(words_boundaries):
                        if start >= sentences_boundary[0] and end <= sentences_boundary[1]:
                            if start == sentences_boundary[0]:
                                # print("j={}  prev_length={}".format(j,prev_length))
                                assert j == prev_length
                            charid2wordid = {**charid2wordid,**{key:j - prev_length for key in range(start,end+1)}}
                            sentence.append(text[start:end])
                    prev_length += len(sentence)
                    sentences.append(sentence)
                    dataset("sentence",sentence)
                    ners_one_sentence = []
                    for entity in entities:
                        entity_boundary = entity["boundaries"]
                        start,end = entity_boundary
                        if start >= sentences_boundary[0] and end <= sentences_boundary[1]:
                            index = list(set([charid2wordid[charid] for charid in range(start,end)]))
                            for k in index:
                                assert k < len(sentence)
                            ner = {"index":index,
                                   "type":"null"}
                            ners_one_sentence.append(ner)
                    ners.append(ners_one_sentence)
                    dataset("ner",ners_one_sentence)

            # dataset("sentence",sentences)
            # dataset("ner",ners)
            # data_dict = {"sentence":sentences,"ner":ners}
            # with open(file_name,"w") as f:
            #     json.dump(data_dict,f)
            return dataset
        datas = load_json(path)
        # train,test = train_test_split(datas,test_size=0.2)
        # val,test = train_test_split(test,test_size=0.5)
        # print("Constructing train...")
        all_dataset = construct(datas,None,debug=True)
        # print("Constructing valid...")
        # construct(val, '../../../cognlp/data/ners/trex/data/valid.json')
        # print("Constructing test...")
        # construct(test, '../../../cognlp/data/ner/trex/data/test.json')

        return all_dataset

    def load_all(self, path):
        dataset = self._load(path)
        train,val,test = dataset.split(8,1,1)
        # datasets = []
        # for f in os.listdir(path):
        #     dataset = self._load(os.path.join(path, f))
        #     datasets.extend(dataset)
        # return datasets
        return [train,val,test]


def get_mention_position(text, sentence_boundary, entity_boundary):
    left_text = text[sentence_boundary[0]:entity_boundary[0]]
    right_text = text[sentence_boundary[0]:entity_boundary[1]]
    left_length = len(nltk.word_tokenize(left_text))
    right_length = len(nltk.word_tokenize(right_text))
    return [left_length, right_length]


