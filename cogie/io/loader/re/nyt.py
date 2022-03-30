import os
from ..loader import Loader
from cogie.utils import load_json
from cogie.core import DataTable

class NYTRELoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, path):
        dataset = DataTable()
        data=load_json(path)
        for item in data:
            ner_label=[]
            rc_label=[]
            ner_check=[]
            rc_check=[]
            text=item["text"].split(" ")
            for label in item["triple_list"]:
                subject_word_loc=text.index(label[0])
                relation=label[1]
                object_word_loc=text.index(label[2])
                if subject_word_loc not in ner_check:
                    ner_label.append([subject_word_loc, subject_word_loc, "None"])
                    ner_check+=[subject_word_loc, subject_word_loc, "None"]
                if object_word_loc not in ner_check:
                    ner_label.append([object_word_loc,object_word_loc,"None"])
                    ner_check += [object_word_loc,object_word_loc,"None"]
                rc_label.append([subject_word_loc,object_word_loc,relation])
            dataset("text",text)
            dataset("ner_label",ner_label)
            dataset("rc_label",rc_label)
        return dataset


    def load_all(self, path):
        train_set = self._load(os.path.join(path, 'train_triples.json'))
        dev_set = self._load(os.path.join(path, 'dev_triples.json'))
        test_set = self._load(os.path.join(path, 'test_triples.json'))
        return train_set, dev_set, test_set