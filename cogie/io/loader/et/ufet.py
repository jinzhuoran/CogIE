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



class UfetLoader(Loader):
    def __init__(self):
        super().__init__()
        self.context_window_size = 100

    def _load(self, file_path):
        dataset = DataTable()
        with open(file_path) as f:
            for line in tqdm(f):
                line = json.loads(line.strip())
                ex_id = line["ex_id"]
                left_context = list(map(lambda x:x.lower(),line["left_context"][-self.context_window_size:]))
                right_context = list(map(lambda x:x.lower(),line["right_context"][:self.context_window_size]))
                mention = list(map(lambda x:x.lower(),line["mention_as_list"]))
                label = line["y_category"]
                self.label_set.update(label)
                dataset("ex_id",ex_id)
                dataset("left_context",left_context)
                dataset("right_context",right_context)
                dataset("mention",mention)
                dataset("label",label)
        return dataset

    def load_all(self, path):
        train_set = self._load(os.path.join(path, 'ufet_train.json'))
        dev_set = self._load(os.path.join(path, 'ufet_dev.json'))
        test_set = self._load(os.path.join(path, 'ufet_test.json'))
        return train_set, dev_set, test_set



