"""
@Author: jinzhuan
@File: ontonotes_cn.py
@Desc: 
"""
import os
from ..loader import Loader
from cogie.core import DataTable


class OntoNotesNerLoader(Loader):

    def __init__(self):
        super().__init__()

    def _load(self, path):
        dataset = DataTable()
        sentence = []
        label = []
        with open(path) as f:
            for line in f:
                if len(line) == 0 or line[0] == "\n":
                    if len(sentence) > 0:
                        dataset('sentence', sentence)
                        dataset('label', label)
                        sentence = []
                        label = []
                    continue
                line = line.strip()
                words = line.split(' ')
                sentence.append(words[0])
                label.append(words[-1])
                self.label_set.add(words[-1])
            if len(sentence) > 0:
                dataset('sentence', sentence)
                dataset('label', label)
            if len(dataset) == 0:
                raise RuntimeError("No data found {}.".format(path))
        return dataset

    def load_all(self, path):
        train_path = os.path.join(path, 'train.char.bmes')
        test_path = os.path.join(path, 'test.char.bmes')
        return self._load(train_path), self._load(test_path)

    def to_I(self, label):
        return label
