"""
@Author: jinzhuan
@File: ontonotes.py
@Desc: 
"""
import os
from ..loader import Loader
from cogie.core import DataTable


class OntoNotesEtLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, path):
        dataset = DataTable()
        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                words = line.split('\t')
                tokens = words[2].split(' ')
                mentions = words[3].strip().split(' ')
                for mention in mentions:
                    self.label_set.add(mention)
                dataset('words', tokens)
                dataset('mentions', mentions)
                dataset('start', int(words[0]))
                dataset('end', int(words[1]))
        return dataset

    def load_all(self, path):
        train_path = os.path.join(path, 'train.txt')
        dev_path = os.path.join(path, 'dev.txt')
        test_path = os.path.join(path, 'test.txt')
        return self._load(train_path), self._load(dev_path), self._load(test_path)
