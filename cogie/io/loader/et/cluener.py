"""
@Author: jinzhuan
@File: cluener.py
@Desc: 
"""
import os
from cogie.core import DataTable
import json
from ..loader import Loader


class BBNEtLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, path):
        dataset = DataTable()
        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                data = json.loads(line)
                dataset('words', data['tokens'])
                dataset('mentions', data['mentions'])
                mentions = data['mentions']
                for mention in mentions:
                    for label in mention['labels']:
                        self.label_set.add(label)
        return dataset

    def load_all(self, path):
        train_path = os.path.join(path, 'train.json')
        test_path = os.path.join(path, 'test.json')
        return self._load(train_path), self._load(test_path)
