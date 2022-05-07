"""
@Author: jinzhuan
@File: conll2003.py
@Desc: 
"""
import os
from cogie.utils import load_json
from ..loader import Loader
from cogie.core.datable import DataTable


class Conll2003NERLoader(Loader):
    def __init__(self):
        super().__init__()

    def _load(self, path):
        dataset = DataTable()
        sentence = []
        label = []
        with open(path) as f:
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        dataset('sentence', sentence)
                        dataset('label', label)
                        sentence = []
                        label = []
                    continue
                words = line.split(' ')
                sentence.append(words[0])
                label.append(words[-1][:-1])
                self.label_set.add(words[-1][:-1])
            if len(sentence) > 0:
                dataset('sentence', sentence)
                dataset('label', label)
            if len(dataset) == 0:
                raise RuntimeError("No data found {}.".format(path))
        return dataset

    def load_all(self, path):
        train_set = self._load(os.path.join(path, 'train.txt'))
        dev_set = self._load(os.path.join(path, 'dev.txt'))
        test_set = self._load(os.path.join(path, 'test.txt'))
        return train_set, dev_set, test_set

    def get_labels(self):
        labels = list(self.label_set)
        labels.sort()
        return labels


class TrexNerLoader:
    def __init__(self):
        self.label_set = set()
        self.label_set.add('B')
        self.label_set.add('I')
        self.label_set.add('O')

    def _load(self, path):
        dataset = load_json(path)
        for data in dataset:
            triples = data['triples']
            for triple in triples:
                self.label_set.add(triple['predicate']['uri'])
        return dataset

    def load(self, path):
        train_set = self._load(os.path.join(path, 'train.txt'))
        dev_set = self._load(os.path.join(path, 'dev.txt'))
        test_set = self._load(os.path.join(path, 'test.txt'))
        return train_set, dev_set, test_set

    def load_all(self, path):
        datasets = []
        for f in os.listdir(path):
            if f == 'vocabulary.txt':
                continue
            dataset = load_json(os.path.join(path, f))
            for data in dataset:
                entities = data['entities']
                for entity in entities:
                    entity
            datasets.extend(dataset)
        return datasets

    def load_train(self, path):
        train_set = self._load(os.path.join(path, 'train.json'))
        return train_set

    def load_data(self, file):
        data_set = self._load(file)
        return data_set

    def get_labels(self):
        labels = list(self.label_set)
        labels.sort()
        return labels
