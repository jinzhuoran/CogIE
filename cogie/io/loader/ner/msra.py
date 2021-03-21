"""
@Author: jinzhuan
@File: msra.py
@Desc: 
"""
import os
from ..loader import Loader


class MSRANerLoader(Loader):

    def __init__(self):
        super().__init__()

    def _load(self, path):
        dataset = []
        sentence = []
        label = []
        with open(path) as f:
            for line in f:
                if len(line) == 0 or line[0] == "\n":
                    if len(sentence) > 0:
                        data = {'words': sentence, 'labels': label}
                        dataset.append(data)
                        sentence = []
                        label = []
                    continue
                words = line.split(' ')
                sentence.append(words[0])
                label.append(words[-1][:-1])
                self.label_set.add(words[-1][:-1])
            if len(sentence) > 0:
                data = {'words': sentence, 'labels': label}
                dataset.append(data)
            if len(dataset) == 0:
                raise RuntimeError("No data found {}.".format(path))
        return dataset

    def load_all(self, path):
        train_path = os.path.join(path, 'MSRA.train')
        dev_path = os.path.join(path, 'MSRA.test')
        test_path = os.path.join(path, 'MSRA.test')
        return self._load(train_path), self._load(dev_path), self._load(test_path)
