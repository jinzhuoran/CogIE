"""
@Author: jinzhuan
@File: msra.py
@Desc: 
"""
import os
from ..loader import Loader


class MSRALoader(Loader):

    def __init__(self):
        super().__init__()
        self.label_set.add('B')
        self.label_set.add('M')
        self.label_set.add('E')
        self.label_set.add('S')

    def _load(self, path):
        dataset = []
        with open(path) as f:
            while True:
                data = {'sentence': "", 'words': [], 'labels': []}
                line = f.readline().strip()
                if not line:
                    break
                words = line.split('  ')
                labels = []
                sentence = ""
                for word in words:
                    sentence += word
                    if len(word) == 1:
                        labels.append('S')
                    elif len(word) > 1:
                        for i in range(len(word)):
                            if i == 0:
                                labels.append('B')
                            elif i == len(word) - 1:
                                labels.append('E')
                            else:
                                labels.append('M')
                data['sentence'] = sentence
                data['words'] = words
                data['labels'] = labels
                dataset.append(data)
        return dataset

    def load_all(self, path):
        train_path = os.path.join(path, 'msr_training.utf8')
        dev_path = os.path.join(path, 'msr_test_gold.utf8')
        test_path = os.path.join(path, 'msr_test_gold.utf8')
        return self._load(train_path), self._load(dev_path), self._load(test_path)
