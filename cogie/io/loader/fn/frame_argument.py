"""
@Author: jinzhuan
@File: frame_argument.py
@Desc: 
"""
import os
from cogie.core import DataTable
from ..loader import Loader


class FrameArgumentLoader(Loader):
    def __init__(self):
        super().__init__()
        self.trigger_label_set = set()
        self.argument_label_set = set()

    def _load(self, path):
        dataset = DataTable()
        sentence = []
        label = []
        frame = -1
        pos = -1
        with open(path) as f:
            for line in f:
                if len(line) == 0 or line[0] == "\n":
                    if len(sentence) > 0:
                        dataset('sentence', sentence)
                        dataset('label', label)
                        dataset('frame', frame)
                        dataset('pos', pos)
                        sentence = []
                        label = []
                        frame = -1
                        pos = -1
                    continue
                words = line.split('\t')
                sentence.append(words[1])
                element = words[-2].replace('S-', 'B-')
                label.append(element)
                if words[-3] not in '_':
                    pos = len(sentence) - 1
                    frame = words[-3]
                    self.trigger_label_set.add(frame)

                self.argument_label_set.add(element)
            if len(sentence) > 0:
                dataset('sentence', sentence)
                dataset('label', label)
                dataset('frame', frame)
                dataset('pos', pos)
        return dataset

    def load_all(self, path):
        train_set = self._load(os.path.join(path, 'train.bios'))
        dev_set = self._load(os.path.join(path, 'dev.bios'))
        test_set = self._load(os.path.join(path, 'test.bios'))
        return train_set, dev_set, test_set

    def get_trigger_labels(self):
        labels = list(self.trigger_label_set)
        labels.sort()
        return labels

    def get_argument_labels(self):
        labels = list(self.argument_label_set)
        labels.sort()
        return labels
