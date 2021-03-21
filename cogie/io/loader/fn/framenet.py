"""
@Author: jinzhuan
@File: framenet.py
@Desc: 
"""
import os
from ..loader import Loader


class FrameNetLoader(Loader):
    def __init__(self):
        super().__init__()
        self.frame_set = set()
        self.element_set = set()

    def _load(self, path):
        dataset = {}
        sentence = []
        frame = []
        element = []
        with open(path) as f:
            for line in f:
                if len(line) == 0 or line[0] == "\n":
                    if len(words) > 0:
                        if "".join(sentence) not in dataset:
                            data = {'sentence': sentence, 'frames': [], 'elements': []}
                            dataset["".join(sentence)] = data
                        dataset["".join(sentence)]['frames'].append(frame)
                        dataset["".join(sentence)]['elements'].append(element)
                        sentence = []
                        frame = []
                        element = []
                    continue
                words = line.split('\t')
                sentence.append(words[1])
                if words[-3] not in '_':
                    frame.append(words[-3])
                    self.frame_set.add(words[-3])
                else:
                    frame.append('<unk>')

                element.append(words[-2])
                self.element_set.add(words[-2])

            if len(sentence) > 0:
                if "".join(sentence) not in dataset:
                    data = {'sentence': sentence, 'frames': [], 'elements': []}
                    dataset["".join(sentence)] = data
                dataset["".join(sentence)]['frames'].append(frame)
                dataset["".join(sentence)]['elements'].append(element)

        return dataset

    def load_all(self, path):
        train_set = self._load(os.path.join(path, 'train.bios'))
        dev_set = self._load(os.path.join(path, 'dev.bios'))
        test_set = self._load(os.path.join(path, 'test.bios'))
        return train_set, dev_set, test_set

    def get_frame_labels(self):
        labels = list(self.frame_set)
        labels.sort()
        return labels

    def get_element_labels(self):
        labels = list(self.element_set)
        labels.sort()
        return labels
