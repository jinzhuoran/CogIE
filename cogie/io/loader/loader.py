"""
@Author: jinzhuan
@File: loader.py
@Desc: 
"""


class Loader:
    def __init__(self):
        self.label_set = set()

    def _load(self, path):
        pass

    def load(self, path):
        pass

    def load_all(self, path):
        pass

    def load_train(self, path):
        pass

    def load_dev(self, path):
        pass

    def load_test(self, path):
        pass

    def get_labels(self):
        labels = list(self.label_set)
        labels.sort()
        return labels
