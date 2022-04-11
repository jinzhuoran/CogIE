"""
@Author: jinzhuan
@File: ace2005.py
@Desc:
"""
import os
from ..loader import Loader
from cogie.utils import load_json
from collections import defaultdict



class ACE2005GPLinkerLoader:
    """
    The ace2005 dataset processing follows https://github.com/nlpcl-lab/ace2005-preprocessing
    """

    def __init__(self):
        self.label_set = set()
        self.label_set = set()

    def _load(self, path):
        data = load_json(path)
        for sample in data:
            if len(sample["golden-event-mentions"]) > 0:
                for event in sample["golden-event-mentions"]:
                    event_type = event["event_type"]
                    for argument in event["arguments"]:
                        role = argument["role"]
                        if (event_type,role) not in self.label_set:
                            self.label_set.add((event_type,role))
                        if (event_type,"trigger") not in self.label_set:
                            self.label_set.add((event_type,"trigger"))
        return data

    def load_all(self, path):
        train_path = os.path.join(path, 'train.json')
        dev_path = os.path.join(path, 'dev.json')
        test_path = os.path.join(path, 'test.json')
        return self._load(train_path), self._load(dev_path), self._load(test_path)

    def get_trigger_labels(self):
        labels = list(self.trigger_label_set)
        labels.sort()
        return labels

    def get_argument_labels(self):
        labels = list(self.argument_label_set)
        labels.sort()
        return labels
