"""
@Author: jinzhuan
@File: ace2005.py
@Desc: 
"""
import os
from ..loader import Loader
from cogie.utils import load_json


class ACE2005NerLoader(Loader):

    def __init__(self):
        super().__init__()
        self.label_set.add('O')

    def _load(self, path):
        data = load_json(path)
        for item in data:
            for entity_mention in item['golden-entity-mentions']:
                for i in range(entity_mention['start'], entity_mention['end']):
                    entity_type = entity_mention['entity-type']
                    if i == entity_mention['start']:
                        self.label_set.add('B-{}'.format(entity_type))
                    else:
                        self.label_set.add('I-{}'.format(entity_type))
        return data

    def load_all(self, path):
        train_path = os.path.join(path, 'train.json')
        dev_path = os.path.join(path, 'dev.json')
        test_path = os.path.join(path, 'test.json')
        return self._load(train_path), self._load(dev_path), self._load(test_path)
