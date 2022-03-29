"""
@Author: jinzhuan
@File: kbp37.py
@Desc: 
"""
import os
from ..loader import Loader
from cogie.utils import load_json
from cogie.core import DataTable


class KBP37RELoader(Loader):

    def __init__(self):
        super().__init__()

    def _load(self, path):
        dataset = load_json(path)
        datable = DataTable()
        for data in dataset:
            token = data['token']
            relation = data['relation']
            subj_start = data['subj_start']
            subj_end = data['subj_end']
            obj_start = data['obj_start']
            obj_end = data['obj_end']
            self.label_set.add(relation)
            datable('token', token)
            datable('relation', relation)
            datable('subj_start', subj_start)
            datable('subj_end', subj_end)
            datable('obj_start', obj_start)
            datable('obj_end', obj_end)
        return datable

    def load_all(self, path):
        train_path = os.path.join(path, 'train.json')
        dev_path = os.path.join(path, 'dev.json')
        test_path = os.path.join(path, 'test.json')
        return self._load(train_path), self._load(dev_path), self._load(test_path)
