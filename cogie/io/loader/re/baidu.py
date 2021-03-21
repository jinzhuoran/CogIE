"""
@Author: jinzhuan
@File: baidu.py
@Desc: 
"""
import os
from ..loader import Loader
import json
from cogie.core import DataTable


class BaiduRelationLoader(Loader):

    def __init__(self):
        super().__init__()

    def _load(self, path):
        datable = DataTable()
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                text = data['text']
                words = []
                for word in text:
                    words.append(word)
                if 'spo_list' in data:
                    spo_list = data['spo_list']
                    for spo in spo_list:
                        if 'predicate' in spo:
                            relation = spo['predicate']
                            subject = spo['subject']
                            object = spo['object']['@value']
                            subject_pos = get_position(text, subject)
                            object_pos = get_position(text, object)
                            if subject_pos is None or object_pos is None:
                                continue
                            subj_start = subject_pos[0]
                            subj_end = subject_pos[1]
                            obj_start = object_pos[0]
                            obj_end = object_pos[1]
                            self.label_set.add(relation)
                            datable('token', words)
                            datable('relation', relation)
                            datable('subj_start', subj_start)
                            datable('subj_end', subj_end)
                            datable('obj_start', obj_start)
                            datable('obj_end', obj_end)
        return datable

    def load_all(self, path):
        train_path = os.path.join(path, 'train_data.json')
        dev_path = os.path.join(path, 'dev_data.json')
        test_path = os.path.join(path, 'test2_data.json')
        return self._load(train_path), self._load(dev_path), self._load(test_path)


def get_position(text, word):
    start = text.find(word)
    if start == -1:
        return None
    else:
        return [start, start + len(word)]
