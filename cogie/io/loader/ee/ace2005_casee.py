# import os
# import json
# from cogie.core import DataTable
# class ACE2005CASEELoader:
#     def __init__(self):
#         pass
#
#     def _load(self, path):
#         dataset = DataTable()
#         with open(path) as f:
#             lines = f.readlines()
#             for line in lines:
#                 sample = json.loads(line)
#                 dataset("content", sample["content"])
#                 dataset("index", sample["index"])
#                 dataset("type", sample["type"])
#                 dataset("args", sample["args"])
#                 dataset("occur", sample["occur"])
#                 dataset("triggers", sample["triggers"])
#                 dataset("id", sample["id"])
#         return dataset
#
#     def load_all(self, path):
#         # train_path = os.path.join(path, 'okok.json')
#         # dev_path = os.path.join(path, 'okok.json')
#         # test_path = os.path.join(path, 'okok.json')
#         train_path = os.path.join(path, 'train.json')
#         dev_path = os.path.join(path, 'dev.json')
#         test_path = os.path.join(path, 'test.json')
#         return self._load(train_path), self._load(dev_path), self._load(test_path)

import os
import json
from cogie.core import DataTable
class ACE2005CASEELoader:
    def __init__(self):
        pass

    def _load(self, path):
        dataset = DataTable()
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                sample = json.loads(line)
                dataset("content", sample["content"])
                dataset("index", sample["index"])
                dataset("type", sample["type"])
                dataset("args", sample["args"])
                dataset("occur", sample["occur"])
                dataset("triggers", sample["triggers"])
                dataset("id", sample["id"])
        return dataset

    def load_all(self, path):
        train_path = os.path.join(path, 'train.json')
        dev_path = os.path.join(path, 'dev.json')
        test_path = os.path.join(path, 'test.json')
        return self._load(train_path), self._load(dev_path), self._load(test_path)