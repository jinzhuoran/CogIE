import os
from ..loader import Loader
import json
from cogie.core import DataTable

class FrameNet4JointLoader(Loader):
    def __init__(self):
        super().__init__()
        # 待增加标签
        self.frame_set = set()
        self.element_set = set()

    def _load(self, path):
        dataset = DataTable()
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                sample = json.loads(line)
                dataset("words", sample["sentence"])
                dataset("lemma", sample["lemmas"])
                dataset("node_types", sample["node_types"])
                dataset("node_attrs", sample["node_attrs"])
                dataset("origin_lexical_units", sample["origin_lexical_units"])
                dataset("p2p_edges", sample["p2p_edges"])
                dataset("p2r_edges", sample["p2r_edges"])
                dataset("origin_frames", sample["origin_frames"])
                dataset("frame_elements", sample["frame_elements"])
        return dataset

    def load_all(self, path):
        train_set = self._load(os.path.join(path, 'traindebug.json'))
        dev_set = self._load(os.path.join(path, 'devdebug.json'))
        test_set = self._load(os.path.join(path, 'testdebug.json'))
        return train_set, dev_set, test_set

    def get_frame_labels(self):
        labels = list(self.frame_set)
        labels.sort()
        return labels

    def get_element_labels(self):
        labels = list(self.element_set)
        labels.sort()
        return labels
