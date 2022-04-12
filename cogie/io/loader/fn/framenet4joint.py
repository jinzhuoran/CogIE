import os
from ..loader import Loader
import json
from cogie.core import DataTable

class FrameNet4JointLoader(Loader):
    def __init__(self):
        super().__init__()
        self.node_types_set = set()
        self.node_attrs_set = set()
        self.p2p_edges_set=set()
        self.p2r_edges_set=set()
        self.node_types_set.add('O')
        self.node_attrs_set.add('O')

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
                for item in sample["node_types"]:
                    self.node_types_set.add(item[1])
                for item in sample["node_attrs"]:
                    self.node_attrs_set.add(item[1])
                for item in sample["p2p_edges"]:
                    self.p2p_edges_set.add(item[-1])
                for item in sample["p2r_edges"]:
                    self.p2r_edges_set.add(item[-1])
        return dataset

    def load_all(self, path):
        train_set = self._load(os.path.join(path, 'traindebug.json'))
        dev_set = self._load(os.path.join(path, 'devdebug.json'))
        test_set = self._load(os.path.join(path, 'testdebug.json'))
        return train_set, dev_set, test_set

    def get_node_types_labels(self):
        labels = list(self.node_types_set)
        labels.sort()
        return labels

    def get_node_attrs_labels(self):
        labels = list(self.node_attrs_set)
        labels.sort()
        return labels

    def get_p2p_edges_labels(self):
        labels = list(self.p2p_edges_set)
        labels.sort()
        return labels

    def get_p2r_edges_labels(self):
        labels = list(self.p2r_edges_set)
        labels.sort()
        return labels
