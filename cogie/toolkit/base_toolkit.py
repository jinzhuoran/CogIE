"""
@Author: jinzhuan
@File: base_toolkit.py
@Desc: 
"""
from transformers import BertTokenizer
from cogie.utils import module2parallel, load_model, Vocabulary


class BaseToolkit:
    def __init__(self, bert_model=None, model_path=None, vocabulary_path=None, device=None,
                 device_ids=None, max_seq_length=256):
        super().__init__()
        self.bert_model = bert_model
        self.model_path = model_path
        self.vocabulary_path = vocabulary_path
        self.device = device
        self.device_ids = device_ids
        self.max_seq_length = max_seq_length

        if self.bert_model:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        if self.vocabulary_path:
            self.vocabulary = Vocabulary.load(self.vocabulary_path)

    def load_model(self):
        if self.model_path:
            self.model = load_model(self.model, self.model_path)
        if self.device_ids:
            self.model = module2parallel(self.model, self.device_ids)

    def run(self):
        pass

    def to_interface(self):
        pass

    def to_data(self, data):
        return {'data', data}
