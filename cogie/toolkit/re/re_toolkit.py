"""
@Author: jinzhuan
@File: re_toolkit.py
@Desc: 
"""
from cogie import *
from ..base_toolkit import BaseToolkit
import threading
import torch


class ReToolkit(BaseToolkit):
    def __init__(self, task='re', language='english', corpus=None):
        config = load_configuration()
        if language == 'chinese':
            if corpus is None:
                corpus = 'baidu'
        elif language == 'english':
            if corpus is None:
                corpus = 'trex'
        self.task = task
        self.language = language
        self.corpus = corpus
        download_model(config[task][language][corpus])
        path = config[task][language][corpus]['path']
        model = config[task][language][corpus]['data']['models']
        vocabulary = config[task][language][corpus]['data']['vocabulary']
        bert_model = config[task][language][corpus]['bert_model']
        device = torch.device(config['device'])
        device_ids = config['device_id']
        max_seq_length = config['max_seq_length']
        super().__init__(bert_model, absolute_path(path, model), absolute_path(path, vocabulary), device, device_ids,
                         max_seq_length)
        if self.language == 'english':
            if self.corpus == 'trex':
                self.model = Bert4Re(self.vocabulary)
        self.load_model()

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(ReToolkit, "_instance"):
            with ReToolkit._instance_lock:
                if not hasattr(ReToolkit, "_instance"):
                    ReToolkit._instance = object.__new__(cls)
        return ReToolkit._instance

    def run(self, words, spans):
        if self.language == 'english':
            if self.corpus == 'trex':
                self.model.eval()
                for span in spans:
                    span["position"] = [span["start"], span["end"]]
                import cogie.io.processor.rc.trex as processor
                input_ids, attention_mask, head_indexes, entity_mentions, relation_mentions, entity_mentions_mask, relation_mentions_mask = \
                    processor.process(words, spans, [], self.tokenizer, self.vocabulary, self.max_seq_length)
                input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
                head_indexes = torch.tensor([head_indexes], dtype=torch.long, device=self.device)
                entity_mentions = torch.tensor([entity_mentions], dtype=torch.long, device=self.device)
                relation_mentions = torch.tensor([relation_mentions], dtype=torch.long, device=self.device)
                entity_mentions_mask = torch.tensor([entity_mentions_mask], dtype=torch.long, device=self.device)
                relation_mentions_mask = torch.tensor([relation_mentions_mask], dtype=torch.long,
                                                      device=self.device)

                with torch.no_grad():
                    outputs = self.model.predict(
                        [input_ids, attention_mask, head_indexes, entity_mentions, relation_mentions,
                         entity_mentions_mask, relation_mentions_mask])

                relations = []
                for output in outputs:

                    for span in spans:
                        if span["start"] == output[1] and span["end"] == output[2]:
                            head_entity_mention = span["mention"]
                        if span["start"] == output[3] and span["end"] == output[4]:
                            tail_entity_mention = span["mention"]
                    head_entity = {"start": output[1], "end": output[2], "mention": head_entity_mention}
                    tail_entity = {"start": output[3], "end": output[4], "mention": tail_entity_mention}
                    relation = {"head_entity": head_entity,
                                "tail_entity": tail_entity,
                                "relations": [self.vocabulary.to_word(output[-1])]
                                }
                    relations.append(relation)
                return relations
