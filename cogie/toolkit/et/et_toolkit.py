"""
@Author: jinzhuan
@File: et_toolkit.py
@Desc: 
"""
from cogie import *
from ..base_toolkit import BaseToolkit
import threading
import torch


class EtToolkit(BaseToolkit):
    def __init__(self, task='et', language='english', corpus=None):
        config = load_configuration()
        if language == 'chinese':
            if corpus is None:
                corpus = 'cluener'
        elif language == 'english':
            if corpus is None:
                corpus = 'ontonotes'
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
            if self.corpus == 'ontonotes':
                self.model = Bert4Et(len(self.vocabulary))
        self.load_model()

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(EtToolkit, "_instance"):
            with EtToolkit._instance_lock:
                if not hasattr(EtToolkit, "_instance"):
                    EtToolkit._instance = object.__new__(cls)
        return EtToolkit._instance

    def run(self, words, spans):
        if self.language == 'english':
            if self.corpus == 'ontonotes':
                self.model.eval()
                labels = ["<unk>"] * len(words)
                entities = []
                import cogie.io.processor.et.ontonotes as processor
                for i in range(len(spans)):
                    input_ids, attention_mask, start_pos, end_pos, label_ids = \
                        processor.process(list(words), spans[i]["start"], spans[i]["end"], labels, self.vocabulary,
                                          self.tokenizer,
                                          self.max_seq_length)
                    input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                    attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
                    start_pos = torch.tensor([start_pos], dtype=torch.long, device=self.device)
                    end_pos = torch.tensor([end_pos], dtype=torch.long, device=self.device)
                    label_ids = torch.tensor([label_ids], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        output = self.model.predict([input_ids, attention_mask, start_pos, end_pos, label_ids])
                    if len(output) == 0:
                        return []
                    output = output[0]
                    prediction = []
                    for j in range(len(output)):
                        if output[j] == 1:
                            prediction.append(self.vocabulary.to_word(j))
                    entities.append({"mention": words[spans[i]["start"]:spans[i]["end"]], "start": spans[i]["start"],
                                     "end": spans[i]["end"], "types": prediction})
                return entities
