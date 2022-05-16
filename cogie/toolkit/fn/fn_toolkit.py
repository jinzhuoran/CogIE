"""
@Author: jinzhuan
@File: fn_toolkit.py
@Desc: 
"""
import threading
import torch
from cogie import *
from ..base_toolkit import BaseToolkit


class FnToolkit(BaseToolkit):

    def __init__(self, task='fn', language='english', corpus='frame'):
        config = load_configuration()
        if language == 'english':
            if corpus is None:
                corpus = 'frame'
        self.task = task
        self.language = language
        self.corpus = corpus
        download_model(config[task][language][corpus])
        path = config[task][language][corpus]['path']
        model = config[task][language][corpus]['data']['model']
        vocabulary = config[task][language][corpus]['data']['vocabulary']
        bert_model = config[task][language][corpus]['bert_model']
        device = torch.device(config['device'])
        device_ids = config['device_id']
        max_seq_length = config['max_seq_length']
        super().__init__(bert_model, absolute_path(path, model), absolute_path(path, vocabulary), device, device_ids,
                         max_seq_length)
        if self.language == 'english':
            if self.corpus == 'frame':
                self.model = Bert4Frame(len(self.vocabulary))
        self.load_model()

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(FnToolkit, "_instance"):
            with FnToolkit._instance_lock:
                if not hasattr(FnToolkit, "_instance"):
                    FnToolkit._instance = object.__new__(cls)
        return FnToolkit._instance

    def run(self, words):
        if self.language == 'english':
            if self.corpus == 'frame':
                self.model.eval()
                frames = []
                import cogie.io.processor.fn.framenet as processor
                input_ids, attention_mask, head_indexes, frame_id, element_id, label_mask = \
                    processor.process(words, [], [], self.tokenizer, self.vocabulary, None, self.max_seq_length)
                with torch.no_grad():
                    prediction, valid_len = self.model.predict(
                        [[input_ids], [attention_mask], [head_indexes], [frame_id], [element_id], [label_mask]])
                if len(prediction) == 0:
                    return []
                prediction = prediction[0]
                valid_len = valid_len[0]

                for i in range(valid_len.item()):
                    if prediction[i].item() != self.vocabulary.to_index("<unk>"):
                        frame = {"word": words[i],
                                 "position": i,
                                 "frame": self.vocabulary.to_word(prediction[i].item())}
                        frames.append(frame)
                return frames
