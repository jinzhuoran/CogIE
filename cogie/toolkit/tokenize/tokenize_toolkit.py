"""
@Author: jinzhuan
@File: tokenize_toolkit.py
@Desc: 
"""
import torch
import nltk
from ..base_toolkit import BaseToolkit
import threading
from cogie import load_configuration, download_model, absolute_path, Bert4WS


class TokenizeToolkit(BaseToolkit):

    def __init__(self, task='ws', language='english', corpus=None):

        config = load_configuration()
        if language == 'chinese':
            if corpus is None:
                corpus = 'msra'
        elif language == 'english':
            if corpus is None:
                corpus = 'nltk'
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
            pass
        elif self.language == 'chinese':
            self.model = Bert4WS(self.vocabulary)
            self.load_model()

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(TokenizeToolkit, "_instance"):
            with TokenizeToolkit._instance_lock:
                if not hasattr(TokenizeToolkit, "_instance"):
                    TokenizeToolkit._instance = object.__new__(cls)
        return TokenizeToolkit._instance

    def run(self, sentence):
        if self.language == 'english':
            words = nltk.word_tokenize(sentence)
            return words
        elif self.language == 'chinese':
            self.model.eval()
            words = []
            for word in sentence:
                words.append(word)
            labels = ["S"] * len(words)
            import cogie.io.processor.ws.msra as processor
            input_id, attention_mask, segment_id, label_id, label_mask = processor.process(words, labels,
                                                                                           self.tokenizer,
                                                                                           self.vocabulary,
                                                                                           self.max_seq_length)
            input_id = torch.tensor([input_id], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
            segment_id = torch.tensor([segment_id], dtype=torch.long, device=self.device)
            label_id = torch.tensor([label_id], dtype=torch.long, device=self.device)
            label_mask = torch.tensor([label_mask], dtype=torch.long, device=self.device)

            with torch.no_grad():
                prediction, valid_len = self.model.predict(
                    [input_id, attention_mask, segment_id, label_id, label_mask])
            if len(prediction) == 0:
                return []
            prediction = prediction[0]
            valid_len = valid_len[0]
            tags = []
            for i in range(valid_len.item()):
                if i != 0 and i != valid_len.item() - 1:
                    tags.append(self.vocabulary.to_word(prediction[i].item()))
            spans = _bmes_tag_to_spans(sentence, tags)
            return spans


def _bmes_tag_to_spans(sentence, tags):
    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ("b", "s"):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ("m", "e") and prev_bmes_tag in ("b", "m") and label == spans[-1][0]:
            spans[-1][1][1] = idx
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(sentence[span[1][0]:span[1][1] + 1], span[1][0], span[1][1] + 1) for span in spans]
