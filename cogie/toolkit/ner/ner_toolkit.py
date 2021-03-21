"""
@Author: jinzhuan
@File: ner_toolkit.py
@Desc:
"""
from cogie import *
from ..base_toolkit import BaseToolkit
import threading
import torch


class NerToolkit(BaseToolkit):

    def __init__(self, task='ner', language='english', corpus=None):
        config = load_configuration()
        if language == 'chinese':
            if corpus is None:
                corpus = 'msra'
        elif language == 'english':
            if corpus is None:
                corpus = 'trex'
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
            if self.corpus == 'trex':
                self.model = Bert4Ner(len(self.vocabulary))
            elif self.corpus == 'ace2005':
                self.model = BertSoftmax(self.vocabulary)
        elif self.language == 'chinese':
            if self.corpus == 'msra':
                self.model = Bert4CNNer(self.vocabulary)
        self.load_model()

    # _instance_lock = threading.Lock()
    #
    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(NerToolkit, "_instance"):
    #         with NerToolkit._instance_lock:
    #             if not hasattr(NerToolkit, "_instance"):
    #                 NerToolkit._instance = object.__new__(cls)
    #     return NerToolkit._instance

    def run(self, words):
        if self.language == 'english':
            if self.corpus == 'trex':
                self.model.eval()
                labels = ["O"] * len(words)
                import cogie.io.processor.ner.conll2003 as processor
                input_ids, attention_masks, segment_ids, valid_masks, label_ids, label_masks = \
                    processor.process(list(words), labels, self.tokenizer, self.vocabulary, self.max_seq_length)

                input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                attention_masks = torch.tensor([attention_masks], dtype=torch.long, device=self.device)
                segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
                valid_masks = torch.tensor([valid_masks], dtype=torch.long, device=self.device)
                label_ids = torch.tensor([label_ids], dtype=torch.long, device=self.device)
                label_masks = torch.tensor([label_masks], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    prediction, valid_len = self.model.predict(
                        [input_ids, attention_masks, segment_ids, valid_masks, label_ids, label_masks])
                if len(prediction) == 0:
                    return []
                prediction = prediction[0]
                valid_len = valid_len[0]
                tag = []
                for i in range(valid_len.item()):
                    if i != 0 and i != valid_len.item() - 1:
                        tag.append(self.vocabulary.to_word(prediction[i].item()))
                spans = _bio_tag_to_spans(words, tag)
                return spans
            elif self.corpus == 'ace2005':
                self.model.eval()
                labels = ["O"] * len(words)
                import cogie.io.processor.ner.ace2005 as processor
                input_id, attention_mask, segment_id, head_index, label_id, label_mask = \
                    processor.process(list(words), labels, self.tokenizer, self.vocabulary, self.max_seq_length)

                input_id = torch.tensor([input_id], dtype=torch.long, device=self.device)
                attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
                segment_id = torch.tensor([segment_id], dtype=torch.long, device=self.device)
                head_index = torch.tensor([head_index], dtype=torch.long, device=self.device)
                label_id = torch.tensor([label_id], dtype=torch.long, device=self.device)
                label_mask = torch.tensor([label_mask], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    prediction, valid_len = self.model.predict(
                        [input_id, attention_mask, segment_id, head_index, label_id, label_mask])
                if len(prediction) == 0:
                    return []
                prediction = prediction[0]
                valid_len = valid_len[0]
                tag = []
                for i in range(valid_len.item()):
                    tag.append(self.vocabulary.to_word(prediction[i].item()))
                spans = _bio_tag_to_spans(words, tag)
                return spans
        elif self.language == 'chinese':
            if self.corpus == 'msra':
                tokens = []
                for word in words:
                    tokens.append(word)
                self.model.eval()
                labels = ["O"] * len(tokens)
                import cogie.io.processor.ner.msra as processor
                input_id, attention_mask, segment_id, label_id, label_mask = \
                    processor.process(list(tokens), labels, self.tokenizer, self.vocabulary, self.max_seq_length)

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
                tag = []
                for i in range(valid_len.item()):
                    if i != 0 and i != valid_len.item() - 1:
                        tag.append(self.vocabulary.to_word(prediction[i].item()))
                spans = _bio_tag_to_spans(tokens, tag)
                return spans


def _bio_tag_to_spans(words, tags, ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()
    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == "b":
            spans.append((label, [idx, idx]))
        elif bio_tag == "i" and prev_bio_tag in ("b", "i") and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == "o":  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [{"mention": words[span[1][0]:span[1][1] + 1], "start": span[1][0], "end": span[1][1] + 1, "type": span[0]} for span in spans
            if span[0] not in ignore_labels]
