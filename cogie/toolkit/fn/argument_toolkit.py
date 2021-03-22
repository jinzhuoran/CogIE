"""
@Author: jinzhuan
@File: argument_toolkit.py
@Desc: 
"""
import threading
import torch
from cogie import *
from ..base_toolkit import BaseToolkit


class ArgumentToolkit(BaseToolkit):

    def __init__(self, task='fn', language='english', corpus='argument'):
        config = load_configuration()
        if language == 'english':
            if corpus is None:
                corpus = 'argument'
        self.task = task
        self.language = language
        self.corpus = corpus
        download_model(config[task][language][corpus])
        path = config[task][language][corpus]['path']
        model = config[task][language][corpus]['data']['models']
        trigger_vocabulary = config[task][language][corpus]['data']['frame_vocabulary']
        argument_vocabulary = config[task][language][corpus]['data']['argument_vocabulary']
        bert_model = config[task][language][corpus]['bert_model']
        device = torch.device(config['device'])
        device_ids = config['device_id']
        max_seq_length = config['max_seq_length']
        super().__init__(bert_model, absolute_path(path, model), None, device, device_ids, max_seq_length)
        if self.language == 'english':
            if self.corpus == 'argument':
                self.trigger_vocabulary = Vocabulary.load(absolute_path(path, trigger_vocabulary))
                self.argument_vocabulary = Vocabulary.load(absolute_path(path, argument_vocabulary))
                self.model = Bert4Argument(label_size=len(self.argument_vocabulary),
                                           frame_vocabulary=self.trigger_vocabulary)
        self.load_model()

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(ArgumentToolkit, "_instance"):
            with ArgumentToolkit._instance_lock:
                if not hasattr(ArgumentToolkit, "_instance"):
                    ArgumentToolkit._instance = object.__new__(cls)
        return ArgumentToolkit._instance

    def run(self, words, frames):
        if self.language == 'english':
            if self.corpus == 'argument':
                self.model.eval()
                arguments = []
                import cogie.io.processor.fn.frame_argument as processor
                for frame_item in frames:
                    input_id, attention_mask, segment_id, head_index, label_id, label_mask = \
                        processor.process(words, [], frame_item['frame'], frame_item['position'], self.tokenizer,
                                          self.trigger_vocabulary, self.argument_vocabulary,
                                          self.max_seq_length)
                    input_id = torch.tensor([input_id], dtype=torch.long, device=self.device)
                    attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
                    segment_id = torch.tensor([segment_id], dtype=torch.long, device=self.device)
                    head_index = torch.tensor([head_index], dtype=torch.long, device=self.device)
                    label_id = torch.tensor([label_id], dtype=torch.long, device=self.device)
                    label_mask = torch.tensor([label_mask], dtype=torch.long, device=self.device)
                    frame = torch.tensor([self.trigger_vocabulary.to_index(frame_item['frame'])], dtype=torch.long,
                                         device=self.device)
                    pos = torch.tensor([frame_item['position']], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        prediction, valid_len = self.model.predict(
                            [input_id, attention_mask, segment_id, head_index, label_id, label_mask, frame, pos])
                    if len(prediction) == 0:
                        return []
                    prediction = prediction[0]
                    tag = []
                    for i in range(len(words)):
                        tag.append(self.argument_vocabulary.to_word(prediction[i].item()))
                    argument = _bio_tag_to_spans(words, tag)
                    item = {'frame': frame_item, 'argument': argument}
                    arguments.append(item)
                return arguments


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
    return [{"mention": words[span[1][0]:span[1][1] + 1], "start": span[1][0], "end": span[1][1] + 1, "role": span[0]}
            for span in spans
            if span[0] not in ignore_labels]
