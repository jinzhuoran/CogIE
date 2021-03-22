"""
@Author: jinzhuan
@File: ee_toolkit.py
@Desc: 
"""
from cogie import *
from ..base_toolkit import BaseToolkit
import threading
import torch


class EeToolkit(BaseToolkit):
    def __init__(self, task='ee', language='english', corpus='ace2005'):
        config = load_configuration()
        if language == 'chinese':
            if corpus is None:
                corpus = 'ace2005'
        elif language == 'english':
            if corpus is None:
                corpus = 'ace2005'
        self.task = task
        self.language = language
        self.corpus = corpus
        download_model(config[task][language][corpus])
        path = config[task][language][corpus]['path']
        model = config[task][language][corpus]['data']['models']
        trigger_vocabulary = config[task][language][corpus]['data']['trigger_vocabulary']
        argument_vocabulary = config[task][language][corpus]['data']['argument_vocabulary']
        bert_model = config[task][language][corpus]['bert_model']
        device = torch.device(config['device'])
        device_ids = config['device_id']
        max_seq_length = config['max_seq_length']
        super().__init__(bert_model, absolute_path(path, model), None, device, device_ids, max_seq_length)
        if self.language == 'english':
            if self.corpus == 'ace2005':
                self.trigger_vocabulary = Vocabulary.load(absolute_path(path, trigger_vocabulary))
                self.argument_vocabulary = Vocabulary.load(absolute_path(path, argument_vocabulary))
                self.model = Bert4EE(self.trigger_vocabulary, self.argument_vocabulary)
        self.load_model()

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(EeToolkit, "_instance"):
            with EeToolkit._instance_lock:
                if not hasattr(EeToolkit, "_instance"):
                    EeToolkit._instance = object.__new__(cls)
        return EeToolkit._instance

    def run(self, words, spans):
        if self.language == 'english':
            if self.corpus == 'ace2005':
                self.model.eval()
                arguments = {
                    "candidates": [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    "events": {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }
                candidates = []
                for span in spans:
                    candidates.append(tuple([span["start"], span["end"], None]))
                arguments["candidates"] = candidates
                import cogie.io.processor.ee.ace2005 as processor
                tokens_x, triggers_y, arguments, head_indexes, _, triggers = processor.process(
                    list(["[CLS]"] + words + ["[SEP]"]), [], arguments,
                    self.tokenizer,
                    self.trigger_vocabulary,
                    self.max_seq_length)
                with torch.no_grad():
                    predictions = self.model.predict([[tokens_x], [head_indexes], [triggers_y], [arguments], words, triggers])

                events = []
                if len(predictions) == 0:
                    return events
                predictions = predictions[0]["events"]

                for trigger, argument in predictions.items():
                    trigger = {"start": trigger[0], "end": trigger[1], "label": trigger[2],
                               "mention": words[trigger[0]:trigger[1]]}
                    arguments = []
                    event = {"trigger": trigger, "arguments": arguments}
                    for item in argument:
                        arguments.append({"start": item[0], "end": item[1], "label": self.argument_vocabulary.to_word(item[2]),
                                          "mention": words[item[0]:item[1]]})
                    events.append(event)
                return events
