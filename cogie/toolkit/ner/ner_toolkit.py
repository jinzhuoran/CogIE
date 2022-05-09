"""
@Author: jinzhuan
@File: ner_toolkit.py
@Desc:
"""
from cogie import *
from cogie.models.ner.w2ner import W2NER
from ..base_toolkit import BaseToolkit
import threading
import torch
import json
from argparse import Namespace

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
                model_config = config[task][language][corpus]['data']['model_config']
                with open(absolute_path(path,model_config),"r") as f:
                    self.model_config = Namespace(**json.load(f))
                    self.model_config.label_num = len(self.vocabulary)
                self.model = W2NER(self.model_config)
                # self.model = Bert4Ner(len(self.vocabulary))
            elif self.corpus == 'ace2005':
                self.model = BertSoftmax(self.vocabulary)
            elif self.corpus == 'conll2003':
                model_config = config[task][language][corpus]['data']['model_config']
                with open(absolute_path(path,model_config),"r") as f:
                    self.model_config = Namespace(**json.load(f))
                    self.model_config.label_num = len(self.vocabulary)
                self.model = W2NER(self.model_config)
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
                # provide an empty label which will not be used afterwards
                labels = [{"index":[1] * len(words),"type":"null"}]
                from cogie.io.processor.ner.trex_ner import process_w2ner
                bert_inputs, attention_masks, \
                grid_labels, grid_mask2d, \
                pieces2word, dist_inputs, \
                sent_length, _ = \
                    process_w2ner(list(words), labels, self.tokenizer, self.vocabulary, self.max_seq_length)

                bert_inputs = torch.tensor([bert_inputs], dtype=torch.long, device=self.device)
                attention_masks = torch.tensor([attention_masks], dtype=torch.long, device=self.device)
                grid_mask2d = torch.tensor([grid_mask2d], dtype=torch.long, device=self.device)
                pieces2word = torch.tensor([pieces2word], dtype=torch.long, device=self.device)
                dist_inputs = torch.tensor([dist_inputs], dtype=torch.long, device=self.device)
                sent_length = torch.tensor([sent_length], dtype=torch.long, device=self.device)

                outputs = self.model(bert_inputs=bert_inputs,
                                    attention_masks=attention_masks,
                                    grid_mask2d=grid_mask2d,
                                    dist_inputs=dist_inputs,
                                    pieces2word=pieces2word,
                                    sent_length=sent_length)
                outputs = torch.argmax(outputs,-1)
                decode_entities = w2ner_decode(outputs.cpu().numpy(),sent_length.cpu().numpy())
                single_decode_entities = sorted(decode_entities[0],key=lambda x:x[0][0])
                ner_result = []
                for idx,entity_tuple in enumerate(single_decode_entities):
                    mention = " ".join([words[i] for i in entity_tuple[0]])
                    start = entity_tuple[0][0]
                    end = entity_tuple[0][-1]+1
                    label = "unkonwn"
                    context_left = words[:start]
                    context_right = words[end:]
                    ner_result.append({"mention":mention,
                                       "start":start,
                                       "end":end,
                                       "type":label,
                                       "context_left":context_left,
                                       "context_right":context_right})

                return ner_result

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

            elif self.corpus == 'conll2003':
                self.model.eval()
                labels = ["O"] * len(words)
                # import cogie.io.processor.ner.conll2003 as processor
                # from cogie.io.processor.ner.trex_ner import TrexW2NERProcessor
                from cogie.io.processor.ner.conll2003 import process_w2ner
                bert_inputs, attention_masks, \
                grid_labels, grid_mask2d, \
                pieces2word, dist_inputs, \
                sent_length, _ = \
                    process_w2ner(list(words), labels, self.tokenizer, self.vocabulary, self.max_seq_length)

                bert_inputs = torch.tensor([bert_inputs], dtype=torch.long, device=self.device)
                attention_masks = torch.tensor([attention_masks], dtype=torch.long, device=self.device)
                # grid_labels = torch.tensor([grid_labels], dtype=torch.long, device=self.device)
                grid_mask2d = torch.tensor([grid_mask2d], dtype=torch.long, device=self.device)
                pieces2word = torch.tensor([pieces2word], dtype=torch.long, device=self.device)
                dist_inputs = torch.tensor([dist_inputs], dtype=torch.long, device=self.device)
                sent_length = torch.tensor([sent_length], dtype=torch.long, device=self.device)

                outputs = self.model(bert_inputs=bert_inputs,
                                    attention_masks=attention_masks,
                                    grid_mask2d=grid_mask2d,
                                    dist_inputs=dist_inputs,
                                    pieces2word=pieces2word,
                                    sent_length=sent_length)
                outputs = torch.argmax(outputs,-1)
                decode_entities = w2ner_decode(outputs.cpu().numpy(),sent_length.cpu().numpy())
                single_decode_entities = sorted(decode_entities[0],key=lambda x:x[0][0])
                tags = [self.vocabulary.idx2word[elem[1]] for elem in single_decode_entities]
                spans = _bio_tag_to_spans(words,tags)
                ner_result = []
                for idx,entity_dict in enumerate(spans):
                    context_left = words[:entity_dict["start"]]
                    context_right = words[entity_dict["end"]:]
                    ner_result.append({**entity_dict,
                                       "context_left":context_left,
                                       "context_right":context_right})
                return ner_result

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

def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)

def w2ner_decode(outputs, length):
    decode_entities = []
    for index, (instance, l) in enumerate(zip(outputs, length)):
        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        for head in head_dict:
            find_entity(head, [], head_dict[head])

        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])

    return decode_entities