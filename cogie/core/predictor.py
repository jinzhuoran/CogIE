"""
@Author: jinzhuan
@File: predictor.py
@Desc: 
"""
from cogie.utils import module2parallel, load_model
import nltk
import logging
import pickle
import cogie.io.processor.ner.conll2003 as ner_processor
import cogie.io.processor.et as et_processor
import cogie.io.processor.fn as fn_processor
import torch
from transformers import BertTokenizer
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model, model_path=None, vocabulary=None, device=None, device_ids=None, max_seq_length=256):
        self.model = model
        self.model_path = model_path
        self.vocabulary = vocabulary
        self.device = device
        self.device_ids = device_ids
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        if self.device_ids:
            self.model = module2parallel(self.model, self.device_ids)
        if self.model_path:
            self.model = load_model(self.model, self.model_path)

    def predict(self, sentence):
        pass


class NerPredictor(Predictor):
    def __init__(self, model, model_path=None, vocabulary=None, device=None, device_ids=None, max_seq_length=256):
        super().__init__(model, model_path, vocabulary, device, device_ids, max_seq_length)

    def predict(self, sentence):
        self.model.eval()
        words = nltk.word_tokenize(sentence)
        labels = ['O'] * len(words)
        input_ids, attention_masks, segment_ids, valid_masks, label_ids, label_masks = \
            ner_processor.process(words, labels, self.tokenizer, self.vocabulary, self.max_seq_length)

        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_masks = torch.tensor([attention_masks], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
        valid_masks = torch.tensor([valid_masks], dtype=torch.long, device=self.device)
        label_ids = torch.tensor([label_ids], dtype=torch.long, device=self.device)
        label_masks = torch.tensor([label_masks], dtype=torch.long, device=self.device)

        with torch.no_grad():
            prediction, valid_len = self.model.predict(
                [input_ids, attention_masks, segment_ids, valid_masks, label_ids, label_masks])
        prediction = prediction[0]
        valid_len = valid_len[0]
        tag = []
        for i in range(valid_len.item()):
            if i != 0 and i != valid_len.item() - 1:
                tag.append(self.vocabulary.to_word(prediction[i].item()))
        spans = _bio_tag_to_spans(tag)
        return tag, spans


def _bio_tag_to_spans(tags, ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[1][0], span[1][1] + 1) for span in spans if span[0] not in ignore_labels]


class EtPredictor(Predictor):
    def __init__(self, model, model_path, vocabulary=None, device=None, device_ids=None, max_seq_length=256):
        super().__init__(model, model_path, vocabulary, device, device_ids, max_seq_length)

    def predict(self, sentence, start, end):
        words = nltk.word_tokenize(sentence)
        labels = ['<unk>'] * len(words)
        input_ids, attention_mask, start_pos, end_pos, label_ids = \
            et_processor.process(words, start, end, labels, self.vocabulary, self.tokenizer, self.max_seq_length)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        start_pos = torch.tensor([start_pos], dtype=torch.long, device=self.device)
        end_pos = torch.tensor([end_pos], dtype=torch.long, device=self.device)
        label_ids = torch.tensor([label_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            output = self.model.predict([input_ids, attention_mask, start_pos, end_pos, label_ids])
        output = output[0]
        prediction = []
        for i in range(len(output)):
            if output[i] == 1:
                prediction.append(self.vocabulary.to_word(i))
        return prediction


class FnPredictor(Predictor):
    def __init__(self, model, model_path, vocabulary=None, device=None, device_ids=None, max_seq_length=256,
                 lu_path=None):
        super().__init__(model, model_path, vocabulary, device, device_ids, max_seq_length)
        self.blank_padding = True
        self.mask_entity = True
        with open(lu_path, 'rb') as file:
            self.lu_dic = pickle.load(file)

    def predict(self, sentence):
        words = nltk.word_tokenize(sentence)
        prediction = []
        for i in range(len(words)):
            if words[i] in self.lu_dic:
                label = '<unk>'
                position = i + 1
                input_ids, attention_mask, word_pos, label_ids = \
                    fn_processor.process(list(words), label, position, self.tokenizer, self.vocabulary,
                                         self.max_seq_length)
                input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
                word_pos = torch.tensor([word_pos], dtype=torch.long, device=self.device)
                label_ids = torch.tensor([label_ids], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    output = self.model.predict([input_ids, attention_mask, word_pos, label_ids])
                prediction.append({'word': words[i], 'position': i, 'frame': self.vocabulary.to_word(output.item())})
        return {'data': prediction}


class WsPredictor(Predictor):
    def __init__(self, model, model_path, vocabulary=None, device=None, device_ids=None, max_seq_length=256,
                 lu_path=None):
        super().__init__(model, model_path, vocabulary, device, device_ids, max_seq_length)
        self.blank_padding = True
        self.mask_entity = True
        with open(lu_path, 'rb') as file:
            self.lu_dic = pickle.load(file)

    def predict(self, sentence):
        words = nltk.word_tokenize(sentence)
        prediction = []
        for i in range(len(words)):
            if words[i] in self.lu_dic:
                label = '<unk>'
                position = i + 1
                input_ids, attention_mask, word_pos, label_ids = \
                    fn_processor.process(list(words), label, position, self.tokenizer, self.vocabulary,
                                         self.max_seq_length)
                input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
                word_pos = torch.tensor([word_pos], dtype=torch.long, device=self.device)
                label_ids = torch.tensor([label_ids], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    output = self.model.predict([input_ids, attention_mask, word_pos, label_ids])
                prediction.append({'word': words[i], 'position': i, 'frame': self.vocabulary.to_word(output.item())})
        return {'data': prediction}
