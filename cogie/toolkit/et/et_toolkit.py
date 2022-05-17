"""
@Author: jinzhuan
@File: et_toolkit.py
@Desc: 
"""
import numpy as np

from cogie import *
from ..base_toolkit import BaseToolkit
import threading
import torch
from cogie.utils.box4et_constant import load_vocab_dict
import json
from argparse import Namespace

from cogie.models.et.box4et import TransformerBoxModel
import cogie.utils.box4et_constant as constant



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
        model = config[task][language][corpus]['data']['model']
        vocabulary = config[task][language][corpus]['data']['vocabulary']
        vocabulary_path = absolute_path(path, vocabulary)
        if self.corpus == 'ufet':
            self.vocabulary = Vocabulary()
            self.vocabulary._word2idx = load_vocab_dict(vocabulary_path)
            self.vocabulary._idx2word = {v: k for k, v in self.vocabulary._word2idx.items()}
            vocabulary_path = None
        bert_model = config[task][language][corpus]['bert_model']
        device = torch.device(config['device'])
        device_ids = config['device_id']
        max_seq_length = config['max_seq_length']
        super().__init__(bert_model, absolute_path(path, model),vocabulary_path, device, device_ids,
                         max_seq_length)
        if self.language == 'english':
            if self.corpus == 'ontonotes':
                self.model = Bert4Et(len(self.vocabulary))
            elif self.corpus == 'ufet':
                model_config = config[task][language][corpus]['data']['model_config']
                with open(absolute_path(path,model_config),"r") as f:
                    self.model_config = Namespace(**json.load(f))
                self.model = TransformerBoxModel(self.model_config, constant.ANSWER_NUM_DICT[self.model_config.goal])
        self.load_model()

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(EtToolkit, "_instance"):
            with EtToolkit._instance_lock:
                if not hasattr(EtToolkit, "_instance"):
                    EtToolkit._instance = object.__new__(cls)
        return EtToolkit._instance

    def run(self, words, spans=None):
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

            elif self.corpus == 'ufet':
                from cogie.io.processor.et.ufet import process_ufet
                # input: span=None ner_result=words
                ner_result = words
                for idx,ner in enumerate(ner_result):
                    word_num = len(ner["context_left"]) + len(ner["context_right"]) + 1
                    self.model.eval()
                    labels = ["person"] * word_num # useless labels
                    ex_id = [0]
                    left_context = ner["context_left"]
                    right_context = ner["context_right"]
                    mention = ner["mention"]
                    sample = [ex_id,left_context,right_context,mention,labels]
                    input_ids,token_type_ids,attention_mask,target = process_ufet(
                        sample,self.tokenizer,self.vocabulary,self.max_seq_length
                    )
                    input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                    attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
                    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long, device=self.device)
                    _,output_logits = self.model.forward(
                        inputs={"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask},
                    )
                    output_logits = torch.where(output_logits >= 0.5, 1, 0)
                    output = output_logits.clone().cpu().numpy()
                    output_index = np.nonzero(output)[1]
                    if output_index.size != 0:
                        types = [self.vocabulary.idx2word[id] for id in output_index]
                    else:
                        types = []
                    ner_result[idx].update({"types":types})
                return ner_result








