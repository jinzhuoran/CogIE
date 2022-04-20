
from ..processor import Processor
from transformers import BertTokenizer
from cogie.core import DataTable
from tqdm import tqdm
import nltk
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from cogie.utils.box4et_constant import ANSWER_NUM_DICT
from cogie.utils import Vocabulary
import os
from cogie.utils.box4et_constant import load_vocab_dict

class UfetProcessor(Processor):
    def __init__(self, label_list=None, path=None, padding=None, unknown=None, bert_model='bert-base-cased',
                 max_length=256):
        super().__init__(label_list, path, bert_model=bert_model,unknown=unknown,
                         max_length=max_length)

    def process(self, dataset):
        datable = DataTable()
        # add your own process code here
        for sample in zip(dataset['ex_id'],
                                 dataset['left_context'],
                                 dataset['right_context'],
                                 dataset['mention'],
                                 dataset['label'],):
            input_ids,token_type_ids,attention_mask,target = process_ufet(sample,self.tokenizer,self.vocabulary,self.max_length)
            datable('input_ids',input_ids)
            datable('token_type_ids', token_type_ids)
            datable('attention_mask', attention_mask)
            datable('target', target)
        return datable

    def load_vocabulary(self, path):
        file = os.path.join(path, 'ufet_types.txt')
        self.vocabulary = Vocabulary()
        self.vocabulary._word2idx = load_vocab_dict(file)
        self.vocabulary._idx2word =  {v: k for k, v in self.vocabulary._word2idx.items()}

def process_ufet(sample,tokenizer,vocab,max_seq_length):
    ex_id,left_seq,right_seq,mention_seq,label = sample
    mention = ' '.join(mention_seq)
    context = ' '.join(left_seq + mention_seq + right_seq)
    target = np.zeros(ANSWER_NUM_DICT["ufet"],np.float32)
    inputs = tokenizer.encode_plus(
        mention,context,
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation_strategy="only_second",
        pad_to_max_length=True,
        return_tensors="pt",
        truncation=True
    )
    input_ids = torch.squeeze(inputs["input_ids"]).numpy()
    token_type_ids = torch.squeeze(inputs["token_type_ids"]).numpy()
    attention_mask = torch.squeeze(inputs["attention_mask"]).numpy()
    for label_str in label:
        target[vocab.word2idx[label_str]] = 1.0
    return input_ids,token_type_ids,attention_mask,target