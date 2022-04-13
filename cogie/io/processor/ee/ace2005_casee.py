import os
from cogie.utils import Vocabulary
from cogie.core import DataTable
from transformers import BertTokenizer
from tqdm import tqdm
import json
import numpy as np

class ACE2005CASEEProcessor:
    def __init__(self,
                 schema_path=None,
                 trigger_path=None,
                 argument_path=None,
                 bert_model='bert-base-cased',
                 max_length=128):
        self.schema_path=schema_path
        self.trigger_path=trigger_path
        self.argument_path=argument_path
        self.bert_model = bert_model
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.trigger_type_list = list()
        self.argument_type_list = list()
        self.args_s_id = {}
        self.args_e_id = {}
        self.schema_id = {}
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema_str = json.load(f)
        trigger_type_set = set()
        argument_type_set = set()
        for trigger_type, argument_type_list in self.schema_str.items():
            trigger_type_set.add(trigger_type)
            for argument_type in argument_type_list:
                argument_type_set.add(argument_type)
        self.trigger_type_list = list(trigger_type_set)
        self.argument_type_list = list(argument_type_set)
        for i in range(len(self.argument_type_list)):
            s = self.argument_type_list[i] + '_s'
            self.args_s_id[s] = i
            e = self.argument_type_list[i] + '_e'
            self.args_e_id[e] = i
        if os.path.exists(self.trigger_path):
            self.trigger_vocabulary = Vocabulary.load(self.trigger_path)
        else:
            self.trigger_vocabulary = Vocabulary(padding=None, unknown="O")
            self.trigger_vocabulary.add_word_lst(self.trigger_type_list)
            self.trigger_vocabulary.build_vocab()
            self.trigger_vocabulary.save(self.trigger_path)
        if os.path.exists(self.argument_path):
            self.argument_vocabulary = Vocabulary.load(self.argument_path)
        else:
            self.argument_vocabulary = Vocabulary(padding=None, unknown="O")
            self.argument_vocabulary.add_word_lst(self.argument_type_list)
            self.argument_vocabulary.build_vocab()
            self.argument_vocabulary.save(self.argument_path)
        for trigger_type, argument_type_list in self.schema_str.items():
            self.schema_id[self.trigger_vocabulary.word2idx[trigger_type]] = [self.argument_vocabulary.word2idx[a] for a
                                                                              in argument_type_list]
        self.trigger_type_len = len(self.trigger_vocabulary)
        self.argument_type_len =len(self.argument_vocabulary)

        print("end")

    def process_train(self, dataset):
        datable = DataTable()
        for content, index, type,args, occur, triggers in \
            tqdm(zip(dataset["content"], dataset["index"], dataset["type"],
                     dataset["args"], dataset["occur"], dataset["triggers"]),total=len(dataset["content"])):
            tokens_x, is_heads, head_indexes = [], [], []
            words = ['[CLS]'] + content + ['[SEP]']
            for w in words:
                tokens = self.tokenizer.tokenize(w) if w not in ['[CLS]', '[SEP]'] else [w]
                tokens_xx = self.tokenizer.convert_tokens_to_ids(tokens)
                if w in ['[CLS]', '[SEP]']:
                    is_head = [0]
                else:
                    is_head = [1] + [0] * (len(tokens) - 1)
                tokens_x.extend(tokens_xx)
                is_heads.extend(is_head)
            token_masks = [True] * len(tokens_x) + [False] * (self.max_length - len(tokens_x))
            token_masks=token_masks[: self.max_length]
            tokens_x = tokens_x[:self.max_length] + [0] * (self.max_length - len(tokens_x))
            for i in range(len(is_heads)):
                if is_heads[i]:
                    head_indexes.append(i)
            head_indexes = head_indexes + [0] * (self.max_length - len(head_indexes))

            data_type_id = self.trigger_vocabulary.word2idx[type]
            type_vec = np.array([0] * self.trigger_type_len)
            for occ in occur:
                idx = self.trigger_vocabulary.word2idx[occ]
                type_vec[idx] = 1

            t_m = [0] * self.max_length
            r_pos = list(range(-0, 0)) + [0] * (0 - 0 + 1) + list(
                range(1, self.max_length -0))
            r_pos = [p + self.max_length for p in r_pos]
            if index!=-1:
                span = triggers[index]
                start_idx=span[0] + 1
                end_idx=span[1] + 1 - 1
                r_pos = list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, self.max_length - end_idx))
                r_pos = [p + self.max_length for p in r_pos]
                t_m= [0] * self.max_length
                t_m[start_idx] = 1
                t_m[end_idx] = 1


            t_index=index


            t_s = [0] * self.max_length
            t_e = [0] * self.max_length

            for t in triggers:
                t_s[t[0] + 1] = 1
                t_e[t[1] + 1 - 1] = 1

            args_s = np.zeros(shape=[self.argument_type_len, self.max_length])
            args_e = np.zeros(shape=[self.argument_type_len, self.max_length])
            arg_mask = [0] * self.argument_type_len
            for args_name in args:
                s_r_i = self.args_s_id[args_name + '_s']
                e_r_i = self.args_e_id[args_name + '_e']
                arg_mask[s_r_i] = 1
                for span in args[args_name]:
                    args_s[s_r_i][span[0] + 1] = 1
                    args_e[e_r_i][span[1] + 1 - 1] = 1
            if len(tokens_x)>128:
                print( len(tokens_x))
            datable("tokens_x", tokens_x)
            datable("token_masks", token_masks)
            datable("head_indexes", head_indexes)
            datable("data_type_id",data_type_id)
            datable("type_vec", type_vec)
            datable("r_pos",r_pos)
            datable("t_m", t_m)
            datable("t_index",t_index)
            datable("t_s",t_s)
            datable("t_e", t_e)
            datable("a_s", args_s)
            datable("a_e", args_e)
            datable("a_m", arg_mask)

        return datable


    def process_dev(self, dataset):
        datable = DataTable()
        for content, index, type, args, occur, triggers in \
                tqdm(zip(dataset["content"], dataset["index"], dataset["type"],
                         dataset["args"], dataset["occur"], dataset["triggers"]), total=len(dataset['words'])):
            pass
        return datable

    def get_trigger_vocabulary(self):
        return self.trigger_vocabulary

    def get_argument_vocabulary(self):
        return self.argument_vocabulary

