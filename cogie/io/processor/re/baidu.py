"""
@Author: jinzhuan
@File: baidu.py
@Desc: 
"""
from ..processor import Processor
from cogie.core import DataTable


class BaiduRelationProcessor(Processor):

    def __init__(self, label_list=None, path=None, padding=None, unknown='<unk>',
                 bert_model='hfl/chinese-roberta-wwm-ext', max_length=256, blank_padding=True, mask_entity=False):
        super().__init__(label_list, path, padding, unknown, bert_model, max_length)
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity

    def process(self, dataset):
        datable = DataTable()
        for i in range(len(dataset)):
            token, relation, subj_start, subj_end, obj_start, obj_end = dataset[i]
            label_id = self.vocabulary.to_index(relation)
            item = {'token': token, 'h': {'pos': [subj_start, subj_end + 1]}, 't': {'pos': [obj_start, obj_end + 1]}}
            indexed_tokens, att_mask, pos1, pos2 = self.tokenize(item)
            datable('input_ids', indexed_tokens)
            datable('attention_mask', att_mask)
            datable('pos1', pos1)
            datable('pos2', pos2)
            datable('label_id', label_id)

            datable('input_ids', indexed_tokens)
            datable('attention_mask', att_mask)
            datable('pos1', pos2)
            datable('pos2', pos1)
            datable('label_id', self.vocabulary.to_index('<unk>'))
        return datable

    def tokenize(self, item):
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False

        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        # pos1 = torch.tensor([[pos1]]).long()
        # pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        # indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        # att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        # att_mask[0, :avai_len] = 1
        att_mask = [0] * len(indexed_tokens)
        for i in range(min(avai_len, self.max_length)):
            att_mask[i] = 1

        return indexed_tokens, att_mask, [pos1], [pos2]
