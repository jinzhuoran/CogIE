"""
@Author: jinzhuan
@File: trex.py
@Desc: 
"""
from ..processor import Processor
from cogie.core import DataTable
from tqdm import tqdm


class TrexRelationProcessor(Processor):

    def __init__(self, label_list=None, path=None, padding=None, unknown='<unk>', bert_model='bert-base-cased',
                 max_length=256):
        super().__init__(label_list, path, padding, unknown, bert_model, max_length)

    def process(self, dataset):
        datable = DataTable()

        for item in tqdm(dataset, desc='Processing Data'):
            words = item['words']
            item_entity_mentions = item['entity_mentions']
            item_relation_mentions = item['relation_mentions']
            input_ids, attention_mask, head_indexes, entity_mentions, relation_mentions, entity_mentions_mask, relation_mentions_mask = process(
                words, item_entity_mentions, item_relation_mentions, self.tokenizer, self.vocabulary, self.max_length)
            if len(input_ids) <= self.max_length and len(head_indexes) <= self.max_length\
                    and len(entity_mentions) <= self.max_length and len(relation_mentions) <= self.max_length:
                datable('input_ids', input_ids)
                datable('attention_mask', attention_mask)
                datable('head_indexes', head_indexes)
                datable('entity_mentions', entity_mentions)
                datable('relation_mentions', relation_mentions)
                datable('entity_mentions_mask', entity_mentions_mask)
                datable('relation_mentions_mask', relation_mentions_mask)

        return datable


def process(words, raw_entity_mentions, raw_relation_mentions, tokenizer, vocabulary, max_seq_length):
    words = ['[CLS]'] + words + ['[SEP]']
    input_ids, is_heads = [], []
    entity_mentions = []
    relation_mentions = []
    for word in words:
        token = tokenizer.tokenize(word) if word not in ['[CLS]', '[SEP]'] else [word]
        input_id = tokenizer.convert_tokens_to_ids(token)

        if word in ['[CLS]', '[SEP]']:
            is_head = [0]
        else:
            is_head = [1] + [0] * (len(token) - 1)

        input_ids.extend(input_id)
        is_heads.extend(is_head)

    head_indexes = []
    for i in range(len(is_heads)):
        if is_heads[i]:
            head_indexes.append(i)

    for raw_entity_mention in raw_entity_mentions:
        position = raw_entity_mention['position']
        entity_mentions.append(position)
    entity_mentions.sort()

    for raw_relation_mention in raw_relation_mentions:
        relation_type = raw_relation_mention['relation_type']
        arguments = raw_relation_mention['arguments']
        if len(arguments) != 2:
            continue
        mention = []
        for argument in arguments:
            mention.append(argument[0])
            mention.append(argument[1])
        mention.append(vocabulary.to_index(relation_type))
        relation_mentions.append(mention)
    relation_mentions.sort()

    attention_mask = [1] * len(input_ids)
    entity_mentions_mask = [1] * len(entity_mentions)
    relation_mentions_mask = [1] * len(relation_mentions)

    input_ids += [0 for _ in range(max_seq_length - len(input_ids))]
    attention_mask += [0 for _ in range(max_seq_length - len(attention_mask))]
    head_indexes += [0 for _ in range(max_seq_length - len(head_indexes))]
    entity_mentions += [[0, 0] for _ in range(max_seq_length - len(entity_mentions))]
    relation_mentions += [[-1, -1, -1, -1, -1] for _ in range(max_seq_length - len(relation_mentions))]
    entity_mentions_mask += [0 for _ in range(max_seq_length - len(entity_mentions_mask))]
    relation_mentions_mask += [0 for _ in range(max_seq_length - len(relation_mentions_mask))]

    return input_ids, attention_mask, head_indexes, entity_mentions, relation_mentions, entity_mentions_mask, relation_mentions_mask
