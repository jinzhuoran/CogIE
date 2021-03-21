"""
@Author: jinzhuan
@File: frame_argument.py
@Desc: 
"""
from cogie.core import DataTable
from cogie.utils import Vocabulary
from ..processor import Processor


class FrameArgumentProcessor(Processor):
    def __init__(self, label_list=None, path=None, padding=None, unknown=None, bert_model='bert-base-cased',
                 max_length=256, trigger_label_list=None, argument_label_list=None):
        super().__init__(label_list, path, padding=padding, unknown=unknown, bert_model=bert_model,
                         max_length=max_length)
        self.trigger_vocabulary = Vocabulary(padding=padding)
        self.trigger_vocabulary.add_word_lst(trigger_label_list)
        self.trigger_vocabulary.build_vocab()

        self.argument_vocabulary = Vocabulary(padding=padding, unknown=unknown)
        self.argument_vocabulary.add_word_lst(argument_label_list)
        self.argument_vocabulary.build_vocab()

    def process(self, dataset):
        datable = DataTable()
        for i in range(len(dataset)):
            sentence, label, frame, pos = dataset[i]
            input_id, attention_mask, segment_id, head_index, label_id, label_mask = process(sentence, label,
                                                                                             frame, pos,
                                                                                             self.tokenizer,
                                                                                             self.trigger_vocabulary,
                                                                                             self.argument_vocabulary,
                                                                                             self.max_length)
            datable('input_ids', input_id)
            datable('attention_mask', attention_mask)
            datable('segment_ids', segment_id)
            datable('head_indexes', head_index)
            datable('label_ids', label_id)
            datable('label_masks', label_mask)
            datable('frame', self.trigger_vocabulary.to_index(frame))
            datable('pos', pos)
        return datable


def process(words, labels, frame, pos, tokenizer, trigger_vocabulary, argument_vocabulary, max_seq_length):
    input_id = []
    label_id = []
    words = ['[CLS]'] + words[0:pos] + ['[unused' + str(trigger_vocabulary.to_index(frame)) + ']'] + [words[pos]] + [
        '[unused' + str(trigger_vocabulary.to_index(frame)) + ']'] + words[pos + 1:] + ['[SEP]']

    is_heads = []
    head_index = []

    for word in words:
        token = tokenizer.tokenize(word)
        input_id.extend(token)
        if word in ['[CLS]', '[SEP]'] or 'unused' in word:
            is_head = [0]
        else:
            is_head = [1] + [0] * (len(token) - 1)
        is_heads.extend(is_head)

    for i in range(len(is_heads)):
        if is_heads[i]:
            head_index.append(i)

    input_id = tokenizer.convert_tokens_to_ids(input_id)
    attention_mask = [1] * len(input_id)
    segment_id = [0] * len(input_id)

    for label in labels:
        label_id.append(argument_vocabulary.to_index(label))
    label_mask = [1] * len(label_id)

    input_id = input_id[0:max_seq_length]
    attention_mask = attention_mask[0:max_seq_length]
    segment_id = segment_id[0:max_seq_length]
    head_index = head_index[0:max_seq_length]
    label_id = label_id[0:max_seq_length]
    label_mask = label_mask[0:max_seq_length]

    input_id += [0 for _ in range(max_seq_length - len(input_id))]
    attention_mask += [0 for _ in range(max_seq_length - len(attention_mask))]
    segment_id += [0 for _ in range(max_seq_length - len(segment_id))]
    head_index += [0 for _ in range(max_seq_length - len(head_index))]
    label_id += [-1 for _ in range(max_seq_length - len(label_id))]
    label_mask += [0 for _ in range(max_seq_length - len(label_mask))]

    return input_id, attention_mask, segment_id, head_index, label_id, label_mask
