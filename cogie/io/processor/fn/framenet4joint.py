"""
@Author: jinzhuan
@File: framenet.py
@Desc:
"""
from cogie.core import DataTable
from cogie.utils import Vocabulary
from transformers import BertTokenizer
from tqdm import tqdm


class FrameNet4JointProcessor:
    def __init__(self,max_span_width = 15, bert_model='bert-base-cased', max_length=128):
        self.max_span_width = max_span_width
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_length = max_length

    def process(self, dataset):
        datable = DataTable()
        for words,lemmas,node_types,node_attrs,origin_lexical_units,p2p_edges,p2r_edges,origin_frames,frame_elements in \
                tqdm(zip(dataset["words"],dataset["lemma"],dataset["node_types"],
                dataset["node_attrs"],dataset["origin_lexical_units"],dataset["p2p_edges"],
                dataset["p2r_edges"],dataset["origin_frames"],dataset["frame_elements"]),total=len(dataset['words'])):
            tokens_x,masks,head_indexes = process(words,self.tokenizer,self.max_length)
            datable("tokens_x", tokens_x)
            datable("masks",masks)
            datable("head_indexes",head_indexes)
        return datable


def process(words,  tokenizer,  max_length):
    #process token
    tokens_x, is_heads,head_indexes = [],[],[]
    words = ['[CLS]'] + words + ['[SEP]']
    for w in words:
        tokens = tokenizer.tokenize(w) if w not in ['[CLS]', '[SEP]'] else [w]
        tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
        if w in ['[CLS]', '[SEP]']:
            is_head = [0]
        else:
            is_head = [1] + [0] * (len(tokens) - 1)
        tokens_x.extend(tokens_xx)
        is_heads.extend(is_head)
    masks = [True]*len(tokens_x) + [False] * (max_length - len(tokens_x))
    tokens_x = tokens_x + [0] * (max_length - len(tokens_x))
    for i in range(len(is_heads)):
        if is_heads[i]:
            head_indexes.append(i)
    head_indexes = head_indexes + [0] * (max_length - len(head_indexes))


    # #process span
    # spans = []
    # node_type_labels_list= []
    # node_attr_labels_list= []
    # node_valid_attrs_list= []  # use for the comprehensive vocabulary
    # valid_p2r_edges_list= []
    # for start, end in enumerate_spans(tokens, max_span_width=self._max_span_width):
    #     span_ix = (start, end)
    #     node_type_label = node_types_dict[span_ix]
    #     node_attr_label = node_attrs_dict[span_ix]
    #
    #     node_type_labels_list.append(node_type_label)
    #     node_attr_labels_list.append(node_attr_label)
    #
    #     lexical_unit = origin_lus_dict[span_ix]
    #     if lexical_unit in self._ontology.lu_frame_map:
    #         # valid_frames = self._ontology.lu_frame_map[lexical_unit] + ["O"]
    #         valid_attrs = self._ontology.lu_frame_map[lexical_unit]
    #     else:
    #         valid_attrs = ["O"]
    #     node_valid_attrs_list.append(
    #         ListField([LabelField(x, label_namespace='node_attr_labels') for x in valid_attrs]))
    #
    #     if node_attr_label in self._ontology.frame_fe_map:
    #         valid_p2r_edge_labels = self._ontology.frame_fe_map[node_attr_label]
    #         valid_p2r_edges_list.append(
    #             ListField([LabelField(x, label_namespace='p2r_edge_labels') for x in valid_p2r_edge_labels]))
    #     else:
    #         valid_p2r_edges_list.append(ListField([LabelField(-1, skip_indexing=True)]))
    #
    #     spans.append(SpanField(start, end, text_field))
    #
    # span_field = ListField(spans)
    # node_type_labels_field = SequenceLabelField(node_type_labels_list, span_field, label_namespace='node_type_labels')
    # node_attr_labels_field = SequenceLabelField(node_attr_labels_list, span_field, label_namespace='node_attr_labels')
    #
    # node_valid_attrs_field = ListField(node_valid_attrs_list)
    # valid_p2r_edges_field = ListField(valid_p2r_edges_list)
    #
    # fields["spans"] = span_field
    # fields["node_type_labels"] = node_type_labels_field
    # fields["node_attr_labels"] = node_attr_labels_field
    # fields["node_valid_attrs"] = node_valid_attrs_field
    # fields["valid_p2r_edges"] = valid_p2r_edges_field

    return tokens_x,masks,head_indexes



