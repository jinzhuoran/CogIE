"""
@Author: jinzhuan
@File: framenet.py
@Desc:
"""
from cogie.core import DataTable
from cogie.utils import Vocabulary
from transformers import BertTokenizer
from tqdm import tqdm
import os
import xml.etree.ElementTree as ElementTree


class FrameNet4JointProcessor:
    def __init__(self,path,max_span_width = 15, bert_model='bert-base-cased', max_length=128):
        self.max_span_width = max_span_width
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_length = max_length
        self.path=path
        self._ontology = FrameOntology(self.path)

    def process(self, dataset):
        datable = DataTable()
        for words,lemmas,node_types,node_attrs,origin_lexical_units,p2p_edges,p2r_edges,origin_frames,frame_elements in \
                tqdm(zip(dataset["words"],dataset["lemma"],dataset["node_types"],
                dataset["node_attrs"],dataset["origin_lexical_units"],dataset["p2p_edges"],
                dataset["p2r_edges"],dataset["origin_frames"],dataset["frame_elements"]),total=len(dataset['words'])):
            tokens_x,masks,head_indexes,spans,\
            node_type_labels_list,node_attr_labels_list,\
            node_valid_attrs_list,valid_p2r_edges_list = self.process_item(words,lemmas,node_types,node_attrs,origin_lexical_units,p2p_edges,p2r_edges,origin_frames,frame_elements )
            datable("tokens_x", tokens_x)
            datable("masks",masks)
            datable("head_indexes",head_indexes)
            datable("spans",spans )
            datable("node_type_labels_list",node_type_labels_list )
            datable("node_attr_labels_list",node_attr_labels_list )
            datable("node_valid_attrs_list",node_valid_attrs_list)
            datable("valid_p2r_edges_list", valid_p2r_edges_list)
        return datable


    def process_item(self,raw_words,lemmas,node_types,node_attrs,origin_lexical_units,p2p_edges,p2r_edges,origin_frames,frame_elements ):
        #process token
        tokens_x, is_heads,head_indexes = [],[],[]
        words = ['[CLS]'] + raw_words + ['[SEP]']
        for w in words:
            tokens = self.tokenizer.tokenize(w) if w not in ['[CLS]', '[SEP]'] else [w]
            tokens_xx = self.tokenizer.convert_tokens_to_ids(tokens)
            if w in ['[CLS]', '[SEP]']:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)
            tokens_x.extend(tokens_xx)
            is_heads.extend(is_head)
        masks = [True]*len(tokens_x) + [False] * (self.max_length - len(tokens_x))
        tokens_x = tokens_x + [0] * (self.max_length - len(tokens_x))
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)
        head_indexes = head_indexes + [0] * (self.max_length - len(head_indexes))

        #process other data
        node_types_dict, node_attrs_dict, origin_lus_dict, \
        p2p_edges_dict, p2r_edges_dict, origin_frames_dict, frame_elements_dict = \
            format_label_fields(node_types, node_attrs, origin_lexical_units,p2p_edges, p2r_edges,
                                                                          origin_frames, frame_elements)

        #process span
        spans = []
        item_node_type_labels_list= []
        item_node_attr_labels_list= []
        node_valid_attrs_list= []  # use for the comprehensive vocabulary
        valid_p2r_edges_list= []
        type_labels_list=[]
        attr_labels_list=[]
        spans=self.get_spans(raw_words,max_span_width=self.max_span_width)
        for start, end in spans:
            span_ix = (start, end)
            node_type_label = node_types_dict[span_ix]
            node_attr_label = node_attrs_dict[span_ix]

            item_node_type_labels_list.append(node_type_label)
            item_node_attr_labels_list.append(node_attr_label)

            lexical_unit = origin_lus_dict[span_ix]
            if lexical_unit in self._ontology.lu_frame_map:
                valid_attrs = self._ontology.lu_frame_map[lexical_unit]
            else:
                valid_attrs = ["O"]
            node_valid_attrs_list.append( [x for x in valid_attrs])

            if node_attr_label in self._ontology.frame_fe_map:
                valid_p2r_edge_labels = self._ontology.frame_fe_map[node_attr_label]
                valid_p2r_edges_list.append([x for x in valid_p2r_edge_labels])
            else:
                valid_p2r_edges_list.append([-1])
        type_labels_list=[item_node_type_labels_list, raw_words]
        attr_labels_list=[item_node_attr_labels_list, raw_words]
        print("end")


        return tokens_x,masks,head_indexes,spans,type_labels_list,attr_labels_list,node_valid_attrs_list,valid_p2r_edges_list

    def get_spans(self,tokens,min_span_width=1 ,max_span_width=None, filter_function= None):
        max_span_width = max_span_width or len(tokens)
        filter_function = filter_function or (lambda x: True)
        spans= []
        for start_index in range(len(tokens)):
            last_end_index = min(start_index + max_span_width, len(tokens))
            first_end_index = min(start_index + min_span_width - 1, len(tokens))
            for end_index in range(first_end_index, last_end_index):
                start = start_index
                end = end_index
                if filter_function(tokens[slice(start_index, end_index + 1)]):
                    spans.append((start, end))
        return spans
class MissingDict(dict):

    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val
def format_label_fields(node_types, node_attrs, origin_lexical_units, p2e_edges, p2r_edges, origin_frames,
                        frame_elements):

    node_types_dict = MissingDict("O",
                                  (
                                      ((span_start, span_end), target)
                                      for ((span_start, span_end), target) in node_types
                                  )
                                  )

    node_attrs_dict = MissingDict("O",
                                  (
                                      ((span_start, span_end), frame)
                                      for ((span_start, span_end), frame) in node_attrs
                                  )
                                  )

    origin_lus_dict = MissingDict("O",
                                  (
                                      ((span_ix[0][0], span_ix[-1][1]), lu)
                                      for span_ix, lu in origin_lexical_units
                                  )
                                  )

    origin_frames_dict = MissingDict("O",
                                     (
                                         (tuple([tuple(x) for x in span_ix]), frame)
                                         for span_ix, frame in origin_frames
                                     )
                                     )

    p2p_edges_dict_values = []
    for (span1_start, span1_end, span2_start, span2_end, relation) in p2e_edges:
        p2p_edges_dict_values.append((((span1_start, span1_end), (span2_start, span2_end)), relation))
    p2p_edges_dict = MissingDict("", p2p_edges_dict_values)

    p2r_edges_dict_values = []
    for (span1_start, span1_end, span2_start, span2_end, relation) in p2r_edges:
        p2r_edges_dict_values.append((((span1_start, span1_end), (span2_start, span2_end)), relation))
    p2r_edges_dict = MissingDict("", p2r_edges_dict_values)

    frame_elements_dict_values = []
    for ((predicate_ixs, frame_label), role_start, role_end, relation) in frame_elements:
        target_ixs_tuple = tuple([tuple(x) for x in predicate_ixs])
        frame_elements_dict_values.append((((target_ixs_tuple, frame_label), (role_start, role_end)), relation))
    frame_elements_dict = MissingDict("", frame_elements_dict_values)

    return node_types_dict, node_attrs_dict, origin_lus_dict, p2p_edges_dict, p2r_edges_dict, origin_frames_dict, frame_elements_dict



class FrameOntology:
    """
    This class is designed to read the ontology for FrameNet 1.x.
    """

    def __init__(self, data_path) -> None:
        self._namespace = {"fn": "http://framenet.icsi.berkeley.edu"}
        self._frame_index_filename = "frameIndex.xml"
        self._frame_dir = "frame"

        self.frames= set([])
        self.frame_elements= set([])
        self.lexical_units= set([])
        self.frame_fe_map = dict()
        self.core_frame_map = dict()
        self.lu_frame_map = dict()
        self.simple_lu_frame_map = dict()

        self._read(data_path)
        print("# frames in ontology: %d", len(self.frames))
        print("# lexical units in ontology: %d", len(self.lexical_units))
        print("# frame-elements in ontology: %d",
              len(self.frame_elements))

    def _simplify_lexunit(self, lexunit):

        # situation: president_(political): president
        if "_(" in lexunit:
            speicial_flag_index = lexunit.index("_(")
            simple_lu = lexunit[:speicial_flag_index]
            return simple_lu

        # situation: snow_event -> snow event
        if not lexunit.isalpha():
            speicial_flag_index = None
            for i in range(len(lexunit)):
                if lexunit[i] != " " and lexunit[i].isalpha() != True:
                    speicial_flag_index = i
                    break
            if speicial_flag_index:
                speicial_flag = lexunit[speicial_flag_index]
                split_lu_tokens = lexunit.split(speicial_flag)
                return " ".join(split_lu_tokens)

        return lexunit

    def _read_ontology_for_frame(self, frame_filename):
        with open(frame_filename, "r", encoding="utf-8") as frame_file:
            tree = ElementTree.parse(frame_file)
        root = tree.getroot()

        fe_for_frame = []
        core_fe_list = []
        for fe_tag in root.findall("fn:FE", self._namespace):
            fe = fe_tag.attrib["name"]
            fe_for_frame.append(fe)
            self.frame_elements.add(fe)
            if fe_tag.attrib["coreType"] == "Core":
                core_fe_list.append(fe)

        lu_for_frame = [lu.attrib["name"].split(".")[0] for lu in root.findall("fn:lexUnit", self._namespace)]
        for lu in lu_for_frame:
            self.lexical_units.add(lu)

        frame_file.close()
        return fe_for_frame, core_fe_list, lu_for_frame

    def _read_ontology(self, frame_index_filename: str) :
        print(frame_index_filename)
        with open(frame_index_filename, "r", encoding="utf-8") as frame_file:
            tree = ElementTree.parse(frame_file)
        root = tree.getroot()

        self.frames = set([frame.attrib["name"]
                           for frame in root.findall("fn:frame", self._namespace)])

    def _read(self, file_path: str):
        frame_index_path = os.path.join(file_path, self._frame_index_filename)
        print("Reading the frame ontology from %s", frame_index_path)
        self._read_ontology(frame_index_path)

        max_fe_for_frame = 0
        total_fe_for_frame = 0.
        max_core_fe_for_frame = 0
        total_core_fe_for_frame = 0.
        max_frames_for_lu = 0
        total_frames_per_lu = 0.
        longest_frame = None

        frame_path = os.path.join(file_path, self._frame_dir)
        print("Reading the frame-element - frame ontology from %s",
              frame_path)
        for frame in self.frames:
            frame_file = os.path.join(frame_path, "{}.xml".format(frame))

            fe_list, core_fe_list, lu_list = self._read_ontology_for_frame(
                frame_file)
            self.frame_fe_map[frame] = fe_list
            self.core_frame_map[frame] = core_fe_list

            # Compute FE stats
            total_fe_for_frame += len(self.frame_fe_map[frame])
            if len(self.frame_fe_map[frame]) > max_fe_for_frame:
                max_fe_for_frame = len(self.frame_fe_map[frame])
                longest_frame = frame

            # Compute core FE stats
            total_core_fe_for_frame += len(self.core_frame_map[frame])
            if len(self.core_frame_map[frame]) > max_core_fe_for_frame:
                max_core_fe_for_frame = len(self.core_frame_map[frame])

            for lex_unit in lu_list:
                if lex_unit not in self.lu_frame_map:
                    self.lu_frame_map[lex_unit] = []
                self.lu_frame_map[lex_unit].append(frame)

                simple_lex_unit = self._simplify_lexunit(lex_unit)
                if simple_lex_unit not in self.simple_lu_frame_map:
                    self.simple_lu_frame_map[simple_lex_unit] = []
                self.simple_lu_frame_map[simple_lex_unit].append(frame)

                # Compute frame stats
                if len(self.lu_frame_map[lex_unit]) > max_frames_for_lu:
                    max_frames_for_lu = len(self.lu_frame_map[lex_unit])
                total_frames_per_lu += len(self.lu_frame_map[lex_unit])

        print("# max FEs per frame = %d (in frame %s)",
              max_fe_for_frame, longest_frame)
        print("# avg FEs per frame = %f",
              total_fe_for_frame / len(self.frames))
        print("# max core FEs per frame = %d", max_core_fe_for_frame)
        print("# avg core FEs per frame = %f",
              total_core_fe_for_frame / len(self.frames))
        print("# max frames per LU = %d", max_frames_for_lu)
        print("# avg frames per LU = %f",
              total_frames_per_lu / len(self.lu_frame_map))




