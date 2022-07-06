# import logging
# import itertools
# from typing import Any, Dict, List, Optional
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from overrides import overrides
# from collections import defaultdict
#
# from allennlp.data import Vocabulary
# from allennlp.models.model import Model
# from allennlp.modules import FeedForward
# from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
# from allennlp.modules import TimeDistributed
# import logging
# import os
# import xml.etree.ElementTree as ElementTree
# from typing import Dict, List, Set, TextIO, Optional, Tuple
#
# from allennlp.common.registrable import Registrable
# from allennlp.common import Params
# from allennlp.nn import util
# from allennlp.common.util import pad_sequence_to_length
# from allennlp.nn.initializers import zero
# from collections import defaultdict
# from allennlp.nn.initializers import zero
#
# import torch
# import torch.nn.functional as F
#
# from allennlp.nn import util
# from allennlp.common.util import pad_sequence_to_length
# import itertools
# import difflib
# import logging
# from typing import Any, Dict, List, Optional
#
# import torch
# import torch.nn.functional as F
# from overrides import overrides
# from typing import Tuple, Set
#
# from allennlp.training.metrics.metric import Metric
# from overrides import overrides
# from allennlp.common.util import pad_sequence_to_length
# from allennlp.data import Vocabulary
# from allennlp.models.model import Model
# from allennlp.modules import FeedForward
# from allennlp.modules import TimeDistributed
# from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
#
#
# logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
#
#
# def get_constrained_frame_label(tokens, lemmas, frame_ix, frame_scores, lu_frame_map, frame2id, id2frame,
#                                 num_frame_labels):
#     frame_tokens = " ".join([" ".join(tokens[start:end + 1]) for start, end in frame_ix])
#     frame_lemmas = " ".join([" ".join(lemmas[start:end + 1]) for start, end in frame_ix])
#     valid_frame_masks = torch.zeros(num_frame_labels).detach().cpu()
#     valid_frame_masks[0] = True
#
#     valid_lemma = None
#     if frame_lemmas in lu_frame_map:
#         valid_lemma = frame_lemmas
#     elif frame_tokens in lu_frame_map:
#         valid_lemma = frame_tokens
#     if valid_lemma:
#         valid_frame_labels = torch.LongTensor(
#             [frame2id[frame] for frame in lu_frame_map[valid_lemma] if frame in frame2id])
#         valid_frame_masks.index_fill_(-1, valid_frame_labels, True)
#     else:
#         valid_frame_masks.fill_(True)
#
#     frame_label = id2frame[torch.argmax(sum(frame_scores) + torch.log(valid_frame_masks)).item()]
#     return frame_label
#
#
# def is_clique(entity, relations):
#     entity = list(entity)
#
#     for idx, fragment1 in enumerate(entity):
#         for idy, fragment2 in enumerate(entity):
#             if idx < idy:
#                 if (fragment1, fragment2) not in relations and (fragment2, fragment1) not in relations:
#                     return False
#
#     return True
#
#
# class MissingDict(dict):
#
#     def __init__(self, missing_val, generator=None) -> None:
#         if generator:
#             super().__init__(generator)
#         else:
#             super().__init__()
#         self._missing_val = missing_val
#
#     def __missing__(self, key):
#         return self._missing_val
#
#
# def get_tag_mask(valid_tags: torch.Tensor, num_classes: int):
#     batch_size, num_spans, _ = valid_tags.size()
#     zeros = torch.zeros(batch_size, num_spans, num_classes).to(util.get_device_of(valid_tags))
#     indices = F.relu(valid_tags.float()
#                      ).long()
#     values = (valid_tags >= 0).data.float()
#     tag_mask = zeros.scatter_(2, indices, values).bool()
#     return tag_mask
#
#
# def get_flat_tag_mask(valid_tags: torch.Tensor, num_classes: int):
#     batch_size, _ = valid_tags.size()
#     zeros = torch.zeros(batch_size, num_classes).to(util.get_device_of(valid_tags))
#     valid_tags = valid_tags.view(batch_size, -1)
#     indices = F.relu(valid_tags.float()
#                      ).long().view(batch_size, -1)
#     values = (valid_tags >= 0).data.float()
#     tag_mask = zeros.scatter_(1, indices, values).bool()
#     tag_mask[:, 0] = True
#     return tag_mask
#
#
# def merge_spans(spans: List[Tuple[int, int]]):
#     # Full empty prediction.
#     if not spans:
#         return spans
#     # Create a sorted copy.
#     sorted_spans = sorted([x for x in spans])
#     prev_start, prev_end = sorted_spans[0]
#     for span in sorted_spans[1:]:
#         if span[0] == prev_end + 1:
#             # Merge these two spans.
#             spans.remove(span)
#             spans.remove((prev_start, prev_end))
#             spans.append((prev_start, span[1]))
#             prev_end = span[1]
#         else:
#             prev_start, prev_end = span
#     return list(spans)
#
#
# class FrameOntology():
#     """
#     This class is designed to read the ontology for FrameNet 1.x.
#     """
#
#     def __init__(self, data_path) -> None:
#         self._namespace = {"fn": "http://framenet.icsi.berkeley.edu"}
#         self._frame_index_filename = "frameIndex.xml"
#         self._frame_dir = "frame"
#
#         self.frames: Set[str] = set([])
#         self.frame_elements: Set[str] = set([])
#         self.lexical_units: Set[str] = set([])
#         self.frame_fe_map = dict()
#         self.core_frame_map = dict()
#         self.lu_frame_map = dict()
#         self.simple_lu_frame_map = dict()
#
#         self._read(data_path)
#         print("# frames in ontology: %d", len(self.frames))
#         print("# lexical units in ontology: %d", len(self.lexical_units))
#         print("# frame-elements in ontology: %d",
#               len(self.frame_elements))
#
#     def _simplify_lexunit(self, lexunit):
#
#         # situation: president_(political): president
#         if "_(" in lexunit:
#             speicial_flag_index = lexunit.index("_(")
#             simple_lu = lexunit[:speicial_flag_index]
#             return simple_lu
#
#         # situation: snow_event -> snow event
#         if not lexunit.isalpha():
#             speicial_flag_index = None
#             for i in range(len(lexunit)):
#                 if lexunit[i] != " " and lexunit[i].isalpha() != True:
#                     speicial_flag_index = i
#                     break
#             if speicial_flag_index:
#                 speicial_flag = lexunit[speicial_flag_index]
#                 split_lu_tokens = lexunit.split(speicial_flag)
#                 return " ".join(split_lu_tokens)
#
#         return lexunit
#
#     def _read_ontology_for_frame(self, frame_filename):
#         with open(frame_filename, "r", encoding="utf-8") as frame_file:
#             tree = ElementTree.parse(frame_file)
#         root = tree.getroot()
#
#         fe_for_frame = []
#         core_fe_list = []
#         for fe_tag in root.findall("fn:FE", self._namespace):
#             fe = fe_tag.attrib["name"]
#             fe_for_frame.append(fe)
#             self.frame_elements.add(fe)
#             if fe_tag.attrib["coreType"] == "Core":
#                 core_fe_list.append(fe)
#
#         lu_for_frame = [lu.attrib["name"].split(".")[0] for lu in root.findall("fn:lexUnit", self._namespace)]
#         for lu in lu_for_frame:
#             self.lexical_units.add(lu)
#
#         frame_file.close()
#         return fe_for_frame, core_fe_list, lu_for_frame
#
#     def _read_ontology(self, frame_index_filename: str) -> Set[str]:
#         print(frame_index_filename)
#         with open(frame_index_filename, "r", encoding="utf-8") as frame_file:
#             tree = ElementTree.parse(frame_file)
#         root = tree.getroot()
#
#         self.frames = set([frame.attrib["name"]
#                            for frame in root.findall("fn:frame", self._namespace)])
#
#     def _read(self, file_path: str):
#         frame_index_path = os.path.join(file_path, self._frame_index_filename)
#         print("Reading the frame ontology from %s", frame_index_path)
#         self._read_ontology(frame_index_path)
#
#         max_fe_for_frame = 0
#         total_fe_for_frame = 0.
#         max_core_fe_for_frame = 0
#         total_core_fe_for_frame = 0.
#         max_frames_for_lu = 0
#         total_frames_per_lu = 0.
#         longest_frame = None
#
#         frame_path = os.path.join(file_path, self._frame_dir)
#         print("Reading the frame-element - frame ontology from %s",
#               frame_path)
#         for frame in self.frames:
#             frame_file = os.path.join(frame_path, "{}.xml".format(frame))
#
#             fe_list, core_fe_list, lu_list = self._read_ontology_for_frame(
#                 frame_file)
#             self.frame_fe_map[frame] = fe_list
#             self.core_frame_map[frame] = core_fe_list
#
#             # Compute FE stats
#             total_fe_for_frame += len(self.frame_fe_map[frame])
#             if len(self.frame_fe_map[frame]) > max_fe_for_frame:
#                 max_fe_for_frame = len(self.frame_fe_map[frame])
#                 longest_frame = frame
#
#             # Compute core FE stats
#             total_core_fe_for_frame += len(self.core_frame_map[frame])
#             if len(self.core_frame_map[frame]) > max_core_fe_for_frame:
#                 max_core_fe_for_frame = len(self.core_frame_map[frame])
#
#             for lex_unit in lu_list:
#                 if lex_unit not in self.lu_frame_map:
#                     self.lu_frame_map[lex_unit] = []
#                 self.lu_frame_map[lex_unit].append(frame)
#
#                 simple_lex_unit = self._simplify_lexunit(lex_unit)
#                 if simple_lex_unit not in self.simple_lu_frame_map:
#                     self.simple_lu_frame_map[simple_lex_unit] = []
#                 self.simple_lu_frame_map[simple_lex_unit].append(frame)
#
#                 # Compute frame stats
#                 if len(self.lu_frame_map[lex_unit]) > max_frames_for_lu:
#                     max_frames_for_lu = len(self.lu_frame_map[lex_unit])
#                 total_frames_per_lu += len(self.lu_frame_map[lex_unit])
#
#         print("# max FEs per frame = %d (in frame %s)",
#               max_fe_for_frame, longest_frame)
#         print("# avg FEs per frame = %f",
#               total_fe_for_frame / len(self.frames))
#         print("# max core FEs per frame = %d", max_core_fe_for_frame)
#         print("# avg core FEs per frame = %f",
#               total_core_fe_for_frame / len(self.frames))
#         print("# max frames per LU = %d", max_frames_for_lu)
#         print("# avg frames per LU = %f",
#               total_frames_per_lu / len(self.lu_frame_map))
# class RoleMetrics(Metric):
#     def __init__(self):
#         self.reset()
#
#     @staticmethod
#     def merge_neighboring_spans(predictions):
#         """
#         Merges adjacent spans with the same label, ONLY for the prediction (to encounter spurious ambiguity).
#         Returns
#         -------
#         List[Tuple[int, int, str]]
#             where each tuple represents start, end and label of a span.
#         """
#         merge_predictions = dict()
#         target_fe_dict = dict()
#         for (span_1, span_2), label in predictions:
#             if span_1 not in target_fe_dict:
#                 target_fe_dict[span_1] = []
#             target_fe_dict[span_1].append((span_2[0], span_2[1], label))
#
#         for target, labeled_spans in target_fe_dict.items():
#             labeled_spans_set = set(labeled_spans)
#             sorted_spans = sorted([x for x in list(labeled_spans_set)])
#             prev_start, prev_end, prev_label = sorted_spans[0]
#             for span in sorted_spans[1:]:
#                 if span[2] == prev_label and span[0] == prev_end + 1:
#                     # Merge these two spans.
#                     labeled_spans_set.remove(span)
#                     labeled_spans_set.remove((prev_start, prev_end, prev_label))
#                     labeled_spans_set.add((prev_start, span[1], prev_label))
#                     prev_end = span[1]
#                 else:
#                     prev_start, prev_end, prev_label = span
#             for span in labeled_spans_set:
#                 merge_predictions[(target, (span[0], span[1]))] = span[2]
#         return merge_predictions
#
#     def __call__(self, decoded_roles, metadata_list):
#         predicted_roles_dict = decoded_roles["predicted_roles_dict"]
#         predicted_merge_roles_list = []
#         for predicted_roles, metadata in zip(predicted_roles_dict, metadata_list):
#             gold_frame_elements = metadata["frame_elements_dict"]
#             self._total_role_gold += len(gold_frame_elements)
#             predicted_roles_set = set(predicted_roles.items())
#             predicted_merge_roles = self.merge_neighboring_spans(predicted_roles_set)
#             predicted_merge_roles_list.append(predicted_merge_roles)
#
#             self._total_role_predicted += len(predicted_merge_roles)
#             for (span_1, span_2), label in predicted_merge_roles.items():
#                 ix = (span_1, span_2)
#                 if ix in gold_frame_elements and gold_frame_elements[ix] == label:
#                     self._total_role_matched += 1.0
#
#         decoded_roles["predicted_merge_roles_list"] = predicted_merge_roles_list
#
#     @overrides
#     def get_metric(self, reset=False):
#         role_recall = self._total_role_matched / (self._total_role_gold + 1e-13)
#         role_precision = self._total_role_matched / (self._total_role_predicted + 1e-13)
#         role_f1 = 2.0 * (role_precision * role_recall) / (role_precision + role_recall + 1e-13)
#
#         if reset:
#             self.reset()
#         all_metrics = {}
#         all_metrics["role_precision"] = role_precision
#         all_metrics["role_recall"] = role_recall
#         all_metrics["role_f1"] = role_f1
#         return all_metrics
#
#     @overrides
#     def reset(self):
#         self._total_role_gold = 0
#         self._total_role_matched = 0
#         self._total_role_predicted = 0
# class NodeMetrics(Metric):
#     """
#     Computes precision, recall, and micro-averaged F1 from a list of predicted and gold labels.
#     """
#     def __init__(self, number_of_classes: int, none_label: int=0):
#         self.number_of_classes = number_of_classes
#         self.none_label = none_label
#         self.reset()
#
#     @overrides
#     def __call__(self,
#                  predictions: torch.Tensor,
#                  gold_labels: torch.Tensor,
#                  mask: Optional[torch.Tensor] = None):
#         predictions = predictions.cpu()
#         gold_labels = gold_labels.cpu()
#         mask = mask.cpu()
#         for i in range(self.number_of_classes):
#             if i == self.none_label:
#                 continue
#             self._true_positives += ((predictions==i)*(gold_labels==i)*mask.bool()).sum()
#             self._false_positives += ((predictions==i)*(gold_labels!=i)*mask.bool()).sum()
#             self._true_negatives += ((predictions!=i)*(gold_labels!=i)*mask.bool()).sum()
#             self._false_negatives += ((predictions!=i)*(gold_labels==i)*mask.bool()).sum()
#
#     @overrides
#     def get_metric(self, reset=False):
#         """
#         Returns
#         -------
#         A tuple of the following metrics based on the accumulated count statistics:
#         precision : float
#         recall : float
#         f1-measure : float
#         """
#         precision = float(self._true_positives) / (float(self._true_positives + self._false_positives) + 1e-13)
#         recall = float(self._true_positives) / (float(self._true_positives + self._false_negatives) + 1e-13)
#         f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
#
#         # Reset counts if at end of epoch.
#         if reset:
#             self.reset()
#
#         return precision, recall, f1_measure
#
#     @overrides
#     def reset(self):
#         self._true_positives = 0
#         self._false_positives = 0
#         self._true_negatives = 0
#         self._false_negatives = 0
# class FrameMetrics(Metric):
#     def __init__(self):
#         self.reset()
#
#     def __call__(self, decoded_frames, metadata_list):
#         predicted_frames_dict = decoded_frames["predicted_frames_dict"]
#         for predicted_frames, metadata in zip(predicted_frames_dict, metadata_list):
#             gold_target_spans = set(metadata["origin_frames_dict"].keys())
#             pred_target_spans = set(predicted_frames.keys())
#
#             self._total_target_gold += len(gold_target_spans)
#             self._total_target_predicted += len(pred_target_spans)
#             self._total_target_matched += len(gold_target_spans & pred_target_spans)
#
#             gold_frame_spans = metadata["origin_frames_dict"]
#             self._total_frame_gold += len(gold_frame_spans)
#             predicted_frames_set = set(predicted_frames.items())
#             self._total_frame_predicted += len(predicted_frames_set)
#             for target_ixs, label in gold_frame_spans.items():
#                 if len(target_ixs) > 1: self._total_disc_target_gold += 1.0
#
#             for target_ixs, label in predicted_frames_set:
#                 if len(target_ixs) > 1: self._total_disc_target_predicted += 1.0
#                 if target_ixs in gold_frame_spans and gold_frame_spans[target_ixs] == label:
#                     self._total_frame_matched += 1
#                     if len(target_ixs) > 1: self._total_disc_target_matched += 1.0
#
#     @overrides
#     def get_metric(self, reset=False):
#         target_recall = self._total_target_matched / (self._total_target_gold + 1e-13)
#         target_precision =  self._total_target_matched / (self._total_target_predicted + 1e-13)
#         target_f1 = 2.0 * (target_precision * target_recall) / (target_precision + target_recall + 1e-13)
#
#         frame_recall = self._total_frame_matched / (self._total_frame_gold + 1e-13)
#         frame_precision =  self._total_frame_matched / (self._total_frame_predicted + 1e-13)
#         frame_f1 = 2.0 * (frame_precision * frame_recall) / (frame_precision + frame_recall + 1e-13)
#
#         disc_target_recall = self._total_disc_target_matched / (self._total_disc_target_gold + 1e-13)
#         disc_target_precision = self._total_disc_target_matched / (self._total_disc_target_predicted + 1e-13)
#         disc_target_f1 = 2.0 * (disc_target_precision * disc_target_recall) / (disc_target_precision + disc_target_recall + 1e-13)
#
#         if reset:
#             self.reset()
#         all_metrics = {}
#         all_metrics["target_precision"] = target_precision
#         all_metrics["target_recall"] = target_recall
#         all_metrics["target_f1"] = target_f1
#         all_metrics["frame_precision"] = frame_precision
#         all_metrics["frame_recall"] = frame_recall
#         all_metrics["frame_f1"] = frame_f1
#         all_metrics["disc_target_precision"] = disc_target_precision
#         all_metrics["disc_target_recall"] = disc_target_recall
#         all_metrics["disc_target_f1"] = disc_target_f1
#         return all_metrics
#
#     @overrides
#     def reset(self):
#         self._total_target_gold = 0
#         self._total_target_matched = 0
#         self._total_target_predicted = 0
#         self._total_frame_gold = 0
#         self._total_frame_matched = 0
#         self._total_frame_predicted = 0
#         self._total_disc_target_gold = 0
#         self._total_disc_target_matched = 0
#         self._total_disc_target_predicted = 0
#
# class EdgeMetrics(Metric):
#     """
#     Computes precision, recall, and micro-averaged F1 from a list of predicted and gold spans.
#     """
#
#     def __init__(self):
#         self.reset()
#
#     @overrides
#     def __call__(self, output_edges, metadata_list):
#
#         predicted_p2p_edges_list = output_edges["decoded_p2p_edges_dict"]
#         predicted_p2r_edges_list = output_edges["decoded_p2r_edges_dict"]
#
#         for predicted_p2p_edges, predicted_p2r_edges, metadata in zip(predicted_p2p_edges_list,
#                                                                       predicted_p2r_edges_list,
#                                                                       metadata_list):
#             gold_p2p_edges = metadata["p2p_edges_dict"]
#             self._total_p2p_edge_gold += len(gold_p2p_edges)
#             self._total_p2p_edge_predicted += len(predicted_p2p_edges)
#             for (span_1, span_2), label in predicted_p2p_edges.items():
#                 ix = (span_1, span_2)
#                 if ix in gold_p2p_edges and gold_p2p_edges[ix] == label:
#                     self._total_p2p_edge_matched += 1
#
#             gold_p2r_edges = metadata["p2r_edges_dict"]
#             self._total_p2r_edge_gold += len(gold_p2r_edges)
#             self._total_p2r_edge_predicted += len(predicted_p2r_edges)
#             for (span_1, span_2), label in predicted_p2r_edges.items():
#                 ix = (span_1, span_2)
#                 if ix in gold_p2r_edges and gold_p2r_edges[ix] == label:
#                     self._total_p2r_edge_matched += 1
#
#     @overrides
#     def get_metric(self, reset=False):
#         p2p_edges_recall = self._total_p2p_edge_matched / (self._total_p2p_edge_gold + 1e-13)
#         p2p_edges_precision = self._total_p2p_edge_matched / (self._total_p2p_edge_predicted + 1e-13)
#         p2p_edges_f1 = 2.0 * (p2p_edges_precision * p2p_edges_recall) / (p2p_edges_precision + p2p_edges_recall + 1e-13)
#
#         p2r_edges_recall = self._total_p2r_edge_matched / (self._total_p2r_edge_gold + 1e-13)
#         p2r_edges_precision = self._total_p2r_edge_matched / (self._total_p2r_edge_predicted + 1e-13)
#         p2r_edges_f1 = 2.0 * (p2r_edges_precision * p2r_edges_recall) / (p2r_edges_precision + p2r_edges_recall + 1e-13)
#
#         # Reset counts if at end of epoch.
#         if reset:
#             self.reset()
#         all_metrics = dict()
#         all_metrics["p2p_edges_precision"] = p2p_edges_precision
#         all_metrics["p2p_edges_recall"] = p2p_edges_recall
#         all_metrics["p2p_edges_f1"] = p2p_edges_f1
#         all_metrics["p2r_edges_precision"] = p2r_edges_precision
#         all_metrics["p2r_edges_recall"] = p2r_edges_recall
#         all_metrics["p2r_edges_f1"] = p2r_edges_f1
#         return all_metrics
#
#     @overrides
#     def reset(self):
#         self._total_p2p_edge_gold = 0
#         self._total_p2p_edge_predicted = 0
#         self._total_p2p_edge_matched = 0
#         self._total_p2r_edge_gold = 0
#         self._total_p2r_edge_predicted = 0
#         self._total_p2r_edge_matched = 0
#
# class EdgeBuilder(Model):
#
#     def __init__(self,
#                  vocab: Vocabulary,
#                  predicate_mention_feedforward: FeedForward,
#                  role_mention_feedforward: FeedForward,
#                  p2p_edges_feedforward: FeedForward,
#                  p2r_edges_feedforward: FeedForward,
#                  predicate_ratio: float,
#                  role_ratio: float,
#                  initializer: InitializerApplicator = InitializerApplicator(),
#                  positive_label_weight: float = 1.0,
#                  regularizer: Optional[RegularizerApplicator] = None) -> None:
#         super(EdgeBuilder, self).__init__(vocab, regularizer)
#
#         self._p2p_edge_labels = max(vocab.get_vocab_size("p2p_edge_labels"), 1)
#         self._p2r_edge_labels = max(vocab.get_vocab_size("p2r_edge_labels"), 1)
#
#         self._p2p_edges_vocab = vocab.get_token_to_index_vocabulary("p2p_edge_labels")
#         self._p2r_edges_vocab = vocab.get_token_to_index_vocabulary("p2r_edge_labels")
#         self._id_to_p2r_edges_vocab = vocab.get_index_to_token_vocabulary("p2r_edge_labels")
#
#         self._num_node_attr = max(vocab.get_vocab_size("node_attr_labels"), 1)
#         self._id_to_node_attr_vocab = vocab.get_index_to_token_vocabulary("node_attr_labels")
#         self._node_attr_vocab = vocab.get_token_to_index_vocabulary("node_attr_labels")
#
#         self._predicate_mention_feedforward = predicate_mention_feedforward
#         self._role_mention_feedforward = role_mention_feedforward
#
#         self._predicate_ratio = predicate_ratio
#         self._role_ratio = role_ratio
#         self._predicate_mention_scorer = TimeDistributed(
#             torch.nn.Linear(predicate_mention_feedforward.get_output_dim(), 1))
#         self._role_mention_scorer = TimeDistributed(torch.nn.Linear(role_mention_feedforward.get_output_dim(), 1))
#
#         self._p2p_edges_feedforward = p2p_edges_feedforward
#         self._p2p_edges_scorer = torch.nn.Linear(p2p_edges_feedforward.get_output_dim(), self._p2p_edge_labels)
#         self._p2r_edges_feedforward = p2r_edges_feedforward
#         self._p2r_edges_scorer = torch.nn.Linear(p2r_edges_feedforward.get_output_dim(), self._p2r_edge_labels)
#
#         self._edge_metrics = EdgeMetrics()
#         self._frame_metrics = FrameMetrics()
#         self._role_metrics = RoleMetrics()
#
#         p2p_edge_weights = torch.cat([torch.tensor([1.0]), positive_label_weight * torch.ones(self._p2p_edge_labels)])
#         p2r_edge_weights = torch.cat([torch.tensor([1.0]), positive_label_weight * torch.ones(self._p2r_edge_labels)])
#         self._p2p_edge_loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1, weight=p2p_edge_weights)
#         self._p2r_edge_loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1, weight=p2r_edge_weights)
#         initializer(self)
#
#     @overrides
#     def forward(self,  # type: ignore
#                 spans: torch.LongTensor,
#                 span_mask: torch.BoolTensor,
#                 span_embeddings: torch.Tensor,
#                 sentence_lengths: torch.Tensor,
#                 output_nodes: Dict[str, Any],
#                 lu_frame_map: Dict[str, List],
#                 frame_fe_map: Dict[str, List],
#                 p2p_edge_labels: torch.IntTensor = None,
#                 p2r_edge_labels: torch.IntTensor = None,
#                 metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
#         num_predicate_spans_to_keep = torch.ceil(self._predicate_ratio * sentence_lengths).int()
#         num_role_spans_to_keep = torch.ceil(self._role_ratio * sentence_lengths).int()
#
#         predicate_span_mentions = self._predicate_mention_feedforward(span_embeddings)
#         role_span_mentions = self._role_mention_feedforward(span_embeddings)
#
#         predicate_span_scores = self._predicate_mention_scorer(predicate_span_mentions).squeeze(-1)
#         role_span_scores = self._role_mention_scorer(role_span_mentions).squeeze(-1)
#
#         top_predicate_span_embeddings, \
#         top_predicate_span_masks, \
#         top_predicate_span_indices, \
#         top_predicate_spans, \
#         top_predicate_span_scores = self._prune_spans(spans, span_mask, predicate_span_mentions, predicate_span_scores,
#                                                       num_predicate_spans_to_keep)
#
#         top_role_span_embeddings, \
#         top_role_span_masks, \
#         top_role_span_indices, \
#         top_role_spans, \
#         top_role_span_scores = self._prune_spans(spans, span_mask, role_span_mentions, role_span_scores,
#                                                  num_role_spans_to_keep)
#
#         p2p_edge_scores, p2r_edge_scores = self.compute_relation_representations(
#             top_role_span_embeddings, top_predicate_span_embeddings, top_role_span_scores, top_predicate_span_scores)
#
#         output_edges = {"spans": spans,
#                         "top_predicate_span_masks": top_predicate_span_masks,
#                         "top_predicate_spans": top_predicate_spans,
#                         "top_predicate_span_indices": top_predicate_span_indices,
#                         "top_predicate_span_scores": top_predicate_span_scores,
#                         "num_predicate_spans_to_keep": num_predicate_spans_to_keep,
#                         "top_role_span_masks": top_role_span_masks,
#                         "top_role_spans": top_role_spans,
#                         "top_role_span_indices": top_role_span_indices,
#                         "top_role_span_scores": top_role_span_scores,
#                         "num_role_spans_to_keep": num_role_spans_to_keep,
#                         "p2p_edge_scores": p2p_edge_scores,
#                         "p2r_edge_scores": p2r_edge_scores,
#                         "loss": 0.0}
#
#         output_edges = self.predict_labels(output_edges, output_nodes, lu_frame_map, frame_fe_map, metadata,
#                                            p2p_edge_labels, p2r_edge_labels)
#         return output_edges
#
#     def _prune_spans(self, spans, span_mask, span_mentions, span_scores, num_spans_to_keep):
#         batch_size, num_spans, _ = spans.size()
#         # Shape: (batch_size, num_spans_to_keep) * 3
#         top_span_scores, top_span_mask, top_span_indices = util.masked_topk(
#             span_scores, span_mask, num_spans_to_keep
#         )
#         top_span_scores = top_span_scores.unsqueeze(-1)
#         top_span_mask = top_span_mask.unsqueeze(-1)
#         # For compute efficiently
#         flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)
#         # Shape: (batch_size, num_spans_to_keep, 2)
#         top_spans = util.batched_index_select(spans, top_span_indices, flat_top_span_indices)
#         # Shape: (batch_size, num_spans_to_keep, embedding_size)
#         top_span_embeddings = util.batched_index_select(
#             span_mentions, top_span_indices, flat_top_span_indices
#         )
#
#         return top_span_embeddings, top_span_mask, top_span_indices, top_spans, top_span_scores
#
#     def compute_relation_representations(self,  # type: ignore
#                                          top_role_span_embeddings,
#                                          top_predicate_span_embeddings,
#                                          top_role_mention_scores,
#                                          top_predicate_mention_scores) -> Dict[str, torch.Tensor]:
#
#         p2p_edge_pairwise_embeddings = self._compute_span_pair_embeddings(top_predicate_span_embeddings,
#                                                                           top_predicate_span_embeddings)
#         p2r_edge_pairwise_embeddings = self._compute_span_pair_embeddings(top_predicate_span_embeddings,
#                                                                           top_role_span_embeddings)
#         batch_size = p2p_edge_pairwise_embeddings.size(0)
#         max_num_predicate_spans = p2r_edge_pairwise_embeddings.size(1)
#         max_num_role_spans = p2r_edge_pairwise_embeddings.size(2)
#
#         p2p_edge_feature_dim = self._p2p_edges_feedforward.input_dim
#         p2r_edge_feature_dim = self._p2r_edges_feedforward.input_dim
#
#         p2p_edge_embeddings_flat = p2p_edge_pairwise_embeddings.view(-1, p2p_edge_feature_dim)
#         p2r_edge_embeddings_flat = p2r_edge_pairwise_embeddings.view(-1, p2r_edge_feature_dim)
#
#         p2p_edge_projected_flat = self._p2p_edges_feedforward(p2p_edge_embeddings_flat)
#         p2p_edge_scores_flat = self._p2p_edges_scorer(p2p_edge_projected_flat)
#
#         p2r_edge_projected_flat = self._p2r_edges_feedforward(p2r_edge_embeddings_flat)
#         p2r_edge_scores_flat = self._p2r_edges_scorer(p2r_edge_projected_flat)
#
#         p2p_edge_scores = p2p_edge_scores_flat.view(batch_size, max_num_predicate_spans, max_num_predicate_spans, -1)
#         p2r_edge_scores = p2r_edge_scores_flat.view(batch_size, max_num_predicate_spans, max_num_role_spans, -1)
#
#         p2p_edge_scores += (
#                     top_predicate_mention_scores.unsqueeze(-1) + top_predicate_mention_scores.transpose(1, 2).unsqueeze(
#                 -1))
#         p2r_edge_scores += (
#                     top_predicate_mention_scores.unsqueeze(-1) + top_role_mention_scores.transpose(1, 2).unsqueeze(-1))
#
#         p2p_edge_shape = [p2p_edge_scores.size(0), p2p_edge_scores.size(1), p2p_edge_scores.size(2), 1]
#         p2r_edge_shape = [p2r_edge_scores.size(0), p2r_edge_scores.size(1), p2r_edge_scores.size(2), 1]
#         dummy_p2p_edge_scores = p2p_edge_scores.new_zeros(*p2p_edge_shape)
#         dummy_p2r_edge_scores = p2r_edge_scores.new_zeros(*p2r_edge_shape)
#
#         p2p_edge_scores = torch.cat([dummy_p2p_edge_scores, p2p_edge_scores], -1)
#         p2r_edge_scores = torch.cat([dummy_p2r_edge_scores, p2r_edge_scores], -1)
#
#         return p2p_edge_scores, p2r_edge_scores
#
#     def predict_labels(self, output_edges, output_nodes, lu_frame_map, frame_fe_map, metadata, p2p_edges_labels=None,
#                        p2r_edges_labels=None):
#         p2p_edge_scores = output_edges["p2p_edge_scores"]
#         p2r_edge_scores = output_edges["p2r_edge_scores"]
#         # Evaluate loss and F1 if labels were provided.
#         if p2p_edges_labels is not None and p2r_edges_labels is not None:
#             # Compute cross-entropy loss.
#             gold_p2p_edges = self._get_pruned_gold_relations(p2p_edges_labels,
#                                                              output_edges["top_predicate_span_indices"],
#                                                              output_edges["top_predicate_span_masks"],
#                                                              output_edges["top_predicate_span_indices"],
#                                                              output_edges["top_predicate_span_masks"])
#             gold_p2p_edges = gold_p2p_edges.long()
#             p2p_edge_cross_entropy = self._get_cross_entropy_loss(p2p_edge_scores, gold_p2p_edges, p2p=True)
#             output_edges["loss"] += p2p_edge_cross_entropy
#
#             gold_p2r_edges = self._get_pruned_gold_relations(p2r_edges_labels,
#                                                              output_edges["top_role_span_indices"],
#                                                              output_edges["top_role_span_masks"],
#                                                              output_edges["top_predicate_span_indices"],
#                                                              output_edges["top_predicate_span_masks"])
#             gold_p2r_edges = gold_p2r_edges.long()
#             p2r_edge_cross_entropy = self._get_cross_entropy_loss(p2r_edge_scores, gold_p2r_edges)
#             output_edges["loss"] += p2r_edge_cross_entropy
#
#         _, predicted_p2p_edges = p2p_edge_scores.max(-1)
#         _, predicted_p2r_edges = p2r_edge_scores.max(-1)
#         predicted_p2p_edges -= 1
#         predicted_p2r_edges -= 1
#         output_edges["predicted_p2p_edges"] = predicted_p2p_edges
#         output_edges["predicted_p2r_edges"] = predicted_p2r_edges
#         self._decode_edges(output_edges)
#         self._edge_metrics(output_edges, metadata)
#
#         decoded_frames = self._decode_frames(output_edges, output_nodes, lu_frame_map, metadata)
#         decoded_roles = self._decode_roles(output_edges, decoded_frames, frame_fe_map, metadata)
#         output_edges["predicted_frames"] = decoded_frames["predicted_frames"]
#         output_edges["predicted_roles"] = decoded_roles["predicted_roles"]
#         print(output_edges["predicted_frames"])
#         print(output_edges["predicted_roles"])
#         self._frame_metrics(decoded_frames, metadata)
#         self._role_metrics(decoded_roles, metadata)
#         return output_edges
#
#     def _decode_edges(self, output_edges):
#         top_predicate_spans_batch = output_edges["top_predicate_spans"].detach().cpu()
#         top_role_spans_batch = output_edges["top_role_spans"].detach().cpu()
#         predicted_p2p_edges_batch = output_edges["predicted_p2p_edges"].detach().cpu()
#         predicted_p2r_edges_batch = output_edges["predicted_p2r_edges"].detach().cpu()
#         num_role_spans_to_keep_batch = output_edges["num_role_spans_to_keep"].detach().cpu()
#         num_predicate_spans_to_keep_batch = output_edges["num_predicate_spans_to_keep"].detach().cpu()
#         res_p2p_edges_dict = []
#         res_p2p_edges = []
#         res_p2r_edges_dict = []
#         res_p2r_edges = []
#
#         # Collect predictions for each sentence in minibatch.
#         zipped = zip(top_predicate_spans_batch, top_role_spans_batch,
#                      predicted_p2p_edges_batch, predicted_p2r_edges_batch,
#                      num_role_spans_to_keep_batch, num_predicate_spans_to_keep_batch)
#         for top_predicate_spans, top_role_spans, predicted_p2p_edges, predicted_p2r_edges, \
#             num_role_spans_to_keep, num_predicate_spans_to_keep in zipped:
#             entry_p2p_edges_dict = {}
#             entry_p2p_edges = []
#             entry_p2r_edges_dict = {}
#             entry_p2r_edges = []
#
#             role_keep = num_role_spans_to_keep.item()
#             predicate_keep = num_predicate_spans_to_keep.item()
#             top_role_spans = [tuple(x) for x in top_role_spans.tolist()]
#             top_predicate_spans = [tuple(x) for x in top_predicate_spans.tolist()]
#
#             # Iterate over all span pairs and labels. Record the span if the label isn't null.
#             for i, j in itertools.product(range(predicate_keep), range(predicate_keep)):
#                 span_1 = top_predicate_spans[i]
#                 span_2 = top_predicate_spans[j]
#                 label = predicted_p2p_edges[i, j].item()
#                 if label >= 0:
#                     label_name = self.vocab.get_token_from_index(label, namespace="p2p_edge_labels")
#                     entry_p2p_edges_dict[(span_1, span_2)] = label_name
#                     list_entry = (span_1[0], span_1[1], span_2[0], span_2[1], label_name)
#                     entry_p2p_edges.append(list_entry)
#
#             for i, j in itertools.product(range(predicate_keep), range(role_keep)):
#                 span_1 = top_predicate_spans[i]
#                 span_2 = top_role_spans[j]
#                 label = predicted_p2r_edges[i, j].item()
#                 if label >= 0:
#                     label_name = self.vocab.get_token_from_index(label, namespace="p2r_edge_labels")
#                     entry_p2r_edges_dict[(span_1, span_2)] = label_name
#                     list_entry = (span_1[0], span_1[1], span_2[0], span_2[1], label_name)
#                     entry_p2r_edges.append(list_entry)
#
#             res_p2p_edges_dict.append(entry_p2p_edges_dict)
#             res_p2p_edges.append(entry_p2p_edges)
#             res_p2r_edges_dict.append(entry_p2r_edges_dict)
#             res_p2r_edges.append(entry_p2r_edges)
#
#         output_edges["decoded_p2p_edges_dict"] = res_p2p_edges_dict
#         output_edges["decoded_p2r_edges"] = res_p2p_edges
#         output_edges["decoded_p2r_edges_dict"] = res_p2r_edges_dict
#         output_edges["decoded_p2r_edges"] = res_p2r_edges
#
#     def _decode_frames(self, output_edges, output_nodes, lu_frame_map, metadata_list):
#         # decoded the predicate-predicate edges
#
#         decoded_p2p_edges_dict = output_edges["decoded_p2p_edges_dict"]
#         decoded_outputs = {}
#
#         # decode the predicates and frames
#         decoded_node_types_dict = output_nodes["decoded_node_types_dict"]
#         decoded_node_attr_scores_dict = output_nodes["decoded_node_attr_scores_dict"]
#
#         predicted_frames_dict = []
#         predicted_frames = []
#         # resolve the issues caused by discontinuous spans
#         predicted_cspan2frames_dict = []  # continuous span and their corresponding frame
#         predicted_dspan2frames_dict = []  # discontinuous span and their corresponding frame
#         for node_types, node_attr_scores, p2p_edges, metadata in zip(decoded_node_types_dict,
#                                                                      decoded_node_attr_scores_dict,
#                                                                      decoded_p2p_edges_dict, metadata_list):
#             frames = dict()
#             cspan2frames = dict()
#             dspan2frames = dict()
#             sentence_tokens = metadata['sentence']
#             lemma_tokens = metadata['lemmas']
#             for node_ix, node_type in node_types.items():
#                 # discontinuous predicates
#                 if "PPRD" in node_type:
#                     frame_ix = set()
#                     frame_score = []
#                     frame_ix.add(node_ix)
#                     frame_score.append(node_attr_scores[node_ix])
#                     for (arg1_span, arg2_span), label in p2p_edges.items():
#                         if label != 'Continuous':
#                             continue
#                         if node_ix == arg1_span:
#                             if arg2_span not in node_attr_scores:
#                                 continue
#                             frame_ix.add(arg2_span)
#                             frame_score.append(node_attr_scores[arg2_span])
#                         if node_ix == arg2_span:
#                             if arg1_span not in node_attr_scores:
#                                 continue
#                             frame_ix.add(arg1_span)
#                             frame_score.append(node_attr_scores[arg1_span])
#
#                     frame_ix = tuple(sorted(list(frame_ix)))
#                     if frame_ix not in frames and len(frame_ix) >= 2 and is_clique(frame_ix, p2p_edges):
#                         frame_label = get_constrained_frame_label(sentence_tokens, lemma_tokens, frame_ix, frame_score,
#                                                                   lu_frame_map, \
#                                                                   self._node_attr_vocab, self._id_to_node_attr_vocab,
#                                                                   self._num_node_attr)
#                         if frame_label == "O":
#                             continue
#                         frames[frame_ix] = frame_label
#                         for span in frame_ix:
#                             dspan2frames[span] = (frame_ix, frame_label)
#                 # continuous predicates
#                 if "FPRD" in node_type:
#                     frame_ix = tuple([node_ix])
#                     frame_score = [node_attr_scores[node_ix]]
#                     frame_label = get_constrained_frame_label(sentence_tokens, lemma_tokens, frame_ix, frame_score,
#                                                               lu_frame_map, \
#                                                               self._node_attr_vocab, self._id_to_node_attr_vocab,
#                                                               self._num_node_attr)
#                     if frame_label == "O":
#                         continue
#                     frames[frame_ix] = frame_label
#                     cspan2frames[frame_ix[0]] = (frame_ix, frame_label)
#             predicted_frames.append(list(frames.items()))
#             predicted_frames_dict.append(frames)
#             predicted_cspan2frames_dict.append(cspan2frames)
#             predicted_dspan2frames_dict.append(dspan2frames)
#
#         decoded_outputs["predicted_frames"] = predicted_frames
#         decoded_outputs["predicted_frames_dict"] = predicted_frames_dict
#         decoded_outputs["predicted_cspan2frames_dict"] = predicted_cspan2frames_dict
#         decoded_outputs["predicted_dspan2frames_dict"] = predicted_dspan2frames_dict
#         return decoded_outputs
#
#     def _decode_roles(self, output_edges, decoded_frames, frame_fe_map, metadata_list):
#         top_predicate_spans_batch = output_edges["top_predicate_spans"].detach().cpu()
#         top_role_spans_batch = output_edges["top_role_spans"].detach().cpu()
#         p2r_edge_scores_batch = output_edges["p2r_edge_scores"].detach().cpu()
#         num_role_spans_to_keep_batch = output_edges["num_role_spans_to_keep"].detach().cpu()
#         num_predicate_spans_to_keep_batch = output_edges["num_predicate_spans_to_keep"].detach().cpu()
#
#         predicted_cspan2frames_batch = decoded_frames['predicted_cspan2frames_dict']
#         predicted_dspan2frames_batch = decoded_frames['predicted_dspan2frames_dict']
#
#         decoded_outputs = {}
#         predicted_roles_dict = []
#         predicted_roles = []
#         # Collect predictions for each sentence in minibatch.
#         zipped = zip(top_role_spans_batch, top_predicate_spans_batch, p2r_edge_scores_batch,
#                      num_role_spans_to_keep_batch, num_predicate_spans_to_keep_batch,
#                      predicted_cspan2frames_batch, predicted_dspan2frames_batch)
#         for top_role_spans, top_predicate_spans, predicted_p2r_edge_scores, \
#             num_role_spans_to_keep, num_predicate_spans_to_keep, \
#             predicted_cspan2frames, predicted_dspan2frames in zipped:
#             role_keep = num_role_spans_to_keep.item()
#             predicate_keep = num_predicate_spans_to_keep.item()
#             top_role_spans = [tuple(x) for x in top_role_spans.tolist()]
#             top_predicate_spans = [tuple(x) for x in top_predicate_spans.tolist()]
#
#             res_role_dict = {}
#             res_role_list = []
#
#             conflict_frame_elements = {}
#             for i, j in itertools.product(range(predicate_keep), range(role_keep)):
#                 span_1 = top_predicate_spans[i]
#                 span_2 = top_role_spans[j]
#                 scores = predicted_p2r_edge_scores[i, j]
#
#                 if span_1 in predicted_cspan2frames:
#                     frame_ix, frame_label = predicted_cspan2frames[span_1]
#                     valid_frame_elements_mask = torch.zeros(self._p2r_edge_labels + 1).detach().cpu()
#                     valid_frame_elements_mask[0] = True
#                     valid_frame_elements = torch.LongTensor(
#                         [self._p2r_edges_vocab[fe] + 1 for fe in frame_fe_map[frame_label] if
#                          fe in self._p2r_edges_vocab])
#                     valid_frame_elements_mask.index_fill_(-1, valid_frame_elements, True)
#                     frame_element_label_id = torch.argmax(scores + torch.log(valid_frame_elements_mask)).item() - 1
#
#                     if frame_element_label_id >= 0:
#                         frame_element_label = self._id_to_p2r_edges_vocab[frame_element_label_id]
#                         res_role_dict[((frame_ix, frame_label), span_2)] = frame_element_label
#                         list_role_entry = ((frame_ix, frame_label), span_2, frame_element_label)
#                         res_role_list.append(list_role_entry)
#
#                 # resolve the conflicts by the condtion where role nodes connect with two or more predicates nodes of a predicate but evoke different frame elements
#                 if span_1 in predicted_dspan2frames:
#                     frame_ix, frame_label = predicted_dspan2frames[span_1]
#                     if (frame_ix, frame_label) not in conflict_frame_elements:
#                         conflict_frame_elements[(frame_ix, frame_label)] = defaultdict(list)
#                     conflict_frame_elements[(frame_ix, frame_label)][span_2].append(scores)
#
#             for (frame_ix, frame_label), sub_roles_dict in conflict_frame_elements.items():
#                 valid_frame_elements_mask = torch.zeros(self._p2r_edge_labels + 1).detach().cpu()
#                 valid_frame_elements_mask[0] = True
#                 valid_frame_elements = torch.LongTensor(
#                     [self._p2r_edges_vocab[fe] + 1 for fe in frame_fe_map[frame_label] if fe in self._p2r_edges_vocab])
#                 valid_frame_elements_mask.index_fill_(-1, valid_frame_elements, True)
#                 for role_span, role_scores in sub_roles_dict.items():
#                     frame_element_label_id = torch.argmax(
#                         sum(role_scores) + torch.log(valid_frame_elements_mask)).item() - 1
#                     if frame_element_label_id < 0:
#                         continue
#                     frame_element_label = self._id_to_p2r_edges_vocab[frame_element_label_id]
#                     res_role_dict[((frame_ix, frame_label), role_span)] = frame_element_label
#                     list_role_entry = ((frame_ix, frame_label), role_span, frame_element_label)
#                     res_role_list.append(list_role_entry)
#
#             predicted_roles_dict.append(res_role_dict)
#             predicted_roles.append(res_role_list)
#
#         decoded_outputs["predicted_roles_dict"] = predicted_roles_dict
#         decoded_outputs["predicted_roles"] = predicted_roles
#         return decoded_outputs
#
#     @overrides
#     def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#         all_metrics = self._edge_metrics.get_metric(reset)
#         all_metrics.update(self._frame_metrics.get_metric(reset))
#         all_metrics.update(self._role_metrics.get_metric(reset))
#         return all_metrics
#
#     @staticmethod
#     def _compute_span_pair_embeddings(top_left_span_embeddings, top_right_span_embeddings):
#
#         # Shape: (batch_size, num_role_spans_to_keep, num_target_spans_to_keep, embedding_size)
#         num_left_candiates = top_left_span_embeddings.size(1)
#         num_right_candidates = top_right_span_embeddings.size(1)
#
#         embeddings_left_expanded = top_left_span_embeddings.unsqueeze(2)
#         embeddings_left_tiled = embeddings_left_expanded.repeat(1, 1, num_right_candidates, 1)
#
#         embeddings_right_expanded = top_right_span_embeddings.unsqueeze(1)
#         embeddings_right_tiled = embeddings_right_expanded.repeat(1, num_left_candiates, 1, 1)
#
#         similarity_embeddings = embeddings_left_expanded * embeddings_right_expanded
#
#         pair_embeddings_list = [embeddings_left_tiled, embeddings_right_tiled, similarity_embeddings]
#         # pair_embeddings_list = [embeddings_target_tiled, embeddings_role_tiled]
#         pair_embeddings = torch.cat(pair_embeddings_list, dim=3)
#
#         return pair_embeddings
#
#     @staticmethod
#     def _get_pruned_gold_relations(relation_labels, top_right_span_indices, top_right_span_masks, top_left_span_indices,
#                                    top_left_span_masks):
#
#         relations = []
#         for sliced, right_ixs, left_ixs, top_right_span_mask, top_left_span_mask in zip(relation_labels,
#                                                                                         top_right_span_indices,
#                                                                                         top_left_span_indices,
#                                                                                         top_right_span_masks.bool(),
#                                                                                         top_left_span_masks.bool()):
#             entry = sliced[left_ixs][:, right_ixs].unsqueeze(0)
#             mask_entry = top_left_span_mask & top_right_span_mask.transpose(0, 1).unsqueeze(0)
#             entry[mask_entry] += 1
#             entry[~mask_entry] = -1
#             relations.append(entry)
#
#         return torch.cat(relations, dim=0)
#
#     def _get_cross_entropy_loss(self, relation_scores, relation_labels, p2p=False):
#
#         # Need to add 1 because the null label is 0, to line up with indices into prediction matrix.
#         labels_flat = relation_labels.view(-1)
#         # Compute cross-entropy loss.
#         if p2p:
#             # Need to add one for the null class.
#             scores_flat = relation_scores.view(-1, self._p2p_edge_labels + 1)
#             loss = self._p2p_edge_loss(scores_flat, labels_flat)
#         else:
#             scores_flat = relation_scores.view(-1, self._p2r_edge_labels + 1)
#             loss = self._p2r_edge_loss(scores_flat, labels_flat)
#         return loss
#
#
# class NodeBuilder(Model):
#
#     def __init__(self,
#                  vocab: Vocabulary,
#                  node_feedforward: FeedForward,
#                  initializer: InitializerApplicator = InitializerApplicator(),
#                  regularizer: Optional[RegularizerApplicator] = None) -> None:
#         super(NodeBuilder, self).__init__(vocab, regularizer)
#
#         self.vocab = vocab
#         self._num_node_type_labels = vocab.get_vocab_size('node_type_labels')
#         self._num_node_attr_labels = vocab.get_vocab_size('node_attr_labels')
#
#         self.node_type_vocab = vocab.get_token_to_index_vocabulary("node_type_labels")
#         self.node_attr_vocab = vocab.get_token_to_index_vocabulary("node_attr_labels")
#
#         self._null_node_type_id = self.node_type_vocab["O"]
#         self._null_node_attr_id = self.node_attr_vocab["O"]
#
#         self._node_feedforward = node_feedforward
#         self._node_type_scorer = TimeDistributed(torch.nn.Linear(
#             node_feedforward.get_output_dim(),
#             self._num_node_type_labels - 1))
#
#         self._node_attr_scorer = TimeDistributed(torch.nn.Linear(
#             node_feedforward.get_output_dim(),
#             self._num_node_attr_labels - 1))
#
#         self._node_type_metrics = NodeMetrics(self._num_node_type_labels, self._null_node_type_id)
#         self._node_attr_metrics = NodeMetrics(self._num_node_attr_labels, self._null_node_attr_id)
#
#         self._loss = torch.nn.CrossEntropyLoss(reduction="sum")
#         initializer(self)
#
#     @overrides
#     def forward(self,  # type: ignore
#                 spans: torch.LongTensor,
#                 span_mask: torch.BoolTensor,
#                 span_embeddings: torch.Tensor,
#                 node_type_labels: torch.LongTensor = None,
#                 node_attr_labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
#
#         node_mentions = self._node_feedforward(span_embeddings)
#         node_type_scores = self._node_type_scorer(node_mentions)
#         node_attr_scores = self._node_attr_scorer(node_mentions)
#
#         # shape = [batch_size, num_spans_to_keep, num_classes-1]
#         node_type_shape = [node_type_scores.size(0), node_type_scores.size(1), 1]
#         node_attr_shape = [node_attr_scores.size(0), node_attr_scores.size(1), 1]
#         dummy_node_type_scores = node_type_scores.new_zeros(*node_type_shape)
#         dummy_node_attr_scores = node_attr_scores.new_zeros(*node_attr_shape)
#
#         # shape = [batch_size, num_spans_to_keep, num_classes]
#         node_type_scores = torch.cat([dummy_node_type_scores, node_type_scores], -1)
#         node_attr_scores = torch.cat([dummy_node_attr_scores, node_attr_scores], -1)
#
#         _, predicted_node_types = node_type_scores.max(2)
#         _, predicted_node_attrs = node_attr_scores.max(2)
#         output_dict = {"spans": spans,
#                        "span_mask": span_mask,
#                        "predicted_node_types": predicted_node_types,
#                        "predicted_node_attrs": predicted_node_attrs,
#                        "predicted_node_attr_scores": node_attr_scores}
#
#         if node_type_labels is not None and node_attr_labels is not None:
#             flat_mask = span_mask.view(-1).bool()
#             flat_node_type_scores = node_type_scores.view(-1, self._num_node_type_labels)
#             flat_node_type_labels = node_type_labels.view(-1)
#             node_type_loss = self._loss(flat_node_type_scores, flat_node_type_labels)
#
#             flat_node_attr_scores = node_attr_scores.view(-1, self._num_node_attr_labels)
#             flat_node_attr_labels = node_attr_labels.view(-1)
#             node_attr_loss = self._loss(flat_node_attr_scores[flat_mask], flat_node_attr_labels[flat_mask])
#             output_dict["loss"] = node_type_loss + node_attr_loss
#
#         self._node_type_metrics(predicted_node_types, node_type_labels, span_mask)
#         self._node_attr_metrics(predicted_node_attrs, node_attr_labels, span_mask)
#         self._decode(output_dict)
#         return output_dict
#
#     def _decode(self, output_dict):
#         predicted_node_type_batch = output_dict["predicted_node_types"].detach().cpu()
#         predicted_node_attr_batch = output_dict["predicted_node_attrs"].detach().cpu()
#         predicted_node_attr_scores_batch = output_dict["predicted_node_attr_scores"].detach().cpu()
#         spans_batch = output_dict["spans"].detach().cpu()
#         span_mask_batch = output_dict["span_mask"].detach().cpu().squeeze(-1).bool()
#
#         res_node_type_list = []
#         res_node_type_dict = []
#         res_node_attr_list = []
#         res_node_attr_dict = []
#         res_node_attr_score_list = []
#         res_node_attr_score_dict = []
#         for spans, span_mask, predicted_node_types, predicted_node_attrs, predicted_node_attr_scores \
#                 in zip(spans_batch, span_mask_batch, predicted_node_type_batch, predicted_node_attr_batch,
#                        predicted_node_attr_scores_batch):
#             node_type_list = []
#             node_type_entry_dict = {}
#             node_attr_list = []
#             node_attr_entry_dict = {}
#             node_attr_score_list = []
#             node_attr_score_entry_dict = {}
#             for span, node_type, node_attr, node_attr_scores in zip(spans, predicted_node_types[span_mask],
#                                                                     predicted_node_attrs[span_mask],
#                                                                     predicted_node_attr_scores[span_mask]):
#                 node_type = node_type.item()
#                 node_attr = node_attr.item()
#                 the_span = (span[0].item(), span[1].item())
#                 if node_type > 0:
#                     node_type_label = self.vocab.get_token_from_index(node_type, "node_type_labels")
#                     node_type_list.append((the_span, node_type_label))
#                     node_type_entry_dict[the_span] = node_type_label
#                     node_attr_score_list.append((the_span, node_attr_scores))
#                     node_attr_score_entry_dict[the_span] = node_attr_scores
#                 if node_attr > 0:
#                     node_attr_label = self.vocab.get_token_from_index(node_attr, "node_attr_labels")
#                     node_attr_list.append((the_span, node_attr_label))
#                     node_attr_entry_dict[the_span] = node_attr_label
#
#             res_node_type_list.append(node_type_list)
#             res_node_type_dict.append(node_type_entry_dict)
#             res_node_attr_score_list.append(node_attr_score_list)
#             res_node_attr_score_dict.append(node_attr_score_entry_dict)
#             res_node_attr_list.append(node_attr_list)
#             res_node_attr_dict.append(node_attr_entry_dict)
#
#         output_dict["decoded_node_types"] = res_node_type_list
#         output_dict["decoded_node_types_dict"] = res_node_type_dict
#         output_dict["decoded_node_attrs"] = res_node_attr_list
#         output_dict["decoded_node_attrs_dict"] = res_node_attr_dict
#         output_dict["decoded_node_attr_scores"] = res_node_attr_score_list
#         output_dict["decoded_node_attr_scores_dict"] = res_node_attr_score_dict
#         return output_dict
#
#     @overrides
#     def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#         node_type_precision, node_type_recall, node_type_f1 = self._node_type_metrics.get_metric(reset)
#         node_attr_precision, node_attr_recall, node_attr_f1 = self._node_attr_metrics.get_metric(reset)
#         return {"node_type_precision": node_type_precision,
#                 "node_type_recall": node_type_recall,
#                 "node_type_f1": node_type_f1,
#                 "node_attr_precision": node_attr_precision,
#                 "node_attr_recall": node_attr_recall,
#                 "node_attr_f1": node_attr_f1}
