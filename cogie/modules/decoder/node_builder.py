# import itertools
# import difflib
# import logging
# from typing import Any, Dict, List, Optional
#
# import torch
# import torch.nn.functional as F
#
# from overrides import overrides
# from allennlp.common.util import pad_sequence_to_length
# from allennlp.data import Vocabulary
# from allennlp.models.model import Model
# from allennlp.modules import FeedForward
# from allennlp.modules import TimeDistributed
# from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
#
# from framenet_parser.utils import FrameOntology, get_tag_mask, merge_spans
# from framenet_parser.metrics.node_metrics import NodeMetrics
#
# logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
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
