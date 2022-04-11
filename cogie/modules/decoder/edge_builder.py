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
# from framenet_parser.metrics.frame_metrics import FrameMetrics
# from framenet_parser.metrics.role_metrics import RoleMetrics
# from framenet_parser.metrics.edge_metrics import EdgeMetrics
# from framenet_parser.utils import is_clique, get_constrained_frame_label
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