"""
@Author: jinzhuan
@File: metrics.py
@Desc: 
"""
import inspect
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Union
from copy import deepcopy
import re

import numpy as np
import torch

from cogie.utils import Vocabulary
from .utils import _CheckError
from .utils import _CheckRes
from .utils import _build_args
from .utils import _check_arg_dict_list
from .utils import _get_func_signature
from .utils import seq_len_to_mask
from .utils import ConfusionMatrix
from typing import List, Optional, Iterable


class MetricBase(object):

    def __init__(self):
        self._param_map = {}  # key is param in function, value is input param.
        self._checked = False
        self._metric_name = self.__class__.__name__

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)

    @property
    def param_map(self):
        if len(self._param_map) == 0:
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset=True):
        raise NotImplemented

    def set_metric_name(self, name: str):
        self._metric_name = name
        return self

    def get_metric_name(self):
        return self._metric_name

    def _init_param_map(self, key_map=None, **kwargs):
        value_counter = defaultdict(set)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise TypeError("key_map must be `dict`, got {}.".format(type(key_map)))
            for key, value in key_map.items():
                if value is None:
                    self._param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self._param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self._param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self._param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")

        # check consistence between signature and _param_map
        func_spect = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature.")

    def __call__(self, pred_dict, target_dict):

        if not self._checked:
            if not callable(self.evaluate):
                raise TypeError(f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}.")
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.evaluate)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_target_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]

        # missing
        if not self._checked:
            duplicated = []
            for input_arg, mapped_arg in self._reverse_param_map.items():
                if input_arg in pred_dict and input_arg in target_dict:
                    duplicated.append(input_arg)
            check_res = _check_arg_dict_list(self.evaluate, [mapped_pred_dict, mapped_target_dict])
            # only check missing.
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = f"{self._param_map[func_arg]}" + f"(assign to `{func_arg}` " \
                                                                         f"in `{self.__class__.__name__}`)"

            check_res = _CheckRes(missing=replaced_missing,
                                  unused=check_res.unused,
                                  duplicated=duplicated,
                                  required=check_res.required,
                                  all_needed=check_res.all_needed,
                                  varargs=check_res.varargs)

            if check_res.missing or check_res.duplicated:
                raise _CheckError(check_res=check_res,
                                  func_signature=_get_func_signature(self.evaluate))
            self._checked = True
        refined_args = _build_args(self.evaluate, **mapped_pred_dict, **mapped_target_dict)

        self.evaluate(**refined_args)

        return


class ConfusionMatrixMetric(MetricBase):
    def __init__(self,
                 vocab=None,
                 pred=None,
                 target=None,
                 seq_len=None,
                 print_ratio=False
                 ):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.confusion_matrix = ConfusionMatrix(
            vocab=vocab,
            print_ratio=print_ratio,
        )

    def evaluate(self, pred, target, seq_len=None):

        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(
                f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                f"got {type(seq_len)}.")

        if pred.dim() == target.dim():
            if torch.numel(pred) != torch.numel(target):
                raise RuntimeError(
                    f"In {_get_func_signature(self.evaluate)}, when pred have same dimensions with target, they should have same element numbers. while target have "
                    f"element numbers:{torch.numel(target)}, pred have element numbers: {torch.numel(pred)}")

            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad.")
        else:
            raise RuntimeError(
                f"In {_get_func_signature(self.evaluate)}, when pred have "
                f"size:{pred.size()}, target should have size: {pred.size()} or "
                f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)
        if seq_len is not None and target.dim() > 1:
            for p, t, l in zip(pred.tolist(), target.tolist(),
                               seq_len.tolist()):
                l = int(l)
                self.confusion_matrix.add_pred_target(p[:l], t[:l])
        elif target.dim() > 1:  # 对于没有传入seq_len，但是又是高维的target，按全长输出
            for p, t in zip(pred.tolist(), target.tolist()):
                self.confusion_matrix.add_pred_target(p, t)
        else:
            self.confusion_matrix.add_pred_target(pred.tolist(),
                                                  target.tolist())

    def get_metric(self, reset=True):

        confusion = {'confusion_matrix': deepcopy(self.confusion_matrix)}
        if reset:
            self.confusion_matrix.clear()
        return confusion


class AccuracyMetric(MetricBase):

    def __init__(self, pred=None, target=None, seq_len=None):

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.total = 0
        self.acc_count = 0

    def evaluate(self, pred, target, seq_len=None):

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = None

        if pred.dim() == target.dim():
            if torch.numel(pred) != torch.numel(target):
                raise RuntimeError(
                    f"In {_get_func_signature(self.evaluate)}, when pred have same dimensions with target, they should have same element numbers. while target have "
                    f"element numbers:{torch.numel(target)}, pred have element numbers: {torch.numel(pred)}")

            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)
        if masks is not None:
            self.acc_count += torch.sum(torch.eq(pred, target).masked_fill(masks.eq(False), 0)).item()
            self.total += torch.sum(masks).item()
        else:
            self.acc_count += torch.sum(torch.eq(pred, target)).item()
            self.total += np.prod(list(pred.size()))

    def get_metric(self, reset=True):
        evaluate_result = {'acc': round(float(self.acc_count) / (self.total + 1e-12), 6)}
        if reset:
            self.acc_count = 0
            self.total = 0
        return evaluate_result


class ClassifyFPreRecMetric(MetricBase):

    def __init__(self, tag_vocab=None, pred=None, target=None, seq_len=None, ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):

        if tag_vocab:
            if not isinstance(tag_vocab, Vocabulary):
                raise TypeError("tag_vocab can only be fastNLP.Vocabulary, not {}.".format(type(tag_vocab)))
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.only_gross = only_gross

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.tag_vocab = tag_vocab

        self._tp, self._fp, self._fn = defaultdict(int), defaultdict(int), defaultdict(int)
        # tp: truth=T, classify=T; fp: truth=T, classify=F; fn: truth=F, classify=T

    def evaluate(self, pred, target, seq_len=None):

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = torch.ones_like(target).long().to(target.device)

        masks = masks.eq(1)

        if pred.dim() == target.dim():
            if torch.numel(pred) != torch.numel(target):
                raise RuntimeError(
                    f"In {_get_func_signature(self.evaluate)}, when pred have same dimensions with target, they should have same element numbers. while target have "
                    f"element numbers:{torch.numel(target)}, pred have element numbers: {torch.numel(pred)}")

            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)
        target = target.masked_select(masks)
        pred = pred.masked_select(masks)
        target_idxes = set(target.reshape(-1).tolist())
        for target_idx in target_idxes:
            self._tp[target_idx] += torch.sum((pred == target_idx).long().masked_fill(target != target_idx, 0)).item()
            self._fp[target_idx] += torch.sum((pred == target_idx).long().masked_fill(target == target_idx, 0)).item()
            self._fn[target_idx] += torch.sum((pred != target_idx).long().masked_fill(target != target_idx, 0)).item()

    def get_metric(self, reset=True):

        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._fn.keys())
            tags.update(set(self._fp.keys()))
            tags.update(set(self._tp.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                if self.tag_vocab is not None:
                    tag_name = self.tag_vocab.to_word(tag)
                else:
                    tag_name = int(tag)
                tp = self._tp[tag]
                fn = self._fn[tag]
                fp = self._fp[tag]
                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag_name)
                    pre_key = 'pre-{}'.format(tag_name)
                    rec_key = 'rec-{}'.format(tag_name)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(self.beta_square,
                                             sum(self._tp.values()),
                                             sum(self._fn.values()),
                                             sum(self._fp.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._tp = defaultdict(int)
            self._fp = defaultdict(int)
            self._fn = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result


def _bmes_tag_to_spans(tags, ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bmeso_tag_to_spans(tags, ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bmes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bioes_tag_to_spans(tags, ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bioes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bioes_tag, label = tag[:1], tag[2:]
        if bioes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bioes_tag in ('i', 'e') and prev_bioes_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bioes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bioes_tag = bioes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bio_tag_to_spans(tags, ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]


def _get_encoding_type_from_tag_vocab(tag_vocab: Union[Vocabulary, dict]) -> str:
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    bmes_tag_set = set('bmes')
    if tag_set == bmes_tag_set:
        return 'bmes'
    bio_tag_set = set('bio')
    if tag_set == bio_tag_set:
        return 'bio'
    bmeso_tag_set = set('bmeso')
    if tag_set == bmeso_tag_set:
        return 'bmeso'
    bioes_tag_set = set('bioes')
    if tag_set == bioes_tag_set:
        return 'bioes'
    raise RuntimeError("encoding_type cannot be inferred automatically. Only support "
                       "'bio', 'bmes', 'bmeso', 'bioes' type.")


def _check_tag_vocab_and_encoding_type(tag_vocab: Union[Vocabulary, dict], encoding_type: str):
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    tags = encoding_type
    for tag in tag_set:
        assert tag in tags, f"{tag} is not a valid tag in encoding type:{encoding_type}. Please check your " \
                            f"encoding_type."
        tags = tags.replace(tag, '')  # 删除该值
    if tags:  # 如果不为空，说明出现了未使用的tag
        warnings.warn(f"Tag:{tags} in encoding type:{encoding_type} is not presented in your Vocabulary. Check your "
                      "encoding_type.")


class SpanFPreRecMetric(MetricBase):

    def __init__(self, tag_vocab, pred=None, target=None, seq_len=None, encoding_type=None, ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):

        if not isinstance(tag_vocab, Vocabulary):
            raise TypeError("tag_vocab can only be fastNLP.Vocabulary, not {}.".format(type(tag_vocab)))
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

        if encoding_type:
            encoding_type = encoding_type.lower()
            _check_tag_vocab_and_encoding_type(tag_vocab, encoding_type)
            self.encoding_type = encoding_type
        else:
            self.encoding_type = _get_encoding_type_from_tag_vocab(tag_vocab)

        if self.encoding_type == 'bmes':
            self.tag_to_span_func = _bmes_tag_to_spans
        elif self.encoding_type == 'bio':
            self.tag_to_span_func = _bio_tag_to_spans
        elif self.encoding_type == 'bmeso':
            self.tag_to_span_func = _bmeso_tag_to_spans
        elif self.encoding_type == 'bioes':
            self.tag_to_span_func = _bioes_tag_to_spans
        else:
            raise ValueError("Only support 'bio', 'bmes', 'bmeso', 'bioes' type.")

        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.only_gross = only_gross

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.tag_vocab = tag_vocab

        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def evaluate(self, pred, target, seq_len):

        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            num_classes = pred.size(-1)
            pred = pred.argmax(dim=-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        batch_size = pred.size(0)
        pred = pred.tolist()
        target = target.tolist()
        for i in range(batch_size):
            pred_tags = pred[i][:int(seq_len[i])]
            gold_tags = target[i][:int(seq_len[i])]

            pred_str_tags = [self.tag_vocab.to_word(tag) for tag in pred_tags]
            gold_str_tags = [self.tag_vocab.to_word(tag) for tag in gold_tags]

            pred_spans = self.tag_to_span_func(pred_str_tags, ignore_labels=self.ignore_labels)
            gold_spans = self.tag_to_span_func(gold_str_tags, ignore_labels=self.ignore_labels)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

    def get_metric(self, reset=True):
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(self.beta_square,
                                             sum(self._true_positives.values()),
                                             sum(self._false_negatives.values()),
                                             sum(self._false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result


class TupleClassifyFPreRecMetric(MetricBase):

    def __init__(self):
        self.prediction_num = 0
        self.golden_num = 0
        self.correction_num = 0

    def evaluate(self, pred, target, seq_len=None):
        self.prediction_num += len(pred)
        self.golden_num += len(target)
        for item in pred:
            if item in target:
                self.correction_num += 1

    def get_metric(self, reset=True):
        if self.prediction_num != 0:
            precision = self.correction_num / self.prediction_num
        else:
            precision = 1.0
        if self.golden_num != 0:
            recall = self.correction_num / self.golden_num
        else:
            recall = 1.0
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return {'f': f1, 'pre': precision, 'rec': recall}


class EventFPreRecMetric(MetricBase):
    def __init__(self, trigger_vocabulary, argument_vocabulary):
        self.trigger_vocabulary = trigger_vocabulary
        self.argument_vocabulary = argument_vocabulary
        self.trigger_metric = SpanFPreRecMetric(tag_vocab=self.trigger_vocabulary)
        self.argument_metric = TupleClassifyFPreRecMetric()

    def get_metric(self, reset=True):
        trigger_eval = self.trigger_metric.get_metric(reset)
        argument_eval = self.argument_metric.get_metric(reset)

        return {
            'trigger_f': trigger_eval['f'],
            'trigger_pre': trigger_eval['pre'],
            'trigger_rec': trigger_eval['rec'],
            'argument_f': argument_eval['f'],
            'argument_pre': argument_eval['pre'],
            'argument_rec': argument_eval['rec'],
        }

    def evaluate(self, trigger_pred, trigger_target, trigger_seq_len, argument_pred, argument_target):
        self.trigger_metric.evaluate(trigger_pred, trigger_target, trigger_seq_len)
        self.argument_metric.evaluate(argument_pred, argument_target)


def _compute_f_pre_rec(beta_square, tp, fn, fp):
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec


def _prepare_metrics(metrics):
    _metrics = []
    if metrics:
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, type):
                    metric = metric()
                if isinstance(metric, MetricBase):
                    metric_name = metric.__class__.__name__
                    if not callable(metric.evaluate):
                        raise TypeError(f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}.")
                    if not callable(metric.get_metric):
                        raise TypeError(f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}.")
                    _metrics.append(metric)
                else:
                    raise TypeError(
                        f"The type of metric in metrics must be `fastNLP.MetricBase`, not `{type(metric)}`.")
        elif isinstance(metrics, MetricBase):
            _metrics = [metrics]
        else:
            raise TypeError(f"The type of metrics should be `list[fastNLP.MetricBase]` or `fastNLP.MetricBase`, "
                            f"got {type(metrics)}.")
    return _metrics


def _accuracy_topk(y_true, y_prob, k=1):
    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    y_true_tile = np.tile(np.expand_dims(y_true, axis=1), (1, k))
    y_match = np.any(y_pred_topk == y_true_tile, axis=-1)
    acc = np.sum(y_match) / y_match.shape[0]
    return acc


def _pred_topk(y_prob, k=1):
    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    x_axis_index = np.tile(
        np.arange(len(y_prob))[:, np.newaxis],
        (1, k))
    y_prob_topk = y_prob[x_axis_index, y_pred_topk]
    return y_pred_topk, y_prob_topk


class CMRC2018Metric(MetricBase):

    def __init__(self, answers=None, raw_chars=None, context_len=None, pred_start=None, pred_end=None):
        super().__init__()
        self._init_param_map(answers=answers, raw_chars=raw_chars, context_len=context_len, pred_start=pred_start,
                             pred_end=pred_end)
        self.em = 0
        self.total = 0
        self.f1 = 0

    def evaluate(self, answers, raw_chars, pred_start, pred_end, context_len=None):
        if pred_start.dim() > 1:
            batch_size, max_len = pred_start.size()
            context_mask = seq_len_to_mask(context_len, max_len=max_len).eq(False)
            pred_start.masked_fill_(context_mask, float('-inf'))
            pred_end.masked_fill_(context_mask, float('-inf'))
            max_pred_start, pred_start_index = pred_start.max(dim=-1, keepdim=True)  # batch_size,
            pred_start_mask = pred_start.eq(max_pred_start).cumsum(dim=-1).eq(0)  # 只能预测这之后的值
            pred_end.masked_fill_(pred_start_mask, float('-inf'))
            pred_end_index = pred_end.argmax(dim=-1) + 1
        else:
            pred_start_index = pred_start
            pred_end_index = pred_end + 1
        pred_ans = []
        for index, (start, end) in enumerate(zip(pred_start_index.flatten().tolist(), pred_end_index.tolist())):
            pred_ans.append(''.join(raw_chars[index][start:end]))
        for answer, pred_an in zip(answers, pred_ans):
            pred_an = pred_an.strip()
            self.f1 += _calc_cmrc2018_f1_score(answer, pred_an)
            self.total += 1
            self.em += _calc_cmrc2018_em_score(answer, pred_an)

    def get_metric(self, reset=True):
        eval_res = {'f1': round(self.f1 / self.total * 100, 2), 'em': round(self.em / self.total * 100, 2)}
        if reset:
            self.em = 0
            self.total = 0
            self.f1 = 0
        return eval_res


# split Chinese
def _cn_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = {'-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：', '？', '！', '“', '”', '；', '’', '《',
               '》', '……', '·', '、', '「', '」', '（', '）', '－', '～', '『', '』'}
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = list(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = list(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def _remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def _find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def _calc_cmrc2018_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = _cn_segmentation(ans, rm_punc=True)
        prediction_segs = _cn_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = _find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def _calc_cmrc2018_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _remove_punctuation(ans)
        prediction_ = _remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


class MultiLabelStrictAccuracyMetric(MetricBase):
    r"""
    多标签分类严格准确率
    """

    def __init__(self, pred=None, target=None, seq_len=None):

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.total = 0
        self.acc_count = 0

    def evaluate(self, pred, target, seq_len=None):
        batch_size, label_size = pred.size()
        self.total += batch_size
        for i in range(batch_size):
            if (pred[i] != target[i]).sum() == 0:
                self.acc_count += 1

    def get_metric(self, reset=True):
        evaluate_result = {'acc': round(float(self.acc_count) / (self.total + 1e-12), 6)}
        if reset:
            self.acc_count = 0
            self.total = 0
        return evaluate_result


class EventMetric(MetricBase):
    def __init__(self, trigger_vocabulary, argument_vocabulary):
        super().__init__()
        self.words_all = []
        self.triggers_all = []
        self.triggers_hat_all = []
        self.arguments_all = []
        self.arguments_hat_all = []
        self.trigger_vocabulary = trigger_vocabulary
        self.argument_vocabulary = argument_vocabulary

    def evaluate(self, words_2d, triggers_2d, trigger_hat_2d, arguments_2d, argument_hat_2d, argument_keys):
        self.words_all.extend(words_2d)
        self.triggers_all.extend(triggers_2d)
        self.triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
        self.arguments_all.extend(arguments_2d)

        if len(argument_keys) > 0:
            self.arguments_hat_all.extend(argument_hat_2d)
        else:
            batch_size = len(arguments_2d)
            argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
            self.arguments_hat_all.extend(argument_hat_2d)

    def get_metric(self, reset=True):
        def calc_metric(y_true, y_pred):
            num_proposed = len(y_pred)
            num_gold = len(y_true)

            y_true_set = set(y_true)
            num_correct = 0
            for item in y_pred:
                if item in y_true_set:
                    num_correct += 1

            print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

            if num_proposed != 0:
                precision = num_correct / num_proposed
            else:
                precision = 1.0

            if num_gold != 0:
                recall = num_correct / num_gold
            else:
                recall = 1.0

            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0

            return precision, recall, f1

        def find_triggers(labels):
            result = []
            labels = [label.split('-') for label in labels]

            for i in range(len(labels)):
                if labels[i][0] == 'B':
                    result.append([i, i + 1, labels[i][1]])

            for item in result:
                j = item[1]
                while j < len(labels):
                    if labels[j][0] == 'I':
                        j = j + 1
                        item[1] = j
                    else:
                        break

            return [tuple(item) for item in result]

        triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(
                zip(self.words_all, self.triggers_all, self.triggers_hat_all, self.arguments_all,
                    self.arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [self.trigger_vocabulary.to_word(hat) for hat in triggers_hat]

            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_true.append((i, t_type_str, a_start, a_end, a_type_idx))

            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_pred.append((i, t_type_str, a_start, a_end, a_type_idx))

        trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
        argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)

        if reset:
            self.words_all = []
            self.triggers_all = []
            self.triggers_hat_all = []
            self.arguments_all = []
            self.arguments_hat_all = []

        return {
            "trigger_f": trigger_f1,
            "trigger_pre": trigger_p,
            "trigger_rec": trigger_r,
            "argument_f": argument_f1,
            "argument_pre": argument_p,
            "argument_rec": argument_r,
        }


class FBetaMeasure(MetricBase):

    def __init__(self, beta: float = 1.0, average: str = None, labels: List[int] = None) -> None:
        average_options = {None, "micro", "macro", "weighted"}
        self._beta = beta
        self._average = average
        self._labels = labels

        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum: Union[None, torch.Tensor] = None
        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum: Union[None, torch.Tensor] = None

    def evaluate(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):

        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        device = gold_labels.device

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=predictions.device)
            self._true_sum = torch.zeros(num_classes, device=predictions.device)
            self._pred_sum = torch.zeros(num_classes, device=predictions.device)
            self._total_sum = torch.zeros(num_classes, device=predictions.device)

        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        gold_labels = gold_labels.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        pred_mask = predictions.sum(dim=-1) != 0
        argmax_predictions = predictions.max(dim=-1)[1].float()

        true_positives = (gold_labels == argmax_predictions) & mask & pred_mask
        true_positives_bins = gold_labels[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=device)
        else:
            true_positive_sum = torch.bincount(
                true_positives_bins.long(), minlength=num_classes
            ).float()

        pred_bins = argmax_predictions[mask & pred_mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes, device=device)

        gold_labels_bins = gold_labels[mask].long()
        if gold_labels.shape[0] != 0:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes, device=predictions.device)

        self._total_sum += mask.sum().to(torch.float)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum

    def get_metric(self, reset: bool = True):
        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        else:
            tp_sum = self._true_positive_sum
            pred_sum = self._pred_sum
            true_sum = self._true_sum

        if self._labels is not None:
            # Retain only selected labels and order them
            tp_sum = tp_sum[self._labels]
            pred_sum = pred_sum[self._labels]  # type: ignore
            true_sum = true_sum[self._labels]  # type: ignore

        if self._average == "micro":
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()  # type: ignore
            true_sum = true_sum.sum()  # type: ignore

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[tp_sum == 0] = 0.0

        if self._average == "macro":
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()
        elif self._average == "weighted":
            weights = true_sum
            weights_sum = true_sum.sum()  # type: ignore
            precision = _prf_divide((weights * precision).sum(), weights_sum)
            recall = _prf_divide((weights * recall).sum(), weights_sum)
            fscore = _prf_divide((weights * fscore).sum(), weights_sum)

        if reset:
            self.reset()

        if self._average is None:
            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "fscore": fscore.tolist(),
            }
        else:
            return {"precision": precision.item(), "recall": recall.item(), "fscore": fscore.item()}

    def reset(self) -> None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None
        self._total_sum = None

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = (
                self._total_sum - self._pred_sum - self._true_sum + self._true_positive_sum
            )
            return true_negative_sum


def _prf_divide(numerator, denominator):
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result


class FBetaMultiLabelMetric(FBetaMeasure):
    def __init__(
        self,
        beta: float = 1.0,
        average: str = None,
        labels: List[int] = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(beta, average, labels)
        self._threshold = threshold

    def evaluate(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=predictions.device)
            self._true_sum = torch.zeros(num_classes, device=predictions.device)
            self._pred_sum = torch.zeros(num_classes, device=predictions.device)
            self._total_sum = torch.zeros(num_classes, device=predictions.device)

        if mask is None:
            mask = torch.ones_like(gold_labels).bool()
        gold_labels = gold_labels.float()

        # If the prediction tensor is all zeros, the record is not classified to any of the labels.
        pred_mask = (predictions.sum(dim=-1) != 0).unsqueeze(-1)
        threshold_predictions = (predictions >= self._threshold).float()

        class_indices = (
            torch.arange(num_classes, device=predictions.device)
            .unsqueeze(0)
            .repeat(gold_labels.size(0), 1)
        )
        true_positives = (gold_labels * threshold_predictions).bool() & mask & pred_mask
        true_positives_bins = class_indices[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=predictions.device)
        else:
            true_positive_sum = torch.bincount(
                true_positives_bins.long(), minlength=num_classes
            ).float()

        pred_bins = class_indices[threshold_predictions.bool() & mask & pred_mask]
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes, device=predictions.device)

        gold_labels_bins = class_indices[gold_labels.bool() & mask]
        if gold_labels_bins.shape[0] != 0:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes, device=predictions.device)

        self._total_sum += mask.expand_as(gold_labels).sum().to(torch.float)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = (
                self._total_sum[0] / self._true_positive_sum.size(0)
                - self._pred_sum
                - self._true_sum
                + self._true_positive_sum
            )
            return true_negative_sum
