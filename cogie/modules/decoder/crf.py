"""
@Author: jinzhuan
@File: crf.py
@Desc: 
"""
from typing import Union
import torch
from torch import nn
from cogie.core.metrics import _get_encoding_type_from_tag_vocab, _check_tag_vocab_and_encoding_type
from cogie.utils import Vocabulary


def allowed_transitions(tag_vocab: Union[Vocabulary, dict], encoding_type=None, include_start_end=False):
    if encoding_type is None:
        encoding_type = _get_encoding_type_from_tag_vocab(tag_vocab)
    else:
        encoding_type = encoding_type.lower()
        _check_tag_vocab_and_encoding_type(tag_vocab, encoding_type)

    pad_token = '<pad>'
    unk_token = '<unk>'

    if isinstance(tag_vocab, Vocabulary):
        id_label_lst = list(tag_vocab.idx2word.items())
        pad_token = tag_vocab.padding
        unk_token = tag_vocab.unknown
    else:
        id_label_lst = list(tag_vocab.items())

    num_tags = len(tag_vocab)
    start_idx = num_tags
    end_idx = num_tags + 1
    allowed_trans = []
    if include_start_end:
        id_label_lst += [(start_idx, 'start'), (end_idx, 'end')]

    def split_tag_label(from_label):
        from_label = from_label.lower()
        if from_label in ['start', 'end']:
            from_tag = from_label
            from_label = ''
        else:
            from_tag = from_label[:1]
            from_label = from_label[2:]
        return from_tag, from_label

    for from_id, from_label in id_label_lst:
        if from_label in [pad_token, unk_token]:
            continue
        from_tag, from_label = split_tag_label(from_label)
        for to_id, to_label in id_label_lst:
            if to_label in [pad_token, unk_token]:
                continue
            to_tag, to_label = split_tag_label(to_label)
            if _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
                allowed_trans.append((from_id, to_id))
    return allowed_trans


def _is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
    if to_tag == 'start' or from_tag == 'end':
        return False
    encoding_type = encoding_type.lower()
    if encoding_type == 'bio':
        if from_tag == 'start':
            return to_tag in ('b', 'o')
        elif from_tag in ['b', 'i']:
            return any([to_tag in ['end', 'b', 'o'], to_tag == 'i' and from_label == to_label])
        elif from_tag == 'o':
            return to_tag in ['end', 'b', 'o']
        else:
            raise ValueError("Unexpect tag {}. Expect only 'B', 'I', 'O'.".format(from_tag))

    elif encoding_type == 'bmes':
        if from_tag == 'start':
            return to_tag in ['b', 's']
        elif from_tag == 'b':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag == 'm':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag in ['e', 's']:
            return to_tag in ['b', 's', 'end']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'M', 'E', 'S'.".format(from_tag))
    elif encoding_type == 'bmeso':
        if from_tag == 'start':
            return to_tag in ['b', 's', 'o']
        elif from_tag == 'b':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag == 'm':
            return to_tag in ['m', 'e'] and from_label == to_label
        elif from_tag in ['e', 's', 'o']:
            return to_tag in ['b', 's', 'end', 'o']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'M', 'E', 'S', 'O'.".format(from_tag))
    elif encoding_type == 'bioes':
        if from_tag == 'start':
            return to_tag in ['b', 's', 'o']
        elif from_tag == 'b':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag == 'i':
            return to_tag in ['i', 'e'] and from_label == to_label
        elif from_tag in ['e', 's', 'o']:
            return to_tag in ['b', 's', 'end', 'o']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'I', 'E', 'S', 'O'.".format(from_tag))
    else:
        raise ValueError("Only support BIO, BMES, BMESO, BIOES encoding type, got {}.".format(encoding_type))


class ConditionalRandomField(nn.Module):

    def __init__(self, num_tags, include_start_end_trans=False, allowed_transitions=None):
        super(ConditionalRandomField, self).__init__()

        self.include_start_end_trans = include_start_end_trans
        self.num_tags = num_tags

        # the meaning of entry in this matrix is (from_tag_id, to_tag_id) score
        self.trans_m = nn.Parameter(torch.randn(num_tags, num_tags))
        if self.include_start_end_trans:
            self.start_scores = nn.Parameter(torch.randn(num_tags))
            self.end_scores = nn.Parameter(torch.randn(num_tags))

        if allowed_transitions is None:
            constrain = torch.zeros(num_tags + 2, num_tags + 2)
        else:
            constrain = torch.full((num_tags + 2, num_tags + 2), fill_value=-10000.0, dtype=torch.float)
            for from_tag_id, to_tag_id in allowed_transitions:
                constrain[from_tag_id, to_tag_id] = 0
        self._constrain = nn.Parameter(constrain, requires_grad=False)

    def _normalizer_likelihood(self, logits, mask):
        seq_len, batch_size, n_tags = logits.size()
        alpha = logits[0]
        if self.include_start_end_trans:
            alpha = alpha + self.start_scores.view(1, -1)

        flip_mask = mask.eq(False)

        for i in range(1, seq_len):
            emit_score = logits[i].view(batch_size, 1, n_tags)
            trans_score = self.trans_m.view(1, n_tags, n_tags)
            tmp = alpha.view(batch_size, n_tags, 1) + emit_score + trans_score
            alpha = torch.logsumexp(tmp, 1).masked_fill(flip_mask[i].view(batch_size, 1), 0) + \
                    alpha.masked_fill(mask[i].eq(True).view(batch_size, 1), 0)

        if self.include_start_end_trans:
            alpha = alpha + self.end_scores.view(1, -1)

        return torch.logsumexp(alpha, 1)

    def _gold_score(self, logits, tags, mask):
        seq_len, batch_size, _ = logits.size()
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)

        # trans_socre [L-1, B]
        mask = mask.eq(True)
        flip_mask = mask.eq(False)
        trans_score = self.trans_m[tags[:seq_len - 1], tags[1:]].masked_fill(flip_mask[1:, :], 0)
        # emit_score [L, B]
        emit_score = logits[seq_idx.view(-1, 1), batch_idx.view(1, -1), tags].masked_fill(flip_mask, 0)
        # score [L-1, B]
        score = trans_score + emit_score[:seq_len - 1, :]
        score = score.sum(0) + emit_score[-1].masked_fill(flip_mask[-1], 0)
        if self.include_start_end_trans:
            st_scores = self.start_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[0]]
            last_idx = mask.long().sum(0) - 1
            ed_scores = self.end_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[last_idx, batch_idx]]
            score = score + st_scores + ed_scores
        # return [B,]
        return score

    def forward(self, feats, tags, mask):
        feats = feats.transpose(0, 1)
        tags = tags.transpose(0, 1).long()
        mask = mask.transpose(0, 1).float()
        all_path_score = self._normalizer_likelihood(feats, mask)
        gold_path_score = self._gold_score(feats, tags, mask)

        return all_path_score - gold_path_score

    def viterbi_decode(self, logits, mask, unpad=False):
        batch_size, seq_len, n_tags = logits.size()
        logits = logits.transpose(0, 1).data  # L, B, H
        mask = mask.transpose(0, 1).data.eq(True)  # L, B
        flip_mask = mask.eq(False)

        # dp
        vpath = logits.new_zeros((seq_len, batch_size, n_tags), dtype=torch.long)
        vscore = logits[0]
        transitions = self._constrain.data.clone()
        transitions[:n_tags, :n_tags] += self.trans_m.data
        if self.include_start_end_trans:
            transitions[n_tags, :n_tags] += self.start_scores.data
            transitions[:n_tags, n_tags + 1] += self.end_scores.data

        vscore += transitions[n_tags, :n_tags]
        trans_score = transitions[:n_tags, :n_tags].view(1, n_tags, n_tags).data
        for i in range(1, seq_len):
            prev_score = vscore.view(batch_size, n_tags, 1)
            cur_score = logits[i].view(batch_size, 1, n_tags) + trans_score
            score = prev_score + cur_score.masked_fill(flip_mask[i].view(batch_size, 1, 1), 0)
            best_score, best_dst = score.max(1)
            vpath[i] = best_dst
            vscore = best_score

        if self.include_start_end_trans:
            vscore += transitions[:n_tags, n_tags + 1].view(1, -1)

        # backtrace
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)
        lens = (mask.long().sum(0) - 1)
        # idxes [L, B], batched idx from seq_len-1 to 0
        idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % seq_len

        ans = logits.new_empty((seq_len, batch_size), dtype=torch.long)
        ans_score, last_tags = vscore.max(1)
        ans[idxes[0], batch_idx] = last_tags
        for i in range(seq_len - 1):
            last_tags = vpath[idxes[i], batch_idx, last_tags]
            ans[idxes[i + 1], batch_idx] = last_tags
        ans = ans.transpose(0, 1)
        if unpad:
            paths = []
            for idx, seq_len in enumerate(lens):
                paths.append(ans[idx, :seq_len + 1].tolist())
        else:
            paths = ans
        return paths, ans_score
