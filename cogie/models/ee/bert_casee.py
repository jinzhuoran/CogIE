from torch import nn
import math
import torch
from cogie.models import BaseModule
from transformers import BertModel


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.gamma_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

        nn.init.zeros_(self.gamma_dense.weight)
        nn.init.zeros_(self.beta_dense.weight)

    def forward(self, x, condition):
        '''

        :param x: [b, t, e]
        :param condition: [b, e]
        :return:
        '''
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        condition = condition.unsqueeze(1).expand_as(x)
        gamma = self.gamma_dense(condition) + self.gamma
        beta = self.beta_dense(condition) + self.beta
        x = gamma * (x - mean) / (std + self.eps) + beta
        return x


class AdaptiveAdditionPredictor(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0):
        super(AdaptiveAdditionPredictor, self).__init__()
        self.v = nn.Linear(hidden_size * 4, 1)
        self.hidden = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, context, mask):
        '''
        :param query: [c, e]
        :param context: [b, t, e]
        :param mask: [b, t], 0 if masked
        :return: [b, e]
        '''

        context_ = context.unsqueeze(1).expand(context.size(0), query.size(0), context.size(1), context.size(2))  # [b, c, t, e]
        query_ = query.unsqueeze(0).unsqueeze(2).expand_as(context_)  # [b, c, t, e]

        scores = self.v(torch.tanh(self.hidden(torch.cat([query_, context_, torch.abs(query_ - context_), query_ * context_], dim=-1))))  # [b, c, t, 1]
        scores = self.dropout(scores)
        mask = (mask < 1).unsqueeze(1).unsqueeze(3).expand_as(scores)  # [b, c, t, 1]
        scores = scores.masked_fill_(mask, -1e10)
        scores = scores.transpose(-1, -2)  # [b, c, 1, t]
        scores = torch.softmax(scores, dim=-1)  # [b, c, 1, t]
        g = torch.matmul(scores, context_).squeeze(2)  # [b, c, e]
        query = query.unsqueeze(0).expand_as(g)  # [b, c, e]

        pred = self.v(torch.tanh(self.hidden(torch.cat([query, g, torch.abs(query - g), query * g], dim=-1)))).squeeze(-1)  # [b, c]
        return pred


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size  x seq_length]
            mask is 0 if it is masked

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2) for l, x in zip(self.linear_layers, (query, key, value))]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        mask = mask. \
            unsqueeze(1). \
            repeat(1, seq_length, 1). \
            unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output



class TypeCls(nn.Module):
    def __init__(self, config):
        super(TypeCls, self).__init__()
        self.type_emb = nn.Embedding(config.type_num, config.hidden_size)
        self.register_buffer('type_indices', torch.arange(0, config.type_num, 1).long())
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.config = config
        self.Predictor = AdaptiveAdditionPredictor(config.hidden_size, dropout_rate=config.decoder_dropout)

    def forward(self, text_rep, mask):
        type_emb = self.type_emb(self.type_indices)
        pred = self.Predictor(type_emb, text_rep, mask)  # [b, c]
        p_type = torch.sigmoid(pred)
        return p_type, type_emb


class TriggerRec(nn.Module):
    def __init__(self, config, hidden_size):
        super(TriggerRec, self).__init__()
        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)

        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.head_cls = nn.Linear(hidden_size, 1, bias=True)
        self.tail_cls = nn.Linear(hidden_size, 1, bias=True)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.config = config

    def forward(self, query_emb, text_emb, mask):
        '''

        :param query_emb: [b, e]
        :param text_emb: [b, t, e]
        :param mask: 0 if masked
        :return: [b, t, 1], [], []
        '''

        h_cln = self.ConditionIntegrator(text_emb, query_emb)

        h_cln = self.dropout(h_cln)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        inp = self.layer_norm(h_sa + h_cln)
        inp = gelu(self.hidden(inp))
        inp = self.dropout(inp)
        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, 1]
        p_e = torch.sigmoid(self.tail_cls(inp))  # [b, t, 1]
        return p_s, p_e, h_cln


class ArgsRec(nn.Module):
    def __init__(self, config, hidden_size, num_labels, seq_len, pos_emb_size):
        super(ArgsRec, self).__init__()
        self.relative_pos_embed = nn.Embedding(seq_len * 2, pos_emb_size)
        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)
        self.hidden = nn.Linear(hidden_size + pos_emb_size, hidden_size)

        self.head_cls = nn.Linear(hidden_size, num_labels, bias=True)
        self.tail_cls = nn.Linear(hidden_size, num_labels, bias=True)

        self.gate_hidden = nn.Linear(hidden_size, hidden_size)
        self.gate_linear = nn.Linear(hidden_size, num_labels)

        self.seq_len = seq_len
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.config = config

    def forward(self, text_emb, relative_pos, trigger_mask, mask, type_emb):
        '''
        :param query_emb: [b, 4, e]
        :param text_emb: [b, t, e]
        :param relative_pos: [b, t, e]
        :param trigger_mask: [b, t]
        :param mask:
        :param type_emb: [b, e]
        :return:  [b, t, a], []
        '''
        trigger_emb = torch.bmm(trigger_mask.unsqueeze(1).float(), text_emb).squeeze(1)  # [b, e]
        trigger_emb = trigger_emb / 2

        h_cln = self.ConditionIntegrator(text_emb, trigger_emb)
        h_cln = self.dropout(h_cln)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        h_sa = self.layer_norm(h_sa + h_cln)

        rp_emb = self.relative_pos_embed(relative_pos)
        rp_emb = self.dropout(rp_emb)

        inp = torch.cat([h_sa, rp_emb], dim=-1)

        inp = gelu(self.hidden(inp))
        inp = self.dropout(inp)

        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, l]
        p_e = torch.sigmoid(self.tail_cls(inp))

        type_soft_constrain = torch.sigmoid(self.gate_linear(type_emb))  # [b, l]
        type_soft_constrain = type_soft_constrain.unsqueeze(1).expand_as(p_s)
        p_s = p_s * type_soft_constrain
        p_e = p_e * type_soft_constrain

        return p_s, p_e, type_soft_constrain


class CasEE(BaseModule):
    def __init__(self, config,  pos_emb_size,args_num,type_num,device,bert_model='bert-base-cased',multi_piece="average"):
        super(CasEE, self).__init__()
        self.config=config
        self.bert_model=bert_model
        self.bert =BertModel.from_pretrained(self.bert_model)
        self.multi_piece=multi_piece
        self.device=device

        config.args_num= args_num
        config.type_num = type_num
        config.hidden_size=768

        self.text_seq_len = config.seq_length

        self.type_cls = TypeCls(config)
        self.trigger_rec = TriggerRec(config, config.hidden_size)
        self.args_rec = ArgsRec(config, config.hidden_size, self.config.args_num, self.text_seq_len, pos_emb_size)
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.loss_0 = nn.BCELoss(reduction='none')
        self.loss_1 = nn.BCELoss(reduction='none')
        self.loss_2 = nn.BCELoss(reduction='none')

    def forward(self, tokens,  mask, head_indexes,type_id, type_vec, trigger_s_vec, trigger_e_vec, relative_pos, trigger_mask, args_s_vec, args_e_vec, args_mask):
        '''

        :param tokens: [b, t]
        :param mask: [b, t], 0 if masked
        :param trigger_s: [b, t]
        :param trigger_e: [b, t]
        :param relative_pos:
        :param trigger_mask: [0000011000000]
        :param args_s: [b, l, t]
        :param args_e: [b, l, t]
        :param args_m: [b, k]
        :return:
        '''

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )
        output_emb = outputs[0]

        batch_size =tokens.shape[0]

        for i in range(batch_size):
            output_emb[i] = torch.index_select(output_emb[i], 0,head_indexes[i])


        p_type, type_emb = self.type_cls(output_emb, mask)
        p_type = p_type.pow(self.config.pow_0)
        type_loss = self.loss_0(p_type, type_vec)
        type_loss = torch.sum(type_loss)

        type_rep = type_emb[type_id, :]
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, output_emb, mask)
        p_s = p_s.pow(self.config.pow_1)
        p_e = p_e.pow(self.config.pow_1)
        p_s = p_s.squeeze(-1)
        p_e = p_e.squeeze(-1)
        trigger_loss_s = self.loss_1(p_s, trigger_s_vec)
        trigger_loss_e = self.loss_1(p_e, trigger_e_vec)
        mask_t = mask.float()  # [b, t]
        trigger_loss_s = torch.sum(trigger_loss_s.mul(mask_t))
        trigger_loss_e = torch.sum(trigger_loss_e.mul(mask_t))

        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        p_s = p_s.pow(self.config.pow_2)
        p_e = p_e.pow(self.config.pow_2)
        args_loss_s = self.loss_2(p_s, args_s_vec.transpose(1, 2))  # [b, t, l]
        args_loss_e = self.loss_2(p_e, args_e_vec.transpose(1, 2))
        mask_a = mask.unsqueeze(-1).expand_as(args_loss_s).float()  # [b, t, l]
        args_loss_s = torch.sum(args_loss_s.mul(mask_a))
        args_loss_e = torch.sum(args_loss_e.mul(mask_a))

        trigger_loss = trigger_loss_s + trigger_loss_e
        args_loss = args_loss_s + args_loss_e

        type_loss = self.config.w1 * type_loss
        trigger_loss = self.config.w2 * trigger_loss
        args_loss = self.config.w3 * args_loss
        loss = type_loss + trigger_loss + args_loss
        return loss, type_loss, trigger_loss, args_loss


    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
         """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]

        if self.use_extra_bert:
            extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)

        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = self.token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets)
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':# average all pieces for multi-piece words
            idxs, masks, token_num, token_len = self.token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            masks = bert_outputs.new(masks).unsqueeze(-1)
            bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
            bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
            bert_outputs = bert_outputs.sum(2)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'.format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs

    def token_lens_to_idxs(self,token_lens):
        """Map token lengths to a word piece index matrix (for torch.gather) and a
        mask tensor.
        For example (only show a sequence instead of a batch):

        token lengths: [1,1,1,3,1]
        =>
        indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
        masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

        Next, we use torch.gather() to select vectors of word pieces for each token,
        and average them as follows (incomplete code):

        outputs = torch.gather(bert_outputs, 1, indices) * masks
        outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
        outputs = bert_outputs.sum(2)

        :param token_lens (list): token lengths.
        :return: a index matrix and a mask tensor.
        """
        max_token_num = max([len(x) for x in token_lens])
        max_token_len = max([max(x) for x in token_lens])
        idxs, masks = [], []
        for seq_token_lens in token_lens:
            seq_idxs, seq_masks = [], []
            offset = 0
            for token_len in seq_token_lens:
                seq_idxs.extend([i + offset for i in range(token_len)]
                                + [-1] * (max_token_len - token_len))
                seq_masks.extend([1.0 / token_len] * token_len
                                 + [0.0] * (max_token_len - token_len))
                offset += token_len
            seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
            seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
            idxs.append(seq_idxs)
            masks.append(seq_masks)
        return idxs, masks, max_token_num, max_token_len

    def token_lens_to_offsets(self,token_lens):
        """Map token lengths to first word piece indices, used by the sentence
        encoder.
        :param token_lens (list): token lengths (word piece numbers)
        :return (list): first word piece indices (offsets)
        """
        max_token_num = max([len(x) for x in token_lens])
        offsets = []
        for seq_token_lens in token_lens:
            seq_offsets = [0]
            for l in seq_token_lens[:-1]:
                seq_offsets.append(seq_offsets[-1] + l)
            offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
        return offsets

    def plm(self, tokens,  mask):
        assert tokens.size(0) == 1

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )
        output_emb = outputs[0]
        return output_emb

    def predict_type(self, text_emb, mask):
        assert text_emb.size(0) == 1
        p_type, type_emb = self.type_cls(text_emb, mask)
        p_type = p_type.view(self.config.type_num).data.cpu().numpy()
        return p_type, type_emb

    def predict_trigger(self, type_rep, text_emb, mask):
        assert text_emb.size(0) == 1
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, text_emb, mask)
        p_s = p_s.squeeze(-1)  # [b, t]
        p_e = p_e.squeeze(-1)
        mask = mask.float()  # [1, t]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.text_seq_len).data.cpu().numpy()  # [b, t]
        p_e = p_e.view(self.text_seq_len).data.cpu().numpy()
        return p_s, p_e, text_rep_type

    def predict_args(self, text_rep_type, relative_pos, trigger_mask, mask, type_rep):
        assert text_rep_type.size(0) == 1
        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        mask = mask.unsqueeze(-1).expand_as(p_s).float()  # [b, t, l]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.text_seq_len, self.config.args_num).data.cpu().numpy()
        p_e = p_e.view(self.text_seq_len, self.cnofig.args_num).data.cpu().numpy()
        return p_s, p_e, type_soft_constrain

    def loss(self, batch, loss_function):
        token = torch.LongTensor(batch["tokens_x"]).to(self.device)
        mask = torch.LongTensor(batch["token_masks"]).to(self.device)
        head_indexes = torch.LongTensor(batch["head_indexes"]).to(self.device)
        d_t = torch.LongTensor(batch["data_type_id"]).to(self.device)
        t_v = torch.FloatTensor(batch["type_vec"]).to(self.device)
        t_s = torch.FloatTensor(batch["t_s"]).to(self.device)
        t_e = torch.FloatTensor(batch["t_e"]).to(self.device)
        r_pos = torch.LongTensor(batch["r_pos"]).to(self.device)
        t_m = torch.LongTensor(batch["t_m"]).to(self.device)
        a_s = torch.FloatTensor(batch["a_s"]).to(self.device)
        a_e = torch.FloatTensor(batch["a_e"]).to(self.device)
        a_m = torch.LongTensor(batch["a_m"]).to(self.device)
        loss=self.forward(token,mask,head_indexes, d_t, t_v, t_s, t_e, r_pos, t_m, a_s, a_e, a_m)[0]
        return loss

    def evaluate(self, batch, metrics):
        pass

    def predict(self, batch):
        pass
