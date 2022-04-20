from torch import nn
import math
import torch
from cogie.models import BaseModule
from transformers import BertModel
import numpy as np
from tqdm import tqdm
from sklearn.metrics import *

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
    def __init__(self, config,trigger_max_span,argument_max_span,trigger_vocabulary,argument_vocabulary,  pos_emb_size,args_num,type_num,device,bert_model='bert-base-cased',multi_piece="average",schema_id=None):
        super(CasEE, self).__init__()
        self.trigger_max_span=trigger_max_span
        self.argument_max_span=argument_max_span
        self.trigger_vocabulary=trigger_vocabulary
        self.argument_vocabulary=argument_vocabulary
        self.schema_id=schema_id
        self.bert_model=bert_model
        self.bert =BertModel.from_pretrained(self.bert_model)
        self.multi_piece=multi_piece
        self.device=device

        config.args_num= args_num
        config.type_num = type_num
        config.hidden_size=768

        self.config = config
        self.args_num = config.args_num
        self.text_seq_len = config.seq_length

        self.type_cls = TypeCls(config)
        self.trigger_rec = TriggerRec(config, config.hidden_size)
        self.args_rec = ArgsRec(config, config.hidden_size, self.args_num, self.text_seq_len, pos_emb_size)
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.loss_0 = nn.BCELoss(reduction='none')
        self.loss_1 = nn.BCELoss(reduction='none')
        self.loss_2 = nn.BCELoss(reduction='none')

    def forward(self, tokens,  mask, head_indexes,type_id, type_vec, trigger_s_vec, trigger_e_vec, relative_pos, trigger_mask, args_s_vec, args_e_vec, args_mask,loss_function):
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


        output_emb,mask = self.plm(tokens, mask,head_indexes)
        batch_size = tokens.shape[0]


        p_type, type_emb = self.type_cls(output_emb, mask)
        p_type = p_type.pow(self.config.pow_0)
        type_loss = self.loss_0(p_type, type_vec)
        type_loss = torch.sum(type_loss)





        # event_flag=type_id!=-1
        # event_num=sum(event_flag).item()
        # tokens=tokens[event_flag]
        # mask=mask[event_flag]
        # head_indexes=head_indexes[event_flag]
        # type_id=type_id[event_flag]
        # type_vec=type_vec[event_flag]
        # trigger_s_vec=trigger_s_vec[event_flag]
        # trigger_e_vec=trigger_e_vec[event_flag]
        # relative_pos=relative_pos[event_flag]
        # trigger_mask=trigger_mask[event_flag]
        # args_s_vec=args_s_vec[event_flag]
        # args_e_vec=args_e_vec[event_flag]
        # args_mask=args_mask[event_flag]
        # output_emb=output_emb[event_flag]
        # trigger_loss=0
        # args_loss=0
        # if event_num>0:

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
        print("loss", loss.item(), "type_loss", type_loss.item(), " trigger_loss", trigger_loss.item(), "args_loss",
              args_loss.item())
        return loss, type_loss, trigger_loss, args_loss

    def plm(self, tokens,  mask,head_indexes):
        # assert tokens.size(0) == 1

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
        # batch_size = tokens.shape[0]
        # for i in range(batch_size):
        #     output_emb[i] = torch.index_select(output_emb[i], 0, head_indexes[i])
        #     mask[i] = torch.index_select(mask[i], 0, head_indexes[i])

        return output_emb,mask

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
        p_s = p_s.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        p_e = p_e.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        return p_s, p_e, type_soft_constrain

    def loss(self, batch, loss_function):
        token = torch.LongTensor(np.array(batch["tokens_id"])).to(self.device)
        mask = torch.LongTensor(np.array(batch["token_masks"])).to(self.device)
        head_indexes = torch.LongTensor(np.array(batch["head_indexes"])).to(self.device)
        d_t = torch.LongTensor(np.array(batch["type_id"])).to(self.device)
        t_v = torch.FloatTensor(np.array(batch["type_vec"])).to(self.device)
        t_s = torch.FloatTensor(np.array(batch["t_s"])).to(self.device)
        t_e = torch.FloatTensor(np.array(batch["t_e"])).to(self.device)
        r_pos = torch.LongTensor(np.array(batch["r_pos"])).to(self.device)
        t_m = torch.LongTensor(np.array(batch["t_m"])).to(self.device)
        a_s = torch.FloatTensor(np.array(batch["a_s"])).to(self.device)
        a_e = torch.FloatTensor(np.array(batch["a_e"])).to(self.device)
        a_m = torch.LongTensor(np.array(batch["a_m"])).to(self.device)
        loss=self.forward(token,mask,head_indexes, d_t, t_v, t_s, t_e, r_pos, t_m, a_s, a_e, a_m,loss_function)[0]
        return loss

    def evaluate(self, batch, metrics):
        if hasattr(self, "module"):
            model = self.module
        self.eval()
        idx=batch["data_ids"]
        typ_oracle=batch["type_id"]
        typ_truth=batch["type_vec"]
        token=batch["tokens_id"]
        mask=batch["token_masks"]
        t_index=batch["t_index"]
        r_p=batch["r_pos"]
        t_m=batch["t_m"]
        tri_truth=batch["triggers_truth"]
        args_truth=batch["args_truth"]
        head_indexes=batch["head_indexes"]
        content=batch["content"]

        typ_oracle = torch.LongTensor(np.array(typ_oracle)).to(self.device)
        typ_truth = torch.FloatTensor(np.array(typ_truth)).to(self.device)
        token = torch.LongTensor(np.array(token)).to(self.device)
        mask = torch.LongTensor(np.array(mask)).to(self.device)
        r_p = torch.LongTensor(np.array(r_p)).to(self.device)
        t_m = torch.LongTensor(np.array(t_m)).to(self.device)
        head_indexes = torch.LongTensor(np.array(head_indexes)).to(self.device)

        if t_index[0] is not None:

            tri_oracle = tri_truth[0][t_index[0]]
            type_pred, type_truth, trigger_pred_tuples, trigger_truth_tuples, args_pred_tuples, args_truth_tuples = self.predict_one(
                self, self.config, typ_truth, token,  mask, head_indexes,r_p, t_m, tri_truth, args_truth, self.schema_id, typ_oracle,
                tri_oracle)
            metrics.evaluate(idx, type_pred, type_truth, trigger_pred_tuples, trigger_truth_tuples, args_pred_tuples,
                             args_truth_tuples)
            result=self.evaluate_without_oracle(content,idx,self.config.seq_length, self.trigger_vocabulary.idx2word, self.argument_vocabulary.idx2word, self.schema_id,token,mask,head_indexes)
            metrics.results.append(result)
            # metrics.evaluate(idx,type_pred,type_truth,trigger_pred_tuples,trigger_truth_tuples,args_pred_tuples,args_truth_tuples)

    def evaluate_without_oracle(self, content,idx, seq_len, id_type, id_args, ty_args_id,token,mask,head_indexes):
        idx = idx[0]
        result = self.extract_all_items_without_oracle(self.device, idx, content, token, mask, seq_len,
                                                       self.config.threshold_0,
                                         self.config.threshold_1, self.config.threshold_2, self.config.threshold_3, self.config.threshold_4, id_type,
                                         id_args, ty_args_id,head_indexes)
        return result

    def extract_all_items_without_oracle(self,device, idx, content: str, token,  mask, seq_len, threshold_0,
                                         threshold_1, threshold_2, threshold_3, threshold_4, id_type: dict,
                                         id_args: dict, ty_args_id: dict,head_indexes):
        assert token.size(0) == 1
        content = content[0]
        result = {'id': idx, 'content': content}
        text_emb,mask = self.plm(token,  mask,head_indexes)

        args_id = {id_args[k]: k for k in id_args}
        args_len_dict = {args_id[k]: self.argument_max_span[k] for k in self.argument_max_span}

        p_type, type_emb = self.predict_type(text_emb, mask)
        type_pred = np.array(p_type > threshold_0, dtype=bool)
        type_pred = [i for i, t in enumerate(type_pred) if t]
        events_pred = []

        for type_pred_one in type_pred:
            type_rep = type_emb[type_pred_one, :]
            type_rep = type_rep.unsqueeze(0)
            p_s, p_e, text_rep_type = self.predict_trigger(type_rep, text_emb, mask)
            trigger_s = np.where(p_s > threshold_1)[0]
            trigger_e = np.where(p_e > threshold_2)[0]
            trigger_spans = []

            for i in trigger_s:
                es = trigger_e[trigger_e >= i]
                if len(es) > 0:
                    e = es[0]
                    if e - i + 1 <= 5:
                        trigger_spans.append((i, e))

            for k, span in enumerate(trigger_spans):
                rp = self.get_relative_pos(span[0], span[1], seq_len)
                rp = [p + seq_len for p in rp]
                tm = self.get_trigger_mask(span[0], span[1], seq_len)
                rp = torch.LongTensor(rp).to(device)
                tm = torch.LongTensor(tm).to(device)
                rp = rp.unsqueeze(0)
                tm = tm.unsqueeze(0)

                p_s, p_e, type_soft_constrain = self.predict_args(text_rep_type, rp, tm, mask, type_rep)

                p_s = np.transpose(p_s)
                p_e = np.transpose(p_e)

                type_name = id_type[type_pred_one]
                pred_event_one = {'type': type_name}
                pred_trigger = {'span': [int(span[0]) - 1, int(span[1]) + 1 - 1],
                                'word': content[int(span[0]) - 1:int(span[1]) + 1 - 1]}  # remove <CLS> token
                pred_event_one['trigger'] = pred_trigger
                pred_args = {}

                args_candidates = ty_args_id[type_pred_one]
                for i in args_candidates:
                    pred_args[id_args[i]] = []
                    args_s = np.where(p_s[i] > threshold_3)[0]
                    args_e = np.where(p_e[i] > threshold_4)[0]
                    for j in args_s:
                        es = args_e[args_e >= j]
                        if len(es) > 0:
                            e = es[0]
                            if e - j + 1 <= args_len_dict[i]:
                                pred_arg = {'span': [int(j) - 1, int(e) + 1 - 1],
                                            'word': content[int(j) - 1:int(e) + 1 - 1]}  # remove <CLS> token
                                pred_args[id_args[i]].append(pred_arg)

                pred_event_one['args'] = pred_args
                events_pred.append(pred_event_one)
        result['events'] = events_pred
        return result

    def get_relative_pos(self,start_idx, end_idx, length):
        '''
        return relative position
        [start_idx, end_idx]
        比如左闭右闭2,3
        那么下标从0开始，开始和结束位置都是0，前面是负数，后面是正数字
        0  1  2  3  4  5  6
        -2 -1 0  0  1  2  3
        '''
        pos = list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))
        return pos

    def get_trigger_mask(self,start_idx, end_idx, length):
        '''
        used to generate trigger mask, where the element of start/end postion is 1
        [000010100000]
        '''
        mask = [0] * length
        mask[start_idx] = 1
        mask[end_idx] = 1
        return mask




    def predict(self, batch):
        pass

    def predict_one(self,model, args, typ_truth, token, mask, head_indexes, r_p, t_m, tri_truth, args_truth, ty_args_id,
                    typ_oracle, tri_oracle):
        type_pred, trigger_pred, args_pred = self.extract_specific_item_with_oracle(model, typ_oracle, token, mask,
                                                                               head_indexes, r_p, t_m, args.args_num,
                                                                               args.threshold_0, args.threshold_1,
                                                                               args.threshold_2, args.threshold_3,
                                                                               args.threshold_4, ty_args_id)
        type_oracle = typ_oracle.item()
        type_truth = typ_truth.view(args.type_num).cpu().numpy().astype(int)
        trigger_truth, args_truth = tri_truth[0], args_truth[0]

        # used to save tuples, which is like:
        trigger_pred_tuples = []  # (type, tri_sta, tri_end), 3-tuple
        trigger_truth_tuples = []
        args_pred_tuples = []  # (type, tri_sta, tri_end, arg_sta, arg_end, arg_role), 6-tuple
        args_truth_tuples = []

        for trigger_pred_one in trigger_pred:
            typ = type_oracle
            sta = trigger_pred_one[0]
            end = trigger_pred_one[1]
            trigger_pred_tuples.append((typ, sta, end))

        for trigger_truth_one in trigger_truth:
            typ = type_oracle
            sta = trigger_truth_one[0]
            end = trigger_truth_one[1]
            trigger_truth_tuples.append((typ, sta, end))

        args_candidates = ty_args_id[type_oracle]  # type constrain
        for i in args_candidates:
            typ = type_oracle
            tri_sta = tri_oracle[0]
            tri_end = tri_oracle[1]
            arg_role = i
            for args_pred_one in args_pred[i]:
                arg_sta = args_pred_one[0]
                arg_end = args_pred_one[1]
                args_pred_tuples.append((typ, arg_sta, arg_end, arg_role))

            for args_truth_one in args_truth[i]:
                arg_sta = args_truth_one[0]
                arg_end = args_truth_one[1]
                args_truth_tuples.append((typ, arg_sta, arg_end, arg_role))

        return type_pred, type_truth, trigger_pred_tuples, trigger_truth_tuples, args_pred_tuples, args_truth_tuples

    def extract_specific_item_with_oracle(self, model, d_t, token, mask, head_indexes, rp, tm, args_num, threshold_0,
                                          threshold_1, threshold_2, threshold_3, threshold_4, ty_args_id):
        assert token.size(0) == 1
        data_type = d_t.item()
        text_emb,mask = self.plm(token, mask,head_indexes)
        batch_size = token.shape[0]
        # for i in range(batch_size):
        #     text_emb [i] = torch.index_select(text_emb [i], 0, head_indexes[i])
        #     mask[i] = torch.index_select(mask[i], 0, head_indexes[i])

        # predict event type
        p_type, type_emb = model.predict_type(text_emb, mask)
        type_pred = np.array(p_type > threshold_0, dtype=int)
        type_rep = type_emb[d_t, :]

        # predict event trigger
        p_s, p_e, text_rep_type = model.predict_trigger(type_rep, text_emb, mask)
        trigger_s = np.where(p_s > threshold_1)[0]
        trigger_e = np.where(p_e > threshold_2)[0]
        trigger_spans = []
        for i in trigger_s:
            es = trigger_e[trigger_e >= i]
            if len(es) > 0:
                e = es[0]
                trigger_spans.append((i, e))

        # predict event argument
        p_s, p_e, type_soft_constrain = model.predict_args(text_rep_type, rp, tm, mask, type_rep)
        p_s = np.transpose(p_s)
        p_e = np.transpose(p_e)
        args_spans = {i: [] for i in range(args_num)}
        for i in ty_args_id[data_type]:
            args_s = np.where(p_s[i] > threshold_3)[0]
            args_e = np.where(p_e[i] > threshold_4)[0]
            for j in args_s:
                es = args_e[args_e >= j]
                if len(es) > 0:
                    e = es[0]
                    args_spans[i].append((j, e))
        return type_pred, trigger_spans, args_spans
