from cogie.models import BaseModule
import torch
import torch.nn as nn
from transformers import  AutoModel
from cogie.modules.encoder import LSTM
import torch.nn.functional as F
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
# from cogie.modules.decoder import NodeBuilder
# from cogie.modules.decoder import EdgeBuilder
class Bert4FnJoint(BaseModule):
    def __init__(self,
                 node_types_vocabulary=None,
                 node_attrs_vocabulary=None,
                 p2p_edges_vocabulary=None,
                 p2r_edges_vocabulary=None,
                 bilstm_hidden_embedding_dim=200,
                 lexical_dropout=0.5,
                 lstm_dropout=0.4,
                 max_span_width=15,
                 feature_size=20,
                 embed_mode='bert-base-cased',
                 device=torch.device("cuda")
                 ):
        super().__init__()
        self.node_types_vocabulary = node_types_vocabulary
        self.node_attrs_vocabulary = node_attrs_vocabulary
        self.p2p_edges_vocabulary = p2p_edges_vocabulary
        self.p2r_edges_vocabulary = p2r_edges_vocabulary
        self.bilstm_hidden_embedding_dim=bilstm_hidden_embedding_dim
        self.lexical_dropout=lexical_dropout
        self.lstm_dropout=lstm_dropout
        self.embed_mode=embed_mode
        self.device=device
        self.max_span_width=max_span_width
        self.feature_size=feature_size

        if self.embed_mode == 'bert-base-cased':
            self.bert = AutoModel.from_pretrained("bert-base-cased")
            self.bert_hidden_embedding_dim = 768
        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        self.bilstm=LSTM(input_size=self.bert_hidden_embedding_dim,
                         hidden_size=self.bilstm_hidden_embedding_dim,
                         dropout=self.lstm_dropout,
                         bidirectional=True,
                         num_layers=6)
        self._endpoint_span_extractor = EndpointSpanExtractor(
            self.bilstm_hidden_embedding_dim,
            combination="x,y",
            num_width_embeddings=self.max_span_width,
            span_width_embedding_dim=self.feature_size,
            bucket_widths=False,
        )
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim=self.bert_hidden_embedding_dim
        )

        # self._node_builder = NodeBuilder.from_params(vocab=vocab, params=modules.pop("node"))
        # self._edge_builder = EdgeBuilder.from_params(vocab=vocab, params=modules.pop("edge"))




    def forward(self, tokens_x,masks,head_indexes,spans,raw_words_len):
        batch_size=tokens_x.shape[0]
        # token_embedding(BERT)
        text_embeddings = self.bert(input_ids=tokens_x,attention_mask=masks)["last_hidden_state"]
        # word_embedding(token_average)#暂时先用首词代替
        for i in range(batch_size):
            text_embeddings[i] = torch.index_select(text_embeddings[i], 0, head_indexes[i])
        text_embeddings=text_embeddings[:,:max(raw_words_len),:]
        text_embeddings = self._lexical_dropout(text_embeddings)
        # word_embedding(BILSTM)
        contextualized_embeddings=self.bilstm(text_embeddings)

        # span_representation
        span_mask = (spans[:, :, 0] >= 0).bool()
        spans = F.relu(spans.float()).long()
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings.contiguous(), spans.contiguous())
        attended_span_embeddings = self._attentive_span_extractor(text_embeddings.contiguous(), spans.contiguous())
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)


        #Node Building+ Edge Building

        output_nodes = {'loss': 0}
        output_edges = {'loss': 0}


        # output_nodes = self._node_builder(
        #     spans, span_mask, span_embeddings,
        #     node_type_labels=node_type_labels, node_attr_labels=node_attr_labels)
        #
        #
        # output_edges = self._edge_builder(
        #     spans, span_mask, span_embeddings, raw_words_len, output_nodes,
        #     self._ontology.simple_lu_frame_map, self._ontology.frame_fe_map,
        #     p2p_edge_labels, p2r_edge_labels, metadata)
        #
        # output_dict = dict(node=output_nodes, edge=output_edges)
        # output_dict['loss'] = output_nodes['loss'] +output_edges['loss']
        return 0


    def loss(self, batch, loss_function):
        batch_size = len(batch["raw_words_len"])
        raw_words_len = batch["raw_words_len"]
        spans_len=batch["n_spans"]
        max_spans_len=max(spans_len)
        tokens_x=torch.LongTensor(batch["tokens_x"]).to(self.device)
        masks=torch.LongTensor(batch["token_masks"]).to(self.device)
        head_indexes=torch.LongTensor(batch["head_indexes"]).to(self.device)
        for i in range(batch_size):
            batch["spans"][i]=batch["spans"][i]+(max_spans_len-spans_len[i])*[(-1,-1)]
        spans= torch.LongTensor(batch["spans"]).to(self.device)
        node_type_labels_list=batch["node_type_labels_list"]
        node_attr_labels_list=batch["node_attr_labels_list"]
        node_valid_attrs_list=batch["node_valid_attrs_list"]
        valid_p2r_edges_list=batch["valid_p2r_edges_list"]
        p2p_edge_labels_and_indices=batch["p2p_edge_labels_and_indices"]
        p2r_edge_labels_and_indices=batch["p2r_edge_labels_and_indices"]
        pred=self.forward(tokens_x,masks,head_indexes,spans,raw_words_len)
        # text = batch[0]
        # ner_label = batch[1].to(self.device)
        # re_label = batch[2].to(self.device)
        # mask = batch[3].to(self.device)
        # ner_pred, re_pred = self.forward(text, mask)
        # loss = loss_function(ner_pred, ner_label, re_pred, re_label)
        # return loss

    def evaluate(self, batch, metrics):
        pass
        # ner_label = batch[1].to(self.device)
        # re_label = batch[2].to(self.device)
        # ner_pred, re_pred = self.predict(batch)
        # metrics.evaluate(ner_pred=ner_pred, re_pred=re_pred, ner_label=ner_label, re_label=re_label)

    def predict(self, batch):
        pass
        # text = batch[0]
        # mask = batch[3].to(self.device)
        # ner_pred, re_pred = self.forward(text, mask)
        # return ner_pred, re_pred