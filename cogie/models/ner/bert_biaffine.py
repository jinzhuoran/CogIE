from torch import nn
from transformers import AutoModel

from cogie.modules import Biaffine, MLP


class BertBiaffine(BaseModule):
    def __init__(
            self,
            label_size,
            sequence_length,
            embedding_size=768,
            hidden_dropout_prob=0.1,
            bert_model='bert-base-cased',
            device=torch.device("cuda"),
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.label_size = label_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.device = device
        self.sequence_length = sequence_length
        self.bert = AutoModel.from_pretrained(bert_model)
        self.start_layer = MLP(n_in=self.embedding_size, n_out=self.sequence_length, dropout=self.hidden_dropout_prob)
        self.end_layer = MLP(n_in=self.embedding_size, n_out=self.sequence_length, dropout=self.hidden_dropout_prob)
        self.biaffine = Biaffine(n_in=self.sequence_length, n_out=self.label_size, bias_x=True, bias_y=True)

    def forward(
            self,
            batch=None,
    ):
        input_ids, attention_mask, segment_ids, valid_masks, label_ids, label_masks = batch
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float, device=self.device)
        for i in range(batch_size):
            pos = 0
            for j in range(max_len):
                if valid_masks[i][j].item() == 1:
                    valid_output[i][pos] = sequence_output[i][j]
                    pos += 1
        bert_out = self.encoder(input_ids=input_ids)
        start_embed = self.start_layer(bert_out[0])
        end_embed = self.end_layer(bert_out[0])
        biaffine_out = self.biaffine(x=start_embed, y=end_embed)
        # [batch_size, seq_len, seq_len, n_labels]
        result = biaffine_out.permute(0, 2, 3, 1)
        return result

    def predict(
            self,
            batch=None,
    ):
        input_ids, attention_mask, segment_ids, valid_masks, label_ids, label_masks = batch
        output = self.forward(batch)
        prediction = torch.argmax(f.log_softmax(output, dim=2), dim=2)
        batch_size, max_len, feat_dim = output.shape
        valid_len = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        for i in range(batch_size):
            seq_len = 0
            for j in range(max_len):
                if label_masks[i][j].item() == 1:
                    seq_len += 1
                else:
                    valid_len[i] = seq_len
                    break
        return prediction, valid_len

    def loss(
            self,
            batch=None,
            loss_function=None,
    ):
        input_ids, attention_mask, segment_ids, valid_masks, label_ids, label_masks = batch
        output = self.forward(batch)
        if label_masks is not None:
            active_loss = label_masks.view(-1) == 1
            active_output = output.view(-1, self.label_size)[active_loss]
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_function(active_output, active_labels)
        else:
            loss = loss_function(output.view(-1, self.label_size), label_ids.view(-1))
        return loss

    def evaluate(
            self,
            batch=None,
            metrics=None,
    ):
        input_ids, attention_mask, segment_ids, valid_masks, label_ids, label_masks = batch
        prediction, valid_len = self.predict(batch)
        metrics.evaluate(prediction, label_ids, valid_len)
