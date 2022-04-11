import logging

import cogie
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler
from cogie.io.loader.ner.conll2003 import Conll2003NERLoader
from cogie.io.processor.ner.conll2003 import Conll2003NERProcessor,Conll2003W2NERProcessor
from cogie.models.ner.w2ner import W2NER
import json
from argparse import Namespace
import json
import transformers

from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import prettytable as pt

import random
import numpy as np
from cogie.utils import seed_everything
init_seed = 123
seed_everything(init_seed)

device = torch.device('cuda')
loader = Conll2003NERLoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/ner/conll2003/data')
vocabulary = cogie.Vocabulary.load('../../../cognlp/data/ner/conll2003/data/vocabulary.txt')
# vocabulary.idx2word = {0: '<pad>', 1: '<suc>', 2: 'b-org', 3: 'b-misc', 4: 'b-per', 5: 'i-per', 6: 'b-loc'}
# vocabulary.word2idx = {'<pad>': 0, '<suc>': 1, 'b-org': 2, 'b-misc': 3, 'b-per': 4, 'i-per': 5, 'b-loc': 6}
vocabulary.idx2word = {0: '<pad>', 1: '<suc>', 2: 'b-org', 3: 'b-misc', 4: 'b-per', 5: 'i-per', 6: 'b-loc', 7: 'i-org', 8: 'i-misc', 9: 'i-loc'}
vocabulary.word2idx = {'<pad>': 0, '<suc>': 1, 'b-org': 2, 'b-misc': 3, 'b-per': 4, 'i-per': 5, 'b-loc': 6, 'i-org': 7, 'i-misc': 8, 'i-loc': 9}

# def squeeze_data(data,num=2):
#     data.datas["sentence"] = data.datas["sentence"][:2]
#     data.datas["label"] = data.datas["label"][:2]
#     return data
# train_data = squeeze_data(train_data)
# dev_data = squeeze_data(dev_data)
# test_data = squeeze_data(test_data)

# def convert_to_json(data,file_name):
#     json_data = []
#     for i, (sentence, labels) in enumerate(zip(data.datas["sentence"], data.datas["label"])):
#         sample = {}
#         sample["sentence"] = sentence
#         ners = []
#         for j, label in enumerate(labels):
#             if label == "O":
#                 continue
#             elif label == "<pad>" or label == "<unk>":
#                 raise ValueError("????label={}???".format(label))
#             else:
#                 ners.append({"index": [j], "type": label})
#         sample["ner"] = ners
#         json_data.append(sample)
#     # json_data = json_data[0:2]
#     with open(file_name, "w") as f:
#         json.dump(json_data, f)
# convert_to_json(train_data,"train.json")
# convert_to_json(dev_data,"dev.json")
# convert_to_json(test_data,"test.json")

processor = Conll2003W2NERProcessor(label_list=loader.get_labels(), path='../../../cognlp/data/ner/conll2003/data/',
                                  bert_model='bert-large-cased',max_length=256)
train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)
# train_sampler = torch.utils.data.SequentialSampler(train_dataset)

dev_datable = processor.process(dev_data)
dev_dataset = cogie.DataTableSet(dev_datable)
dev_sampler = RandomSampler(dev_dataset)
# dev_sampler = torch.utils.data.SequentialSampler(dev_dataset)

# test_datable = processor.process(test_data)
# test_dataset = cogie.DataTableSet(test_datable)
# test_sampler = RandomSampler(test_dataset)

with open("./conll03.json","r") as f:
    config = json.load(f)
config = Namespace(**config)
config.label_num = len(vocabulary.word2idx)
print("label num:",config.label_num)
config.vocab = vocabulary
model = W2NER(config)

# metric = cogie.SpanFPreRecMetric(vocabulary)
metric = None
loss = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.000005)
# optimizer = optim.Adam(model.parameters(),lr=1e-5)


bert_params = set(model.bert.parameters())
other_params = list(set(model.parameters()) - bert_params)
no_decay = ['bias', 'LayerNorm.weight']
params = [
    {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
     'lr': config.bert_learning_rate,
     'weight_decay': config.weight_decay},
    {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
     'lr': config.bert_learning_rate,
     'weight_decay': 0.0},
    {'params': other_params,
     'lr': config.learning_rate,
     'weight_decay': config.weight_decay},
]

updates_total = len(train_dataset) // config.batch_size * config.epochs
optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                              num_warmup_steps=config.warm_factor * updates_total,
                                                              num_training_steps=updates_total)
# logger = logging.getLogger()
#
# class Trainer(object):
#     def __init__(self, model):
#         self.model = model
#         self.criterion = nn.CrossEntropyLoss()
#
#         bert_params = set(self.model.bert.parameters())
#         other_params = list(set(self.model.parameters()) - bert_params)
#         no_decay = ['bias', 'LayerNorm.weight']
#         params = [
#             {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
#              'lr': config.bert_learning_rate,
#              'weight_decay': config.weight_decay},
#             {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
#              'lr': config.bert_learning_rate,
#              'weight_decay': 0.0},
#             {'params': other_params,
#              'lr': config.learning_rate,
#              'weight_decay': config.weight_decay},
#         ]
#
#         self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
#         self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
#                                                                       num_warmup_steps=config.warm_factor * updates_total,
#                                                                       num_training_steps=updates_total)
#
#     def train(self, epoch, data_loader):
#         self.model.train()
#         loss_list = []
#         pred_result = []
#         label_result = []
#
#         for i, data_batch in enumerate(tqdm(data_loader)):
#             data_batch = [data.cuda() for data in data_batch]
#
#             bert_inputs, attention_masks, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
#             outputs = model(bert_inputs, attention_masks, grid_mask2d, dist_inputs, pieces2word, sent_length)
#
#             # bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
#             # outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
#
#             grid_mask2d = grid_mask2d.clone()
#             loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
#             self.optimizer.step()
#             self.optimizer.zero_grad()
#
#             loss_list.append(loss.cpu().item())
#             # print("loss:",loss.cpu().item())
#
#             outputs = torch.argmax(outputs, -1)
#             grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
#             outputs = outputs[grid_mask2d].contiguous().view(-1)
#
#             label_result.append(grid_labels)
#             pred_result.append(outputs)
#
#             self.scheduler.step()
#
#         label_result = torch.cat(label_result)
#         pred_result = torch.cat(pred_result)
#
#         p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
#                                                       pred_result.cpu().numpy(),
#                                                       average="macro")
#
#         table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
#         table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
#                       ["{:3.4f}".format(x) for x in [f1, p, r]])
#         logger.info("\n{}".format(table))
#         return f1
#
#     def save(self, path):
#         torch.save(self.model.state_dict(), path)
#
# train_loader = DataLoader(dataset=train_dataset,batch_size=5,
#                           shuffle=False)
# model = model.cuda()
# trainer = Trainer(model)
# for i in range(config.epochs):
#     logger.info("Epoch: {}".format(i))
#     trainer.train(i, train_loader)

trainer = cogie.Trainer(model,
                        train_dataset,
                        dev_data=dev_dataset,
                        n_epochs=10,
                        batch_size=config.batch_size,
                        loss=loss,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metrics=metric,
                        train_sampler=train_sampler,
                        dev_sampler=dev_sampler,
                        drop_last=False,
                        gradient_accumulation_steps=1,
                        num_workers=5,
                        save_path="../../../cognlp/data/ner/conll2003/model",
                        save_file=None,
                        print_every=None,
                        scheduler_steps=1,
                        validate_steps=4000,
                        save_steps=None,
                        grad_norm=1,# 梯度裁减
                        use_tqdm=True,
                        device=device,
                        device_ids=[0],
                        callbacks=None,
                        metric_key=None,
                        writer_path='../../../cognlp/data/ner/conll2003/tensorboard',
                        fp16=False,
                        fp16_opt_level='O1',
                        # checkpoint_path="./checkpoint-1",
                        task='conll2003',
                        logger_path='../../../cognlp/data/ner/conll2003/logger')

trainer.train()




