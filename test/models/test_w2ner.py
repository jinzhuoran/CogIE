import json
import json
import logging
import numpy as np
import prettytable as pt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from argparse import Namespace
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm

import cogie
from cogie.io.loader.ner.conll2003 import Conll2003NERLoader
from cogie.io.processor.ner.conll2003 import Conll2003NERProcessor, Conll2003W2NERProcessor
from cogie.models.ner.w2ner import W2NER
from cogie.utils import seed_everything

init_seed = 123
seed_everything(init_seed)

device = torch.device('cuda')
loader = Conll2003NERLoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/ner/conll2003/data')

# vocabulary.idx2word = {0: '<pad>', 1: '<suc>', 2: 'b-org', 3: 'b-misc', 4: 'b-per', 5: 'i-per', 6: 'b-loc'}
# vocabulary.word2idx = {'<pad>': 0, '<suc>': 1, 'b-org': 2, 'b-misc': 3, 'b-per': 4, 'i-per': 5, 'b-loc': 6}
# vocabulary.idx2word = {0: '<pad>', 1: '<suc>', 2: 'b-org', 3: 'b-misc', 4: 'b-per', 5: 'i-per', 6: 'b-loc', 7: 'i-org', 8: 'i-misc', 9: 'i-loc'}
# vocabulary.word2idx = {'<pad>': 0, '<suc>': 1, 'b-org': 2, 'b-misc': 3, 'b-per': 4, 'i-per': 5, 'b-loc': 6, 'i-org': 7, 'i-misc': 8, 'i-loc': 9}

processor = Conll2003W2NERProcessor(label_list=loader.get_labels(),path='../../../cognlp/data/ner/conll2003/data/',
                                    bert_model='bert-large-cased', max_length=256)
vocabulary = cogie.Vocabulary.load('../../../cognlp/data/ner/conll2003/data/vocabulary.txt')
train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)
# train_sampler = torch.utils.data.SequentialSampler(train_dataset)

dev_datable = processor.process(dev_data)
dev_dataset = cogie.DataTableSet(dev_datable)
dev_sampler = RandomSampler(dev_dataset)
# dev_sampler = torch.utils.data.SequentialSampler(dev_dataset)

test_datable = processor.process(test_data)
test_dataset = cogie.DataTableSet(test_datable)
test_sampler = RandomSampler(test_dataset)

with open("./conll03.json", "r") as f:
    config = json.load(f)
config = Namespace(**config)
config.label_num = len(vocabulary.word2idx)
print("label num:", config.label_num)
config.vocab = vocabulary
model = W2NER(config)

# metric = cogie.ClassifyFPreRecMetric(vocabulary)
metric = cogie.ClassifyFPreRecMetric(f_type='macro')
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

trainer = cogie.Trainer(model,
                        train_dataset,
                        dev_data=test_dataset,
                        n_epochs=3,
                        batch_size=config.batch_size,
                        loss=loss,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metrics=metric,
                        train_sampler=train_sampler,
                        dev_sampler=test_sampler,
                        drop_last=False,
                        gradient_accumulation_steps=1,
                        num_workers=5,
                        save_path="../../../cognlp/data/ner/conll2003/model",
                        save_file=None,
                        print_every=None,
                        scheduler_steps=1,
                        validate_steps=None,
                        save_steps=None,
                        grad_norm=1,  # 梯度裁减
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
