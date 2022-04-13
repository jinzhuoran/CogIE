import cogie
from cogie.io.loader.rc.trex import TrexRelationLoader
from cogie.io.loader.ner.trex_ner import TrexNerLoader
from cogie.io.processor.ner.trex_ner import TrexW2NERProcessor
from torch.utils.data import RandomSampler
import json
from argparse import Namespace
import torch
import torch.nn as nn
import transformers
from cogie.models.ner.w2ner import W2NER

device = torch.device('cuda')
loader = TrexNerLoader()
train_data, dev_data, test_data  = loader.load_all('../../../cognlp/data/ner/trex/data/trex_debug.json')

processor = TrexW2NERProcessor(label_list=loader.get_labels(),path='../../../cognlp/data/ner/conll2003/data/',
                                  bert_model='bert-large-cased',max_length=512)
train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)

dev_datable = processor.process(dev_data)
dev_dataset = cogie.DataTableSet(dev_datable)
dev_sampler = RandomSampler(dev_dataset)
# dev_sampler = torch.utils.data.SequentialSampler(dev_dataset)

test_datable = processor.process(test_data)
test_dataset = cogie.DataTableSet(test_datable)
test_sampler = RandomSampler(test_dataset)

with open("../models/conll03.json","r") as f:
    config = json.load(f)
config = Namespace(**config)
vocabulary = processor.get_vocabulary()
config.label_num = len(vocabulary.word2idx)
print("label num:",config.label_num)
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
                        n_epochs=10,
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
                        validate_steps=350,
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


