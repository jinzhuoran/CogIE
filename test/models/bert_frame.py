"""
@Author: jinzhuan
@File: bert_frame.py
@Desc: 
"""
import sys
sys.path.append('/data/mentianyi/code/CogIE')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler
from cogie.io.processor.fn.framenet import FrameNetProcessor
from cogie.io.loader.fn.framenet import FrameNetLoader
from cogie import *

torch.cuda.set_device(5)
device = torch.device('cuda:5')

loader = FrameNetLoader()
train_data, dev_data, test_data = loader.load_all('../../cognlp/data/fn/framenet/data')
processor = FrameNetProcessor(frame_path='../../cognlp/data/fn/framenet/data/frame_vocabulary.txt',
                              element_path='../../cognlp/data/fn/framenet/data/element_vocabulary.txt')

vocabulary = Vocabulary.load('../../cognlp/data/fn/framenet/data/frame_vocabulary.txt')

train_datable = processor.process(train_data)
train_dataset = DataTableSet(train_datable, to_device=False)
train_sampler = RandomSampler(train_dataset)

dev_datable = processor.process(dev_data)
dev_dataset = DataTableSet(dev_datable, to_device=False)
dev_sampler = RandomSampler(dev_dataset)

dev_datable = processor.process(test_data)
dev_dataset = DataTableSet(dev_datable, to_device=False)
dev_sampler = RandomSampler(dev_dataset)

model = Bert4Frame(len(vocabulary), device=device)
metric = AccuracyMetric()
loss = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.00005)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=20,
                  batch_size=20,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  train_sampler=train_sampler,
                  dev_sampler=dev_sampler,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  save_path='../../../cognlp/data/fn/framenet/model',
                  save_file=None,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=None,
                  save_steps=300,
                  grad_norm=None,
                  collate_fn=train_dataset.to_dict,
                  use_tqdm=True,
                  device=device,
                  device_ids=[5],
                  callbacks=None,
                  metric_key=None,
                  writer_path='../../../cognlp/data/fn/framenet/tensorboard',
                  fp16=False,
                  fp16_opt_level='O1',
                  checkpoint_path=None,
                  task='framenet-test',
                  logger_path='../../../cognlp/data/fn/framenet/logger')

trainer.train()
print(1)
