import sys

sys.path.append('/data/zhuoran/code/cognlp')
sys.path.append('/data/zhuoran/cognlp')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler
from cogie import *

torch.cuda.set_device(4)
device = torch.device('cuda')

from cogie.io.loader.fn.frame_argument import FrameArgumentLoader
from cogie.io.processor.fn.frame_argument import FrameArgumentProcessor

from cogie.models.fn.bert_argument import Bert4Argument

loader = FrameArgumentLoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/fn/framenet/data')

processor = FrameArgumentProcessor(path='../../../cognlp/data/fn/framenet/data',
                                   bert_model='bert-base-cased',
                                   trigger_label_list=loader.get_trigger_labels(),
                                   argument_label_list=loader.get_argument_labels())
vocabulary = Vocabulary.load('../../../cognlp/data/fn/framenet/data/vocabulary.txt')
frame_vocabulary = processor.trigger_vocabulary
frame_vocabulary.save('../../../cognlp/data/fn/argument/toolkit/frame_vocabulary.txt')
train_datable = processor.process(train_data)
train_dataset = DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)

test_datable = processor.process(test_data)
test_dataset = DataTableSet(test_datable)
test_sampler = RandomSampler(test_dataset)

model = Bert4Argument(len(vocabulary), bert_model='bert-base-cased', embedding_size=768,
                      frame_vocabulary=frame_vocabulary)
metric = SpanFPreRecMetric(vocabulary)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
trainer = Trainer(model,
                  train_dataset,
                  dev_data=test_dataset,
                  n_epochs=200,
                  batch_size=80,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  metrics=metric,
                  train_sampler=train_sampler,
                  dev_sampler=test_sampler,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=5,
                  save_path='../../../cognlp/data/fn/framenet/model',
                  save_file=None,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=50,
                  save_steps=None,
                  grad_norm=None,
                  use_tqdm=True,
                  device=device,
                  device_ids=[4, 5, 6, 7],
                  callbacks=None,
                  metric_key=None,
                  writer_path='../../../cognlp/data/fn/framenet/tensorboard',
                  fp16=False,
                  fp16_opt_level='O1',
                  checkpoint_path='../../../cognlp/data/fn/framenet/model/argument/2021-03-15-19:59:30/checkpoint-8604/',
                  task='argument',
                  logger_path='../../../cognlp/data/fn/framenet/logger')

trainer.train()
print(1)