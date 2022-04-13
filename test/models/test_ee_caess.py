from cogie import *
import torch
import torch.nn as nn
import torch.optim as optim
from cogie.core.metrics import EventMetric
from cogie.core.trainer import Trainer
from cogie.io.loader.ee.ace2005_casee import ACE2005CASEELoader
from cogie.io.processor.ee.ace2005_casee import ACE2005CASEEProcessor
from cogie.utils.util import get_samples_weight

torch.cuda.set_device(4)
device = torch.device('cuda:0')

loader = ACE2005CASEELoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/ee/ace2005casee/data')
processor = ACE2005CASEEProcessor(schema_path='../../../cognlp/data/ee/ace2005casee/data/schema.json',
                                  trigger_path='../../../cognlp/data/ee/ace2005casee/data/trigger_vocabulary.txt',
                                  argument_path='../../../cognlp/data/ee/ace2005casee/data/argument_vocabulary.txt'
                                  )
train_datable = processor.process_train(train_data)
train_dataset = DataTableSet(train_datable, to_device=False)

dev_datable = processor.process_dev(dev_data)
dev_dataset = DataTableSet(dev_datable, to_device=False)
#
# test_datable = processor.process(test_data)
# test_dataset = DataTableSet(test_datable, to_device=False)
#
# model = Bert4EE(trigger_vocabulary=trigger_vocabulary, argument_vocabulary=argument_vocabulary)
# loss = nn.CrossEntropyLoss(ignore_index=0)
# optimizer = optim.Adam(model.parameters(), lr=0.00005)
# metric = EventMetric(trigger_vocabulary, argument_vocabulary)
#
# trainer = Trainer(model,
#                   train_dataset,
#                   dev_data=test_dataset,
#                   n_epochs=30,
#                   batch_size=20,
#                   loss=loss,
#                   optimizer=optimizer,
#                   scheduler=None,
#                   metrics=metric,
#                   train_sampler=train_sampler,
#                   dev_sampler=None,
#                   drop_last=False,
#                   gradient_accumulation_steps=1,
#                   num_workers=None,
#                   save_path='../../../cognlp/data/ee/ace2005/model',
#                   save_file=None,
#                   print_every=None,
#                   scheduler_steps=None,
#                   validate_steps=None,
#                   save_steps=100,
#                   grad_norm=1.0,
#                   use_tqdm=True,
#                   device=device,
#                   device_ids=[4],
#                   collate_fn=train_dataset.to_dict,
#                   callbacks=None,
#                   metric_key=None,
#                   writer_path='../../../cognlp/data/ee/ace2005/tensorboard',
#                   fp16=False,
#                   fp16_opt_level='O1',
#                   checkpoint_path=None,
#                   task='ace2005-event-test',
#                   logger_path='../../../cognlp/data/ee/ace2005/logger')
#
# trainer.train()