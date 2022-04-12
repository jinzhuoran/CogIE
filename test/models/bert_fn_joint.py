"""
@Author: jinzhuan
@File: bert_frame.py
@Desc:
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler
from cogie.io.processor.fn.framenet4joint import FrameNet4JointProcessor
from cogie.io.loader.fn.framenet4joint import FrameNet4JointLoader
from cogie import *

torch.cuda.set_device(0)
device = torch.device('cuda:0')

loader = FrameNet4JointLoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/fn/joint/data')

processor = FrameNet4JointProcessor(node_types_label_list =loader.get_node_types_labels(),
                                    node_attrs_label_list =loader.get_node_attrs_labels(),
                                    p2p_edges_label_list =loader.get_p2p_edges_labels(),
                                    p2r_edges_label_list =loader.get_p2r_edges_labels(),
                                    path='../../../cognlp/data/fn/joint/data/',
                                    bert_model='bert-base-cased',
                                    max_span_width = 15,
                                    max_length=128)
train_datable = processor.process(train_data)
train_dataset = DataTableSet(train_datable, to_device=False)
train_sampler = RandomSampler(train_dataset)

# dev_datable = processor.process(dev_data)
# dev_dataset = DataTableSet(dev_datable, to_device=False)
# dev_sampler = RandomSampler(dev_dataset)
#
# dev_datable = processor.process(test_data)
# dev_dataset = DataTableSet(dev_datable, to_device=False)
# dev_sampler = RandomSampler(dev_dataset)
model = Bert4FnJoint(node_types_vocabulary=processor.get_node_types_vocabulary,
                     node_attrs_vocabulary=processor.get_node_attrs_vocabulary,
                     p2p_edges_vocabulary=processor.get_p2p_edges_vocabulary,
                     p2r_edges_vocabulary=processor.get_p2r_edges_vocabulary,
                     device=device)
metric = AccuracyMetric()
loss = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.00005)

trainer = Trainer(model,
                  train_dataset,
                  dev_data=None,
                  n_epochs=20,
                  batch_size=3,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  train_sampler=train_sampler,
                  dev_sampler=None,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  save_path='../../../cogie/data/fn/framenet_joint/model',
                  save_file=None,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=None,
                  save_steps=1,
                  grad_norm=None,
                  collate_fn=train_dataset.to_dict,
                  use_tqdm=True,
                  device=device,
                  device_ids=[0],
                  callbacks=None,
                  metric_key=None,
                  writer_path='../../../cogie/data/fn/framenet_joint/tensorboard',
                  fp16=False,
                  fp16_opt_level='O1',
                  checkpoint_path=None,
                  task='framenet-test',
                  logger_path='../../../cogie/data/fn/framenet_joint/logger')

trainer.train()
print(1)
