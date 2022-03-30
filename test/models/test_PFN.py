import cogie
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler
from cogie.io.loader.re.nyt import NYTRELoader
from cogie.io.processor.re.nyt import NYTREProcessor


device = torch.device('cuda')
loader =NYTRELoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/re/nyt/data')
processor =NYTREProcessor(path='../../../cognlp/data/re/nyt/data',bert_model='bert-base-cased')

# ner_vocabulary = cogie.Vocabulary.load('../../../cognlp/data/re/nyt/data/ner2idx')
# rc_vocabulary= cogie.Vocabulary.load('../../../cognlp/data/re/nyt/data/rel2idx')

train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)

dev_datable = processor.process(dev_data)
dev_dataset = cogie.DataTableSet(dev_datable)
dev_sampler = RandomSampler(dev_dataset)

test_datable = processor.process(test_data)
test_dataset = cogie.DataTableSet(test_datable)
test_sampler = RandomSampler(test_dataset)

#
# model = cogie.Bert4Ner(len(vocabulary), bert_model='bert-large-cased', embedding_size=1024)
# metric = cogie.SpanFPreRecMetric(vocabulary)
# loss = nn.CrossEntropyLoss(ignore_index=0)
# optimizer = optim.Adam(model.parameters(), lr=0.000005)
# trainer = cogie.Trainer(model,
#                         train_dataset,
#                         dev_data=dev_dataset,
#                         n_epochs=20,
#                         batch_size=20,
#                         loss=loss,
#                         optimizer=optimizer,
#                         scheduler=None,
#                         metrics=metric,
#                         train_sampler=train_sampler,
#                         dev_sampler=dev_sampler,
#                         drop_last=False,
#                         gradient_accumulation_steps=1,
#                         num_workers=5,
#                         save_path="../../../cognlp/data/ner/conll2003/model",
#                         save_file=None,
#                         print_every=None,
#                         scheduler_steps=None,
#                         validate_steps=100,
#                         save_steps=None,
#                         grad_norm=None,
#                         use_tqdm=True,
#                         device=device,
#                         device_ids=[0,1],
#                         callbacks=None,
#                         metric_key=None,
#                         writer_path='../../../cognlp/data/ner/conll2003/tensorboard',
#                         fp16=False,
#                         fp16_opt_level='O1',
#                         checkpoint_path=None,
#                         task='conll2003',
#                         logger_path='../../../cognlp/data/ner/conll2003/logger')
#
# trainer.train()