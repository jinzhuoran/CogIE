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


device = torch.device('cuda')
loader = Conll2003NERLoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/ner/conll2003/data')
vocabulary = cogie.Vocabulary.load('../../../cognlp/data/ner/conll2003/data/vocabulary.txt')

processor = Conll2003W2NERProcessor(label_list=loader.get_labels(), path='../../../cognlp/data/ner/conll2003/data/',
                                  bert_model='bert-large-cased')
train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)

dev_datable = processor.process(dev_data)
dev_dataset = cogie.DataTableSet(dev_datable)
dev_sampler = RandomSampler(dev_dataset)
#
# test_datable = processor.process(test_data)
# test_dataset = cogie.DataTableSet(test_datable)
# test_sampler = RandomSampler(test_dataset)

with open("./conll03.json","r") as f:
    config = json.load(f)
config = Namespace(**config)
config.label_num = len(vocabulary.word2idx)
config.vocab = vocabulary
model = W2NER(config)

metric = cogie.SpanFPreRecMetric(vocabulary)
loss = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.000005)
trainer = cogie.Trainer(model,
                        train_dataset,
                        dev_data=dev_dataset,
                        n_epochs=2,
                        batch_size=5,
                        loss=loss,
                        optimizer=optimizer,
                        scheduler=None,
                        metrics=metric,
                        train_sampler=train_sampler,
                        dev_sampler=dev_sampler,
                        drop_last=False,
                        gradient_accumulation_steps=1,
                        num_workers=5,
                        save_path="../../../cognlp/data/ner/conll2003/model",
                        save_file=None,
                        print_every=None,
                        scheduler_steps=None,
                        validate_steps=1,
                        save_steps=None,
                        grad_norm=None,
                        use_tqdm=True,
                        device=device,
                        device_ids=[0],
                        callbacks=None,
                        metric_key=None,
                        writer_path='../../../cognlp/data/ner/conll2003/tensorboard',
                        fp16=False,
                        fp16_opt_level='O1',
                        checkpoint_path=None,
                        task='conll2003',
                        logger_path='../../../cognlp/data/ner/conll2003/logger')

trainer.train()




