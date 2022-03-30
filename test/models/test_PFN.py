import cogie
import torch
import os
import torch.nn as nn
import torch.optim as optim
from cogie.utils import load_json
from torch.utils.data import RandomSampler
from cogie.io.loader.re.nyt import NYTRELoader
from cogie.io.processor.re.nyt import NYTREProcessor
from cogie.core.loss import BCEloss


device = torch.device('cuda')
loader =NYTRELoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/re/nyt/data')
processor =NYTREProcessor(path='../../../cognlp/data/re/nyt/data',bert_model='bert-base-cased')

# ner_vocabulary = cogie.Vocabulary.load('../../../cognlp/data/re/nyt/data/ner2idx.json')
# rc_vocabulary= cogie.Vocabulary.load('../../../cognlp/data/re/nyt/data/rel2idx.json')
ner_vocabulary = load_json('../../../cognlp/data/re/nyt/data/ner2idx.json')
rc_vocabulary = load_json('../../../cognlp/data/re/nyt/data/rel2idx.json')

# train_datable = processor.process(train_data)
# train_dataset = cogie.DataTableSet(train_datable)
# train_sampler = RandomSampler(train_dataset)

# dev_datable = processor.process(dev_data)
# dev_dataset = cogie.DataTableSet(dev_datable)
# dev_sampler = RandomSampler(dev_dataset)

test_datable = processor.process(test_data)
test_dataset = cogie.DataTableSet(test_datable)
test_sampler = RandomSampler(test_dataset)

model = cogie.Bert4RePFN(dropout=0.1,hidden_size=300,ner2idx=ner_vocabulary,rel2idx=rc_vocabulary,embed_mode='bert-base-cased')
metric = cogie.ReMetric(ner2idx=ner_vocabulary,rel2idx=rc_vocabulary)
loss = BCEloss()
optimizer = optim.Adam(model.parameters(), lr=0.00002, weight_decay=0)
trainer = cogie.Trainer(model,
                        test_dataset,
                        dev_data=test_dataset,
                        n_epochs=100,
                        batch_size=20,
                        loss=loss,
                        optimizer=optimizer,
                        scheduler=None,
                        metrics=None,
                        train_sampler=test_sampler,
                        dev_sampler=test_sampler,
                        collate_fn=processor.collate_fn,
                        drop_last=False,
                        gradient_accumulation_steps=1,
                        num_workers=4,
                        save_path="../../../cognlp/data/re/nyt/model",
                        save_file=None,
                        print_every=None,
                        scheduler_steps=None,
                        validate_steps=100,
                        save_steps=None,
                        grad_norm=None,
                        use_tqdm=True,
                        device=device,
                        device_ids=[0],
                        callbacks=None,
                        metric_key=None,
                        writer_path='../../../cognlp/data/re/nyt/tensorboard',
                        fp16=False,
                        fp16_opt_level='O1',
                        checkpoint_path=None,
                        task='nyt',
                        logger_path='../../../cognlp/data/re/nyt/logger')

trainer.train()

print("end")