import torch
import torch.optim as optim
from torch.utils.data import RandomSampler

import cogie
from cogie.core.loss import BCEloss
from cogie.io.loader.re.nyt import NYTRELoader
from cogie.io.processor.re.nyt import NYTREProcessor
from cogie.utils import load_json

torch.cuda.set_device(4)
device = torch.device('cuda:0')

loader = NYTRELoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/spo/nyt/data')
processor = NYTREProcessor(path='../../../cognlp/data/spo/nyt/data', bert_model='bert-base-cased')

ner_vocabulary = load_json('../../../cognlp/data/spo/nyt/data/ner2idx.json')
rc_vocabulary = load_json('../../../cognlp/data/spo/nyt/data/rel2idx.json')

train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)

dev_datable = processor.process(dev_data)
dev_dataset = cogie.DataTableSet(dev_datable)
dev_sampler = RandomSampler(dev_dataset)

test_datable = processor.process(test_data)
test_dataset = cogie.DataTableSet(test_datable)
test_sampler = RandomSampler(test_dataset)

model = cogie.PFN(dropout=0.1, hidden_size=300, ner2idx=ner_vocabulary, rel2idx=rc_vocabulary,
                  embed_mode='bert-base-cased')
metric = cogie.SPOMetric(ner2idx=ner_vocabulary, rel2idx=rc_vocabulary, eval_metric="micro")
loss = BCEloss()
optimizer = optim.Adam(model.parameters(), lr=0.00002, weight_decay=0)
trainer = cogie.Trainer(model,
                        train_dataset,
                        dev_data=test_dataset,
                        n_epochs=100,
                        batch_size=20,
                        loss=loss,
                        optimizer=optimizer,
                        scheduler=None,
                        metrics=metric,
                        train_sampler=train_sampler,
                        dev_sampler=test_sampler,
                        collate_fn=processor.collate_fn,
                        drop_last=False,
                        gradient_accumulation_steps=1,
                        num_workers=4,
                        save_path="../../../cognlp/data/spo/nyt/model",
                        save_file=None,
                        print_every=None,
                        scheduler_steps=None,
                        validate_steps=2810,
                        save_steps=1,
                        grad_norm=None,
                        use_tqdm=True,
                        device=device,
                        device_ids=[4],
                        callbacks=None,
                        metric_key=None,
                        writer_path='../../../cognlp/data/spo/nyt/tensorboard',
                        fp16=False,
                        fp16_opt_level='O1',
                        checkpoint_path=None,
                        task='nyt',
                        logger_path='../../../cognlp/data/spo/nyt/logger')

trainer.train()
