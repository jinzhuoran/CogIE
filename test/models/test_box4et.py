import torch
from cogie.io.loader.et.ufet import UfetLoader
from cogie.io.processor.et.ufet import UfetProcessor
from cogie.models.et.box4et import TransformerBoxModel
import cogie
from torch.utils.data import RandomSampler
import json
from argparse import Namespace
import torch
import cogie.utils.box4et_constant as constant
import torch.nn as nn
import torch.optim as optim
from cogie.core.metrics import MultiLabelStrictAccuracyMetric,FBetaMultiLabelMetric

device = torch.device('cuda')
loader = UfetLoader()
train_data, dev_data, test_data  = loader.load_all('../../../cognlp/data/et/ufet/data/')
print("Hello World!")

processor = UfetProcessor(path='../../../cognlp/data/et/ufet/data/',
                                  bert_model='bert-large-uncased-whole-word-masking')
train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)

# dev_datable = processor.process(dev_data)
# dev_dataset = cogie.DataTableSet(dev_datable)
# dev_sampler = RandomSampler(dev_dataset)

test_datable = processor.process(test_data)
test_dataset = cogie.DataTableSet(test_datable)
test_sampler = RandomSampler(test_dataset)

with open("ufet_config.json","r") as f:
    config = json.load(f)
args = Namespace(**config)
model = TransformerBoxModel(args,constant.ANSWER_NUM_DICT[args.goal])

metric = FBetaMultiLabelMetric(average='macro')
loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.000005)
trainer = cogie.Trainer(model,
                        test_dataset,
                        dev_data=test_dataset,
                        n_epochs=10000,
                        batch_size=8,
                        loss=loss,
                        optimizer=optimizer,
                        scheduler=None,
                        metrics=metric,
                        train_sampler=test_sampler,
                        dev_sampler=test_sampler,
                        drop_last=False,
                        gradient_accumulation_steps=1,
                        num_workers=5,
                        save_path="../../../cognlp/data/et/ufet/model",
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
                        writer_path='../../../cognlp/data/et/ufet/tensorboard',
                        fp16=False,
                        fp16_opt_level='O1',
                        checkpoint_path="../../../cognlp/data/et/ufet/model/pretrained/checkpoint-1",
                        logger_path='../../../cognlp/data/et/ufet/logger')

trainer.train()


print("Finished!")
