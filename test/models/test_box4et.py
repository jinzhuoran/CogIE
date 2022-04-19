import torch
from cogie.io.loader.et.ufet import UfetLoader
from cogie.io.processor.et.ufet import UfetProcessor
import cogie
from torch.utils.data import RandomSampler
import json
from argparse import Namespace
import torch
import torch.nn as nn

device = torch.device('cuda')
loader = UfetLoader()
train_data, dev_data, test_data  = loader.load_all('../../../cognlp/data/et/ufet/data/')
print("Hello World!")

processor = UfetProcessor(label_list=loader.get_labels(), path='../../../cognlp/data/ner/conll2003/data/',
                                  bert_model='bert-large-uncased-whole-word-masking')
train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)

dev_datable = processor.process(dev_data)
dev_dataset = cogie.DataTableSet(dev_datable)
dev_sampler = RandomSampler(dev_dataset)

test_datable = processor.process(test_data)
test_dataset = cogie.DataTableSet(test_datable)
test_sampler = RandomSampler(test_dataset)

with open("ufet_config.json","r") as f:
    config = json.load(f)
args = Namespace(**config)

