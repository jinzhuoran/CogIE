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