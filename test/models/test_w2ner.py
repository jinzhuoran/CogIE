import cogie
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler
from cogie.io.loader.ner.conll2003 import Conll2003NERLoader
from cogie.io.processor.ner.conll2003 import Conll2003NERProcessor,Conll2003W2NERProcessor


device = torch.device('cuda')
loader = Conll2003NERLoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/ner/conll2003/data')
vocabulary = cogie.Vocabulary.load('../../../cognlp/data/ner/conll2003/data/vocabulary.txt')

processor = Conll2003W2NERProcessor(label_list=loader.get_labels(), path='../../../cognlp/data/ner/conll2003/data/',
                                  bert_model='bert-large-cased')
train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)

# dev_datable = processor.process(dev_data)
# dev_dataset = cogie.DataTableSet(dev_datable)
# dev_sampler = RandomSampler(dev_dataset)
#
# test_datable = processor.process(test_data)
# test_dataset = cogie.DataTableSet(test_datable)
# test_sampler = RandomSampler(test_dataset)


