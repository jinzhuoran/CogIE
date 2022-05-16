import cogie
from cogie.io.loader.rc.trex import TrexRelationLoader
from cogie.io.loader.spo.trex_spo import TrexSpoLoader
from cogie.io.processor.ner.trex_ner import TrexW2NERProcessor
from torch.utils.data import RandomSampler
import json
from argparse import Namespace
import torch
import torch.nn as nn
import transformers
from cogie.models.ner.w2ner import W2NER

device = torch.device('cuda')
loader = TrexSpoLoader(debug=False)
train_data, dev_data, test_data  = loader.load_all('../../../cognlp/data/ner/trex/data/processed_data')
print("Testing Finished!")


