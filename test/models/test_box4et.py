import torch
from cogie.io.loader.et.ufet import UfetLoader

device = torch.device('cuda')
loader = UfetLoader()
train_data, dev_data, test_data  = loader.load_all('../../../cognlp/data/et/ufet/data/')
print("Hello World!")

