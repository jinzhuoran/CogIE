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
import json

device = torch.device('cuda')
loader = Conll2003NERLoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/ner/conll2003/data')
vocabulary = cogie.Vocabulary.load('../../../cognlp/data/ner/conll2003/data/vocabulary.txt')
vocabulary.idx2word = {2: 'B-LOC', 3: 'B-ORG', 4: 'I-MISC', 5: 'B-MISC', 6: 'I-LOC', 7: 'B-PER', 8: 'I-ORG', 9: 'I-PER', 0: '<pad>', 1: '<unk>'}
vocabulary.word2idx = {'B-LOC': 2, 'B-ORG': 3, 'I-MISC': 4, 'B-MISC': 5, 'I-LOC': 6, 'B-PER': 7, 'I-ORG': 8, 'I-PER': 9, '<pad>': 0, '<unk>': 1}
def convert_to_json(data,file_name):
    json_data = []
    for i, (sentence, labels) in enumerate(zip(data.datas["sentence"], data.datas["label"])):
        sample = {}
        sample["sentence"] = sentence
        ners = []
        for j, label in enumerate(labels):
            if label == "O":
                continue
            elif label == "<pad>" or label == "<unk>":
                raise ValueError("????label={}???".format(label))
            else:
                ners.append({"index": [j], "type": label})
        sample["ner"] = ners
        json_data.append(sample)
    json_data = json_data[0:2]
    with open(file_name, "w") as f:
        json.dump(json_data, f)
convert_to_json(train_data,"train.json")
convert_to_json(dev_data,"dev.json")
convert_to_json(test_data,"test.json")

# train_json_data = []
# for i,(sentence,labels) in enumerate(zip(train_data.datas["sentence"],train_data.datas["label"])):
#     sample = {}
#     sample["sentence"] = sentence
#     ners = []
#     for j,label in enumerate(labels):
#         if label == "O":
#             continue
#         elif label == "<pad>" or label == "<unk>":
#             raise ValueError("????label={}???".format(label))
#         else:
#             ners.append({"index":[j],"type":label})
#     sample["ner"] = ners
#     train_json_data.append(sample)
# with open("train.json","w") as f:
#     json.dump(train_json_data,f)

processor = Conll2003W2NERProcessor(label_list=loader.get_labels(), path='../../../cognlp/data/ner/conll2003/data/',
                                  bert_model='bert-large-cased',max_length=200)
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
config.label_num = len(vocabulary.word2idx)   # O should not be considered as one class
print("label num:",config.label_num)
config.vocab = vocabulary
model = W2NER(config)

# metric = cogie.SpanFPreRecMetric(vocabulary)
metric = None
loss = nn.CrossEntropyLoss(ignore_index=0)
# optimizer = optim.Adam(model.parameters(), lr=0.000005)
optimizer = optim.Adam(model.parameters(),lr=1e-5)
trainer = cogie.Trainer(model,
                        train_dataset,
                        dev_data=dev_dataset,
                        n_epochs=100,
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
                        validate_steps=1000,
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




