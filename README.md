# CogIE

**Documentation is so boring? Try the DEMO website (http://cognet.top/cogie)!**

**CogIE** is an information extraction toolkit for bridging text and **CogNet**. This easy-to-use python package has the following advantages:

- **Versatility**.  We provide a professional and integrated IE toolkit. CogIE takes raw text as input and extracts entities, relations, events and frames with high-performance models.
- **Intellectuality**.  We build a bridge between raw text and CogNet. CogIE aligns the extracted facts to CogNet and leverages different types of knowledge to enrich results.
- **Extensibility**.  We contribute not just user-friendly APIs, but an extensible programming framework. Our goal in designing CogIE is to provide a universal toolkit for all sorts of users.

## What's New?
- Apr 2022: A series of updates are coming soon!
- Apr 13 2022: We have supported Unified Named Entity Recognition according to [W2NER](https://github.com/ljynlp/W2NER).
- Apr 1 2022: We have supported Joint Entity and Relation Extraction according to [PFN](https://github.com/Coopercoppers/PFN).
- Mar 30 2022: We have supported Entity Linking according to [BLINK](https://github.com/facebookresearch/BLINK).


## What's CogIE doing?

### Named Entity Recognition

Named entity recognition (NER) is the task of identifying named entities like person, location, organization, drug, time, clinical procedure, biological protein, etc. in text. NER systems are often used as the first step in question answering, information retrieval, co-reference resolution, topic modeling, etc. CogIE can not only recognize the common four entity types: locations, persons, organizations, and miscellaneous entities, but also supports the recognition of 54 entity types.

### Entity Typing

Entity Typing is an important task in text analysis. Assigning one or more types to mentions of entities in documents enables effective structured analysis of unstructured text corpora. The extracted type information can be used in a wide range of ways (e.g., serving as primitives for information extraction and knowledge base (KB) completion, and assisting question answering). There are 87 fine-grained entity lables (e.g., /person, /person/artist, /person/artist/actor) in CogIE.

### Entity Linking

Entity linking is an essential component of many information extraction and Natural Language Understanding (NLU) pipelines since it resolves the lexical ambiguity of entity mentions and determines their meanings. CogIE bridges raw data with lots of KBs, the most critical of which is CogNet. CogNet is a KB dedicated to integrating three types of knowledge: 

- linguistic knowledge, which schematically describes situations, objects, and events;
- world knowledge, which provides explicit knowledge about specific instances; 
- commonsense knowledge, which describes implicit general facts.

### Relation Extraction

Relation extraction aims at predicting semantic relations between pairs of entities. More specifically, after identifying entity mentions in text, the main goal of RE is to classify relations. There are 500 relation classes in CogIE.

### Event Extraction

Events are classified as things that happen or occur, and usually involve entities as their properties. Event extraction need to identify events that are composed of an event trigger, an event type, and a set of arguments with different roles.

### Frame-Semantic Parsing

Frame semantic parsing is the task of automatically extracting semantic structures in text following the framework of FrameNet. It consists of three separate subtasks: 

- target identification: the task of identifying all frame evoking words in a given sentence;
- frame identification: the task of identifying all frames of pre-identified targets in a given sentence; 
- argument identification: the task of identifying all frame-specific frame.

CogIE links raw text to CogNet by matching frames, there are almost 800 LUs and 1900 FEs in CogIE.

## How to use CogIE?

### Interface Call

```python
import cogie

# tokenize sentence into words
tokenize_toolkit = cogie.TokenizeToolkit(task='ws', language='english', corpus=None)
words = tokenize_toolkit.run('Ontario is the most populous province in Canada.')
# named entity recognition
ner_toolkit = cogie.NerToolkit(task='ner', language='english', corpus='trex')
ner_result = ner_toolkit.run(words)
# relation extraction
re_toolkit = cogie.ReToolkit(task='re', language='english', corpus='trex')
re_result = re_toolkit.run(words, ner_result)

token_toolkit = cogie.TokenizeToolkit(task='ws', language='english', corpus=None)
words = token_toolkit.run(
    'The true voodoo-worshipper attempts nothing of importance without certain sacrifices which are intended to propitiate his unclean gods.')
# frame identification
fn_toolkit = cogie.FnToolkit(task='fn', language='english', corpus=None)
fn_result = fn_toolkit.run(words)
# argument identification
argument_toolkit = cogie.ArgumentToolkit(task='fn', language='english', corpus='argument')
argument_result = argument_toolkit.run(words, fn_result)
```



### Model Train

```python
import cogie
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import RandomSampler
from cogie.io.loader.ner.conll2003 import Conll2003NERLoader
from cogie.io.processor.ner.conll2003 import Conll2003NERProcessor

device = torch.device('cuda')
# load dataset
loader = Conll2003NERLoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/ner/conll2003/data')
# process raw dataset
processor = Conll2003NERProcessor(label_list=loader.get_labels(), path='../../../cognlp/data/ner/conll2003/data/',
                                  bert_model='bert-large-cased')
vocabulary = cogie.Vocabulary.load('../../../cognlp/data/ner/conll2003/data/vocabulary.txt')

# add data to DataTableSet
train_datable = processor.process(train_data)
train_dataset = cogie.DataTableSet(train_datable)
train_sampler = RandomSampler(train_dataset)

dev_datable = processor.process(dev_data)
dev_dataset = cogie.DataTableSet(dev_datable)
dev_sampler = RandomSampler(dev_dataset)

test_datable = processor.process(test_data)
test_dataset = cogie.DataTableSet(test_datable)
test_sampler = RandomSampler(test_dataset)

# define model, metric, loss, optimizer 
model = cogie.Bert4Ner(len(vocabulary), bert_model='bert-base-cased', embedding_size=768)
metric = cogie.SpanFPreRecMetric(vocabulary)
loss = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# start training
trainer = cogie.Trainer(model,
                        train_dataset,
                        dev_data=dev_dataset,
                        n_epochs=20,
                        batch_size=20,
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
                        validate_steps=100,
                        save_steps=None,
                        grad_norm=None,
                        use_tqdm=True,
                        device=device,
                        device_ids=[0, 1, 2, 3],
                        callbacks=None,
                        metric_key=None,
                        writer_path='../../../cognlp/data/ner/conll2003/tensorboard',
                        fp16=False,
                        fp16_opt_level='O1',
                        checkpoint_path=None,
                        task='conll2003',
                        logger_path='../../../cognlp/data/ner/conll2003/logger')

trainer.train()
```
