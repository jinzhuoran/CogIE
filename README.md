# CogIE

**Documentation is so boring? Try the DEMO website (http://cognet.top/cogie)!**

**CogIE** is an information extraction toolkit for bridging text and **CogNet**. This easy-to-use python package has the following advantages:

- **Versatility**.  We provide a professional and integrated IE toolkit. CogIE takes raw text as input and extracts entities, relations, events and frames with high-performance models.
- **Intellectuality**.  We build a bridge between raw text and CogNet. CogIE aligns the extracted facts to CogNet and leverages different types of knowledge to enrich results.
- **Extensibility**.  We contribute not just user-friendly APIs, but an extensible programming framework. Our goal in designing CogIE is to provide a universal toolkit for all sorts of users.

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