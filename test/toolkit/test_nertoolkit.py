from cogie.toolkit.ner.ner_toolkit import NerToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit


ner_toolkit = NerToolkit(corpus="conll2003")
tokenize_toolkit = TokenizeToolkit()

# sentence = "Why has United States invaded Ukraine and what does Trump want?"
sentence = "There were three or four people who could have done it " \
           "but when I spoke to Alan he was up for it and really wanted it."

words = tokenize_toolkit.run(sentence)
ner_result = ner_toolkit.run(words)

