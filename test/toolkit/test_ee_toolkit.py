from cogie.toolkit.ee.ee_toolkit import EeToolkit
from cogie.toolkit.ner.ner_toolkit import NerToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit

ner_toolkit = NerToolkit(corpus="trex")
ee_toolkit = EeToolkit(task='ee', language='english', corpus='ace2005')
tokenize_toolkit = TokenizeToolkit()

sentence ="British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country's energy regulator as the new chairman of finance watchdog the Financial Services Authority (FSA)."

words = tokenize_toolkit.run(sentence)
spans = ner_toolkit.run(words)
ee_result = ee_toolkit.run(words,spans)

print("end")