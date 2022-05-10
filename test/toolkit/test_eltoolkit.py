from cogie.toolkit.ner.ner_toolkit import NerToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit
from cogie.toolkit.el.el_toolkit import ElToolkit



ner_toolkit = NerToolkit(corpus="conll2003")
# ner_toolkit = NerToolkit(corpus="trex")
tokenize_toolkit = TokenizeToolkit()
el_toolkit = ElToolkit(corpus="wiki")


# sentence = "Why has United States invaded Ukraine and what does Trump want Putin to do?"
sentence = "TianyiMen is writing code diligently in CASIA."

words = tokenize_toolkit.run(sentence)
ner_result = ner_toolkit.run(words)
el_result = el_toolkit.run(ner_result)

print("Entity Linking Result:")
for entity in el_result:
    for key in ["mention","title","text","url"]:
        print("{}:{}".format(key,entity[key]))
    print("------------------------------")

