from cogie.toolkit.ner.ner_toolkit import NerToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit
from cogie.toolkit.el.el_toolkit import ElToolkit
from cogie.toolkit.et.et_toolkit import EtToolkit


ner_toolkit = NerToolkit(corpus="conll2003")
# ner_toolkit = NerToolkit(corpus="trex")
tokenize_toolkit = TokenizeToolkit()
et_toolkit = EtToolkit(corpus="ufet")


# sentence = "Why has United States invaded Ukraine and what does Trump want Putin to do?"
sentence = "TianyiMen is writing code diligently in CASIA with only one screen on his desk."

words = tokenize_toolkit.run(sentence)
ner_result = ner_toolkit.run(words)
et_result = et_toolkit.run(ner_result)
print(et_result)

returned_et_result = [
            {
                "mention":words[entity["start"]:entity["end"]],
                "start":entity["start"],
                "end":entity["end"],
                "type":entity["types"]
            }
            for entity in et_result
        ]
print("Entity Typing Result:")
for entity in et_result:
    for key in ["mention","type","types"]:
        print("{}:{}".format(key,entity[key]))
    print("------------------------------")

