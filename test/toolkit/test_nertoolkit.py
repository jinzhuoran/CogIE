from cogie.toolkit.ner.ner_toolkit import NerToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit


ner_toolkit = NerToolkit(corpus="conll2003")
tokenize_toolkit = TokenizeToolkit()

sentence = "Why has United States invaded Ukraine and what does Trump want Putin to do?"
# sentence = "China plays an important role in this invisible war started by Trump."
# sentence = "There were three or four people who could have done it " \
#            "but when I spoke to Alan he was up for it and really wanted it."

words = tokenize_toolkit.run(sentence)
ner_result = ner_toolkit.run(words)

result_dict = []
for word_id_list,label in ner_result:
    for word_id in word_id_list:
        result_dict.append([word_id,ner_toolkit.vocabulary.idx2word[label]])
result_dict.sort(key=lambda x:x[0])
for word_id,label in result_dict:
    print(words[word_id],label)

