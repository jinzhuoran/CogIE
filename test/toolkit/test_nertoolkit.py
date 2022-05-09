from cogie.toolkit.ner.ner_toolkit import NerToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit


# ner_toolkit = NerToolkit(corpus="conll2003")
ner_toolkit = NerToolkit(corpus="trex")
tokenize_toolkit = TokenizeToolkit()


# sentence = "Why has United States invaded Ukraine and what does Trump want Putin to do?"
# sentence = "On 27 January 1945, Soviet troops cautiously entered Auschwitz."
# sentence = "There were three or four people who could have done it " \
#            "but when I spoke to Alan he was up for it and really wanted it."

sentence = 'Kings Have Long Arms are an English "rocktronica" act, formed in Sheffield and masterminded by Salford-born Adrian Flanagan (aka "Longy"). Kings Have Long Arms have collaborated with Philip Oakey from The Human League on the track "Rock and Roll is Dead", Marion from The Lovers and Mira from Ladytron. They have achieved recognition from the UK media as well as in Europe, where they headlined the 2004 Feedback Festival in Paris. The band\'s debut album, I Rock Eye Pop was released in 2006. It featured former Smiths members Andy Rourke and Mike Joyce, Philip Oakey, vocalist Denise Johnson and Ray Dorset (Mungo Jerry). Kings Have Long Arms released a single, "Big Umbrella", in January 2008, on Domino Records. It featured guest vocals from Candie Payne.'



words = tokenize_toolkit.run(sentence)
ner_result = ner_toolkit.run(words)

for entity in ner_result:
    print("Mention:",entity["mention"],"  Type:",entity["type"])

