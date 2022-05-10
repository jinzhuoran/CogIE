from cogie.toolkit.fn.fn_toolkit import FnToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit

fn_toolkit = FnToolkit(task='fn', language='english', corpus='frame')
tokenize_toolkit = TokenizeToolkit()

sentence = 'Kings Have Long Arms are an English "rocktronica" act, formed in Sheffield and masterminded by Salford-born Adrian Flanagan (aka "Longy"). Kings Have Long Arms have collaborated with Philip Oakey from The Human League on the track "Rock and Roll is Dead", Marion from The Lovers and Mira from Ladytron. They have achieved recognition from the UK media as well as in Europe, where they headlined the 2004 Feedback Festival in Paris. The band\'s debut album, I Rock Eye Pop was released in 2006. It featured former Smiths members Andy Rourke and Mike Joyce, Philip Oakey, vocalist Denise Johnson and Ray Dorset (Mungo Jerry). Kings Have Long Arms released a single, "Big Umbrella", in January 2008, on Domino Records. It featured guest vocals from Candie Payne.'

words = tokenize_toolkit.run(sentence)
fn_result = fn_toolkit.run(words)

for entity in fn_result:
    print("Word:",entity["word"],"Position:",entity["position"],"Frame:",entity["frame"])

print("end")