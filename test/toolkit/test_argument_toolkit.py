from cogie.toolkit.fn.argument_toolkit import ArgumentToolkit
from cogie.toolkit.fn.fn_toolkit import FnToolkit
from cogie.toolkit.tokenize.tokenize_toolkit import TokenizeToolkit

fn_toolkit = FnToolkit(task='fn', language='english', corpus='frame')
argument_toolkit = ArgumentToolkit(task='fn', language='english', corpus='argument')
tokenize_toolkit = TokenizeToolkit()

sentence = 'All lanthanide elements form trivalent cations, Ln3+, whose chemistry is largely determined by the ionic radius, which decreases steadily from lanthanum to lutetium. They are termed as lanthanides because the lighter elements in the series are chemically similar to lanthanum. Strictly speaking, both lanthanum and lutetium have been labeled as group 3 elements, because they both have a single valelemence electron in the d shell.'

words = tokenize_toolkit.run(sentence)
frame = fn_toolkit.run(words)
argument_result = argument_toolkit.run(words,frame)

print("end")