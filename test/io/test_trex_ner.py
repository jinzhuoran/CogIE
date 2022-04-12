import cogie
from cogie.io.loader.rc.trex import TrexRelationLoader
from cogie.io.loader.ner.trex_ner import TrexNerLoader

loader = TrexNerLoader()
train_data, dev_data, test_data  = loader.load_all('../../../cognlp/data/ner/trex/data/trex_debug.json')


