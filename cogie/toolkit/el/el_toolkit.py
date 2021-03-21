"""
@Author: jinzhuan
@File: el_toolkit.py
@Desc: 
"""
from cogie import *
import threading
from ..base_toolkit import BaseToolkit
# import blink.predictor as predictor
from cogie.utils.cognet import CognetServer
from cogie.utils.util import get_all_forms


class ElToolkit(BaseToolkit):

    def __init__(self, task='el', language='english', corpus=None):
        super().__init__()
        self.task = task
        self.language = language
        self.corpus = corpus
        config = load_configuration()
        download_model(config[task]['cognet'])
        path = config['el']['cognet']['path']
        file = config['el']['cognet']['data']['file']
        self.wikidata2wikipedia = load_json(absolute_path(path, file))
        self.cognet = CognetServer()
        # self.id2url, self.et_ner_model, self.et_models = predictor.get_et_predictor()

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(ElToolkit, "_instance"):
            with ElToolkit._instance_lock:
                if not hasattr(ElToolkit, "_instance"):
                    ElToolkit._instance = object.__new__(cls)
        return ElToolkit._instance

    def run(self, sentence):
        url = "https://en.wikipedia.org/wiki/"
        # links = predictor.run(10, *self.et_models, text=sentence, id2url=self.id2url, ner_model=self.et_ner_model)
        # for link in links:
        #     forms = get_all_forms(link["title"])
        #     cognet_link = "unk"
        #     for form in forms:
        #         wikipedia = url + form
        #         if wikipedia in self.wikidata2wikipedia:
        #             wikidata = self.wikidata2wikipedia[wikipedia]
        #             cognet_link = self.cognet.query("<" + wikidata + ">")
        #     link["cognet_link"] = cognet_link
        # return links
