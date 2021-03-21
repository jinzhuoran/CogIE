"""
@Author: jinzhuan
@File: __init__.py.py
@Desc: 
"""
from .bert_ner import *
from .bert_cn_ner import *
__all__ = [
    "Bert4Ner",
    "Bert4NerParallel",
    "Bert4CNNer",
    "Bert4CNNerParallel",
]
