"""
@Author: jinzhuan
@File: __init__.py.py
@Desc: 
"""
from .bert_crf import *
from .bert_softmax import *
__all__ = [
    "BertCRF",
    "BertCRFParallel",
    "BertSoftmax",
    "BertSoftmaxParallel",
]
