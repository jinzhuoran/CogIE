"""
@Author: jinzhuan
@File: __init__.py.py
@Desc: 
"""
from .crf import *
from .ffn import *
__all__ = [
    "ConditionalRandomField",
    "FeedForwardNetwork",
]
