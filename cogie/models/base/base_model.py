"""
@Author: jinzhuan
@File: base_model.py
@Desc: 
"""
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def loss(self, batch, loss_function):
        pass

    def evaluate(self, batch, metrics):
        pass

    def predict(self, batch):
        pass
