"""
@Author: jinzhuan
@File: base_function.py
@Desc: 
"""
class BaseFunction:
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        pass

    def loss(self, batch, loss_function):
        pass

    def evaluate(self, batch, metrics):
        pass

    def predict(self, batch):
        pass
