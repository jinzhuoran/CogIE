"""
@Author: jinzhuan
@File: lstm.py
@Desc: 
"""
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size=50, hidden_size=256, dropout=0, bidirectional=False, num_layers=1, activation_function="tanh"):
        super().__init__()
        if bidirectional:
            hidden_size /= 2
        self.lstm = nn.LSTM(input_size,
                          hidden_size,
                          num_layers,
                          nonlinearity=activation_function,
                          dropout=dropout,
                          bidirectional=bidirectional)

    def forward(self, x):
        # Check size of tensors
        x = x.transpose(0, 1) # (L, B, I_EMBED)
        x, h, c = self.lstm(x) # (L, B, H_EMBED)
        x = x.transpose(0, 1) # (B, L, I_EMBED)
        return x
