"""
@Author: jinzhuan
@File: loss.py
@Desc: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


class DiceLoss(nn.Module):
    def __init__(self,
                 smooth: Optional[float] = 1e-8,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 reduction: Optional[str] = "mean") -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self,
                input: Tensor,
                target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:

        flat_input = input.view(-1)
        flat_target = target.view(-1)

        if self.with_logits:
            flat_input = torch.sigmoid(flat_input)

        if mask is not None:
            mask = mask.view(-1).float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask

        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            return 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            return 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input,), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}"


class AdaptiveDiceLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.1,
                 smooth: Optional[float] = 1e-8,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 reduction: Optional[str] = "mean") -> None:
        super(AdaptiveDiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.alpha = alpha
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self,
                input: Tensor,
                target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:

        flat_input = input.view(-1)
        flat_target = target.view(-1)

        if self.with_logits:
            flat_input = torch.sigmoid(flat_input)

        if mask is not None:
            mask = mask.view(-1).float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask

        intersection = torch.sum((1-flat_input)**self.alpha * flat_input * flat_target, -1) + self.smooth
        denominator = torch.sum((1-flat_input)**self.alpha * flat_input) + flat_target.sum() + self.smooth
        return 1 - 2 * intersection / denominator

    def __str__(self):
        return f"Adaptive Dice Loss, smooth:{self.smooth}; alpha:{self.alpha}"
