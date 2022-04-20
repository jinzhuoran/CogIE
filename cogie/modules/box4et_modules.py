"""Pytorch modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Tuple

# Import custom modules
from ..utils.box_wrapper import BoxTensor, log1mexp
from ..utils.box_wrapper import CenterBoxTensor
from ..utils.box_wrapper import ConstantBoxTensor, CenterSigmoidBoxTensor


euler_gamma = 0.57721566490153286060


def _compute_gumbel_min_max(
  box1: BoxTensor,
  box2: BoxTensor,
  gumbel_beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns min and max points."""
  min_point = torch.stack([box1.z, box2.z])
  min_point = torch.max(
    gumbel_beta * torch.logsumexp(min_point / gumbel_beta, 0),
    torch.max(min_point, 0)[0])

  max_point = torch.stack([box1.Z, box2.Z])
  max_point = torch.min(
    -gumbel_beta * torch.logsumexp(-max_point / gumbel_beta, 0),
    torch.min(max_point, 0)[0])
  return min_point, max_point


def _compute_hard_min_max(
  box1: BoxTensor,
  box2: BoxTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns min and max points."""
  min_point = torch.max(box1.z, box2.z)
  max_point = torch.min(box1.Z, box2.Z)
  return min_point, max_point


class LinearProjection(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               bias: bool = True):
    super(LinearProjection, self).__init__()
    self.linear = nn.Linear(input_dim, output_dim, bias=bias)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    outputs = self.linear(inputs)
    return outputs


class SimpleFeedForwardLayer(nn.Module):
  """2-layer feed forward"""
  def __init__(self,
               input_dim: int,
               output_dim: int,
               bias: bool = True,
               activation: Optional[nn.Module] = None):
    super(SimpleFeedForwardLayer, self).__init__()
    self.linear_projection1 = nn.Linear(input_dim,
                                        (input_dim + output_dim) // 2,
                                        bias=bias)
    self.linear_projection2 = nn.Linear((input_dim + output_dim) // 2,
                                        output_dim,
                                        bias=bias)
    self.activation = activation if activation else nn.Sigmoid()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    inputs = self.activation(self.linear_projection1(inputs))
    inputs = self.activation(self.linear_projection2(inputs))
    return inputs


class HighwayNetwork(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               n_layers: int,
               activation: Optional[nn.Module] = None):
    super(HighwayNetwork, self).__init__()
    self.n_layers = n_layers
    self.nonlinear = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    self.gate = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    for layer in self.gate:
      layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
    self.final_linear_layer = nn.Linear(input_dim, output_dim)
    self.activation = nn.ReLU() if activation is None else activation
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    for layer_idx in range(self.n_layers):
      gate_values = self.sigmoid(self.gate[layer_idx](inputs))
      nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
      inputs = gate_values * nonlinear + (1. - gate_values) * inputs
    return self.final_linear_layer(inputs)


class TypeSelfAttentionLayer(nn.Module):
  def __init__(self,
               scale: float = 1.0,
               attn_dropout: float = 0.0):
    super(TypeSelfAttentionLayer, self).__init__()
    self.scale = scale
    self.dropout = nn.Dropout(attn_dropout)

  def forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    attn = torch.matmul(q, k.transpose(1, 2)) / self.scale
    if mask is not None:
      attn = attn.masked_fill(mask == 0, -1e9)
    attn = self.dropout(F.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)
    return output, attn


class SimpleDecoder(nn.Module):
  def __init__(self, output_dim: int, answer_num: int):
    super(SimpleDecoder, self).__init__()
    self.answer_num = answer_num
    self.linear = nn.Linear(output_dim, answer_num, bias=False)

  def forward(self,
              inputs: torch.Tensor) -> torch.Tensor:
    output_embed = self.linear(inputs)
    return output_embed


class BoxDecoder(nn.Module):

  box_types = {
    'BoxTensor': BoxTensor,
    'CenterBoxTensor': CenterBoxTensor,
    'ConstantBoxTensor': ConstantBoxTensor,
    'CenterSigmoidBoxTensor': CenterSigmoidBoxTensor
  }

  def __init__(self,
               num_embeddings: int,
               embedding_dim: int,
               box_type: str,
               padding_idx: Optional[int] = None,
               max_norm: Optional[float] = None,
               norm_type: float = 2.,
               scale_grad_by_freq: bool = False,
               sparse: bool = False,
               _weight: Optional[torch.Tensor] = None,
               init_interval_delta: float = 0.5,
               init_interval_center: float = 0.01,
               inv_softplus_temp: float = 1.,
               softplus_scale: float = 1.,
               n_negatives: int = 0,
               neg_temp: float = 0.,
               box_offset: float = 0.5,
               pretrained_box: Optional[torch.Tensor] = None,
               use_gumbel_baysian: bool = False,
               gumbel_beta: float = 1.0):
    super(BoxDecoder, self).__init__()

    self.num_embeddings = num_embeddings
    self.box_embedding_dim = embedding_dim
    self.box_type = box_type
    try:
      self.box = self.box_types[box_type]
    except KeyError as ke:
      raise ValueError("Invalid box type {}".format(box_type)) from ke
    self.box_offset = box_offset  # Used for constant tensor
    self.init_interval_delta = init_interval_delta
    self.init_interval_center = init_interval_center
    self.inv_softplus_temp = inv_softplus_temp
    self.softplus_scale = softplus_scale
    self.n_negatives = n_negatives
    self.neg_temp = neg_temp
    self.use_gumbel_baysian = use_gumbel_baysian
    self.gumbel_beta = gumbel_beta
    self.box_embeddings = nn.Embedding(num_embeddings,
                                       embedding_dim * 2,
                                       padding_idx=padding_idx,
                                       max_norm=max_norm,
                                       norm_type=norm_type,
                                       scale_grad_by_freq=scale_grad_by_freq,
                                       sparse=sparse,
                                       _weight=_weight)
    self.type_attention = TypeSelfAttentionLayer()  # not in use
    if pretrained_box is not None:
      print('Init box emb with pretrained boxes.')
      print(self.box_embeddings.weight)
      self.box_embeddings.weight = nn.Parameter(pretrained_box)
      print(self.box_embeddings.weight)

  def init_weights(self):
    print('before', self.box_embeddings.weight)
    torch.nn.init.uniform_(
      self.box_embeddings.weight[..., :self.box_embedding_dim],
      -self.init_interval_center, self.init_interval_center)
    torch.nn.init.uniform_(
      self.box_embeddings.weight[..., self.box_embedding_dim:],
      self.init_interval_delta, self.init_interval_delta)
    print('after', self.box_embeddings.weight)

  def log_soft_volume(
    self,
    z: torch.Tensor,
    Z: torch.Tensor,
    temp: float = 1.,
    scale: float = 1.,
    gumbel_beta: float = 0.) -> torch.Tensor:
    eps = torch.finfo(z.dtype).tiny  # type: ignore

    if isinstance(scale, float):
      s = torch.tensor(scale)
    else:
      s = scale

    if gumbel_beta <= 0.:
      return (torch.sum(
        torch.log(F.softplus(Z - z, beta=temp).clamp_min(eps)),
        dim=-1) + torch.log(s)
              )  # need this eps to that the derivative of log does not blow
    else:
      return (torch.sum(
        torch.log(
          F.softplus(Z - z - 2 * euler_gamma * gumbel_beta, beta=temp).clamp_min(
            eps)),
          dim=-1) + torch.log(s))

  def type_box_volume(self) -> torch.Tensor:
    inputs = torch.arange(0,
                          self.box_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)
    emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim
    if self.box_type == 'ConstantBoxTensor':
      type_box = self.box.from_split(emb, self.box_offset)
    else:
      type_box = self.box.from_split(emb)

    vol = self.log_soft_volume(type_box.z,
                               type_box.Z,
                               temp=self.inv_softplus_temp,
                               scale=self.softplus_scale,
                               gumbel_beta=self.gumbel_beta)
    return vol

  def get_pairwise_conditional_prob(self,
                                    type_x_ids: torch.Tensor,
                                    type_y_ids: torch.Tensor) -> torch.Tensor:
    inputs = torch.arange(0,
                          self.box_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)
    emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim
    type_x = emb[type_x_ids]
    type_y = emb[type_y_ids]
    type_x_box = self.box.from_split(type_x)
    type_y_box = self.box.from_split(type_y)

    # Compute intersection volume
    if self.use_gumbel_baysian:
      # Gumbel intersection
      min_point, max_point = _compute_gumbel_min_max(type_x_box,
                                                     type_y_box,
                                                     self.gumbel_beta)
    else:
      min_point, max_point = _compute_hard_min_max(type_x_box, type_y_box)

    intersection_vol = self.log_soft_volume(min_point,
                                            max_point,
                                            temp=self.inv_softplus_temp,
                                            scale=self.softplus_scale,
                                            gumbel_beta=self.gumbel_beta)
    # Compute y volume here
    y_vol = self.log_soft_volume(type_y_box.z,
                                 type_y_box.Z,
                                 temp=self.inv_softplus_temp,
                                 scale=self.softplus_scale,
                                 gumbel_beta=self.gumbel_beta)

    # Need to be careful about numerical issues
    conditional_prob = intersection_vol - y_vol
    return torch.cat([conditional_prob.unsqueeze(-1),
                      log1mexp(conditional_prob).unsqueeze(-1)],
                     dim=-1)


  def forward(
    self,
    mc_box: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    is_training: bool = True,
    batch_num: Optional[int] = None
  ) -> Tuple[torch.Tensor, None]:
    inputs = torch.arange(0,
                          self.box_embeddings.num_embeddings,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)
    emb = self.box_embeddings(inputs)  # num types x 2*box_embedding_dim

    if self.box_type == 'ConstantBoxTensor':
      type_box = self.box.from_split(emb, self.box_offset)
    else:
      type_box = self.box.from_split(emb)

    # Get intersection
    batch_size = mc_box.data.size()[0]
    # Expand both mention&context and type boxes to the shape of batch_size x
    # num_types x box_embedding_dim. (torch.expand doesn't use extra memory.)
    if self.use_gumbel_baysian:  # Gumbel box
      min_point = torch.stack(
        [mc_box.z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
         type_box.z.unsqueeze(0).expand(batch_size, -1, -1)])
      min_point = torch.max(
        self.gumbel_beta * torch.logsumexp(min_point / self.gumbel_beta, 0),
        torch.max(min_point, 0)[0])

      max_point = torch.stack([
        mc_box.Z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
        type_box.Z.unsqueeze(0).expand(batch_size, -1, -1)])
      max_point = torch.min(
        -self.gumbel_beta * torch.logsumexp(-max_point / self.gumbel_beta, 0),
        torch.min(max_point, 0)[0])

    else:
      min_point = torch.max(
        torch.stack([
          mc_box.z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
          type_box.z.unsqueeze(0).expand(batch_size, -1, -1)]), 0)[0]

      max_point = torch.min(
        torch.stack([
          mc_box.Z.unsqueeze(1).expand(-1, self.num_embeddings, -1),
          type_box.Z.unsqueeze(0).expand(batch_size, -1, -1)]), 0)[0]

    # Get soft volume
    # batch_size x num types
    # Compute the volume of the intersection
    vol1 = self.log_soft_volume(min_point,
                                max_point,
                                temp=self.inv_softplus_temp,
                                scale=self.softplus_scale,
                                gumbel_beta=self.gumbel_beta)

    # Compute  the volume of the mention&context box
    vol2 = self.log_soft_volume(mc_box.z,
                                mc_box.Z,
                                temp=self.inv_softplus_temp,
                                scale=self.softplus_scale,
                                gumbel_beta=self.gumbel_beta)

    # Returns log probs
    log_probs = vol1 - vol2.unsqueeze(-1)

    # Clip values > 1. for numerical stability.
    if (log_probs > 0.0).any():
      print("WARNING: Clipping log probability since it's grater than 0.")
      log_probs[log_probs > 0.0] = 0.0

    if is_training and targets is not None and self.n_negatives > 0:
      pos_idx = torch.where(targets.sum(dim=0) > 0.)[0]
      neg_idx = torch.where(targets.sum(dim=0) == 0.)[0]

      if self.n_negatives < neg_idx.size()[0]:
        neg_idx = neg_idx[torch.randperm(len(neg_idx))[:self.n_negatives]]
        log_probs_pos = log_probs[:, pos_idx]
        log_probs_neg = log_probs[:, neg_idx]
        _log_probs = torch.cat([log_probs_pos, log_probs_neg], dim=-1)
        _targets = torch.cat([targets[:, pos_idx], targets[:, neg_idx]], dim=-1)
        _weights = None
        if self.neg_temp > 0.0:
          _neg_logits = log_probs_neg - log1mexp(log_probs_neg)
          _neg_weights = F.softmax(_neg_logits * self.neg_temp, dim=-1)
          _pos_weights = torch.ones_like(log_probs_pos,
                                         device=self.box_embeddings.weight.device)
          _weights = torch.cat([_pos_weights, _neg_weights], dim=-1)
        return _log_probs, _weights, _targets
      else:
        return log_probs, None, targets
    elif is_training and targets is not None and self.n_negatives <= 0:
      return log_probs, None, targets
    else:
      return log_probs, None, None


class BCEWithLogProbLoss(nn.BCELoss):

  def _binary_cross_entropy(self,
                            input: torch.Tensor,
                            target: torch.Tensor,
                            weight: Optional[torch.Tensor] = None,
                            reduction: str = 'mean') -> torch.Tensor:
    """Computes binary cross entropy.

    This function takes log probability and computes binary cross entropy.

    Args:
      input: Torch float tensor. Log probability. Same shape as `target`.
      target: Torch float tensor. Binary labels. Same shape as `input`.
      weight: Torch float tensor. Scaling loss if this is specified.
      reduction: Reduction method. 'mean' by default.
    """
    loss = -target * input - (1 - target) * log1mexp(input)

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()

  def forward(self, input, target, weight=None):
    return self._binary_cross_entropy(input,
                                      target,
                                      weight=weight,
                                      reduction=self.reduction)
