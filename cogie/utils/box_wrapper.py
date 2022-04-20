"""
This code is based on the repo of "Improving Local Identifiability in
Probabilistic Box Embeddings" by Dasgupta et al., (2020).

  Paper: https://arxiv.org/pdf/2010.04831.pdf
  Code : https://github.com/iesl/gumbel-box-embeddings

"""

import math
import torch
import torch.nn.functional as F

from scipy import special
from torch import Tensor
from typing import Tuple, Union, Type, TypeVar

tanh_eps = 1e-20
_log1mexp_switch = math.log(0.5)


def log1mexp(x: torch.Tensor,
             split_point=_log1mexp_switch,
             exp_zero_eps=1e-7) -> torch.Tensor:
  """
  Computes log(1 - exp(x)).

  Splits at x=log(1/2) for x in (-inf, 0] i.e. at -x=log(2) for -x in [0, inf).

  = log1p(-exp(x)) when x <= log(1/2)
  or
  = log(-expm1(x)) when log(1/2) < x <= 0

  For details, see

  https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

  https://github.com/visinf/n3net/commit/31968bd49c7d638cef5f5656eb62793c46b41d76
  """
  logexpm1_switch = x > split_point
  Z = torch.zeros_like(x)
  # this clamp is necessary because expm1(log_p) will give zero when log_p=1,
  # ie. p=1
  logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))
  # hack the backward pass
  # if expm1(x) gets very close to zero, then the grad log() will produce inf
  # and inf*0 = nan. Hence clip the grad so that it does not produce inf
  logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
  Z[logexpm1_switch] = logexpm1.detach() + (logexpm1_bw - logexpm1_bw.detach())
  #Z[1 - logexpm1_switch] = torch.log1p(-torch.exp(x[1 - logexpm1_switch]))
  Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))

  return Z


def log1pexp(x: torch.Tensor) -> torch.Tensor:
  """ Computes log(1+exp(x))

  see: Page 7, eqn 10 of https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  also see: https://github.com/SurajGupta/r-source/blob/master/src/nmath/plogis.c
  """
  Z = torch.zeros_like(x)
  zone1 = (x <= 18.)
  zone2 = (x > 18.) * (x < 33.3)  # And operator using *
  zone3 = (x >= 33.3)
  Z[zone1] = torch.log1p(torch.exp(x[zone1]))
  Z[zone2] = x[zone2] + torch.exp(-(x[zone2]))
  Z[zone3] = x[zone3]

  return Z


def _box_shape_ok(t: Tensor) -> bool:
  if len(t.shape) < 2:
    return False
  else:
    if t.size(-2) != 2:
      return False

    return True


def _shape_error_str(tensor_name, expected_shape, actual_shape):
  return "Shape of {} has to be {} but is {}".format(tensor_name,
                             expected_shape,
                             tuple(actual_shape))


class ExpEi(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    dev = input.device
    with torch.no_grad():
      x = special.exp1(input.detach().cpu()).to(dev)
      input.to(dev)
    return x

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output*(-torch.exp(-input)/input)
    return grad_input


# see: https://realpython.com/python-type-checking/#type-hints-for-methods
# to know why we need to use TypeVar
TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")


class BoxTensor(object):
  """ A wrapper to which contains single tensor which
  represents single or multiple boxes.

  Have to use composition instead of inheritance because
  it is not safe to interit from :class:`torch.Tensor` because
  creating an instance of such a class will always make it a leaf node.
  This works for :class:`torch.nn.Parameter` but won't work for a general
  box_tensor.
  """

  def __init__(self, data: Tensor) -> None:
    """
    Arguments:
      data: Tensor of shape (**, zZ, num_dims). Here, zZ=2, where
        the 0th dim is for bottom left corner and 1st dim is for
        top right corner of the box
    """

    if _box_shape_ok(data):
      self.data = data
    else:
      raise ValueError(_shape_error_str('data', '(**,2,num_dims)', data.shape))
    super().__init__()

  def __repr__(self):
    return 'box_tensor_wrapper(' + self.data.__repr__() + ')'

  @property
  def z(self) -> Tensor:
    """Lower left coordinate as Tensor"""

    return self.data[..., 0, :]

  @property
  def Z(self) -> Tensor:
    """Top right coordinate as Tensor"""

    return self.data[..., 1, :]

  @property
  def centre(self) -> Tensor:
    """Centre coordinate as Tensor"""

    return (self.z + self.Z)/2

  @classmethod
  def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> TBoxTensor:
    """
    Creates a box by stacking z and Z along -2 dim.
    That is if z.shape == Z.shape == (**, num_dim),
    then the result would be box of shape (**, 2, num_dim)
    """

    if z.shape != Z.shape:
      raise ValueError(
        "Shape of z and Z should be same but is {} and {}".format(
          z.shape, Z.shape))
    box_val: Tensor = torch.stack((z, Z), -2)

    return cls(box_val)

  @classmethod
  def from_split(cls: Type[TBoxTensor], t: Tensor,
           dim: int = -1) -> TBoxTensor:
    """Creates a BoxTensor by splitting on the dimension dim at midpoint

    Args:
      t: input
      dim: dimension to split on

    Returns:
      BoxTensor: output BoxTensor

    Raises:
      ValueError: `dim` has to be even
    """
    len_dim = t.size(dim)

    if len_dim % 2 != 0:
      raise ValueError(
        "dim has to be even to split on it but is {}".format(
          t.size(dim)))
    split_point = int(len_dim / 2)
    z = t.index_select(
      dim,
      torch.tensor(
        list(range(split_point)), dtype=torch.int64, device=t.device))

    Z = t.index_select(
      dim,
      torch.tensor(
        list(range(split_point, len_dim)),
        dtype=torch.int64,
        device=t.device))

    return cls.from_zZ(z, Z)


class CenterBoxTensor(BoxTensor):

  @property
  def center(self) -> Tensor:
    return self.data[..., 0, :]

  @property
  def z(self) -> Tensor:
    #return self.data[..., 0, :] - torch.sigmoid(self.data[..., 1, :])
    return self.data[..., 0, :] \
         - torch.nn.functional.softplus(self.data[..., 1, :], beta=10.)

  @property
  def Z(self) -> Tensor:
    #return self.data[..., 0, :] + torch.sigmoid(self.data[..., 1, :])
    return self.data[..., 0, :] \
         + torch.nn.functional.softplus(self.data[..., 1, :], beta=10.)


class CenterSigmoidBoxTensor(BoxTensor):

  @property
  def center(self) -> Tensor:
    return self.data[..., 0, :]

  @property
  def z(self) -> Tensor:
    z = self.data[..., 0, :] \
      - torch.nn.functional.softplus(self.data[..., 1, :], beta=10.)
    return torch.sigmoid(z)

  @property
  def Z(self) -> Tensor:
    Z = self.data[..., 0, :] \
      + torch.nn.functional.softplus(self.data[..., 1, :], beta=10.)
    return torch.sigmoid(Z)


class ConstantBoxTensor(BoxTensor):

  def __init__(self, data: Tensor, box_offset: float = 0.5) -> None:

    if _box_shape_ok(data):
      self.data = data
    else:
      raise ValueError(
        _shape_error_str('data', '(**,2,num_dims)', data.shape))
    super(ConstantBoxTensor, self).__init__(data)
    self.box_offset = box_offset

  @property
  def center(self) -> Tensor:
    return self.data[..., 0, :]

  @property
  def z(self) -> Tensor:
    return self.data[..., 0, :] - torch.ones_like(self.data[..., 0, :]) * self.box_offset

  @property
  def Z(self) -> Tensor:
    return self.data[..., 0, :] + torch.ones_like(self.data[..., 0, :]) * self.box_offset

  @classmethod
  def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor, box_offset: float) -> TBoxTensor:
    """
    Creates a box by stacking z and Z along -2 dim.
    That is if z.shape == Z.shape == (**, num_dim),
    then the result would be box of shape (**, 2, num_dim)
    """

    if z.shape != Z.shape:
      raise ValueError(
        "Shape of z and Z should be same but is {} and {}".format(
          z.shape, Z.shape))
    box_val: Tensor = torch.stack((z, Z), -2)

    return cls(box_val, box_offset=box_offset)

  @classmethod
  def from_split(cls: Type[TBoxTensor], t: Tensor, box_offset: float,
           dim: int = -1) -> TBoxTensor:
    """Creates a BoxTensor by splitting on the dimension dim at midpoint

    Args:
      t: input
      dim: dimension to split on

    Returns:
      BoxTensor: output BoxTensor

    Raises:
      ValueError: `dim` has to be even
    """
    len_dim = t.size(dim)

    if len_dim % 2 != 0:
      raise ValueError(
        "dim has to be even to split on it but is {}".format(
          t.size(dim)))
    split_point = int(len_dim / 2)
    z = t.index_select(
      dim,
      torch.tensor(
        list(range(split_point)), dtype=torch.int64, device=t.device))

    Z = t.index_select(
      dim,
      torch.tensor(
        list(range(split_point, len_dim)),
        dtype=torch.int64,
        device=t.device))

    return cls.from_zZ(z, Z, box_offset)