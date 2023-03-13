import numpy as np
import math
import warnings
import copy
import torch.nn as nn
import torch.nn.functional as F
# local modules

from .model_util import CropParameters, recursive_clone
from .base.base_model import BaseModel

from .EReFormer import TransformerRecurrent
# from .submodules import ResidualBlock, ConvGRU, ConvLayer
# from utils.color_utils import merge_channels_into_color_image

# from .legacy import FireNet_legacy


def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
    
class E2DepthTransformerRecurrent(BaseModel):
   
    def __init__(self, EReFormer_kwargs):
        super().__init__(EReFormer_kwargs)
        # self.num_bins = EReFormer_kwargs['num_bins_events']  # legacy
        # self.num_decoders = EReFormer_kwargs['num_decoders']  # legacy
        self.transformerrecurrent = TransformerRecurrent(EReFormer_kwargs)

    @property
    def states(self):
        return {'copy_states_d':copy_states(self.transformerrecurrent.states)}

    @states.setter
    def states(self, states):
        self.transformerrecurrent.states = states

    def reset_states(self):
        
        self.transformerrecurrent.states = [None] * self.transformerrecurrent.num_decoders
        
    def init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        self.transformerrecurrent.SwinTransformer.init_weights()

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 depth prediction
        """
        output_dict = self.transformerrecurrent.forward(event_tensor)
        return output_dict


