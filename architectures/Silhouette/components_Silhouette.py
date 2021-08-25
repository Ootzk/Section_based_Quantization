from collections import namedtuple
from typing import Union, Optional

import torch
import torch.nn as nn
import brevitas
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat

from ..common import *


__all__ = [
    'SilhouetteConv2d'
]
  
    
    
class SilhouetteConv2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 
                 threshold: float = 0.1,
                 activation_bit_width: int = 8,
                 weight_bit_width: int = 8,
                 weight_scaling_per_output_channel: bool = False,
                 interpolate_before_sectioning: bool = False):
        super().__init__()
        self.interpolate_before_sectioning = interpolate_before_sectioning
        self.threshold = nn.Parameter(torch.tensor(threshold, requires_grad=True))
        
        self.conv_layer = qnn.QuantConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                          
                                          input_quant=Uint8ActPerTensorFloat, # generally, input_quant is deactivated
                                          input_bit_width=activation_bit_width,
                                          
                                          weight_bit_width=weight_bit_width, 
                                          weight_scaling_per_output_channel=weight_scaling_per_output_channel)
    
    
    def forward(self, x: torch.Tensor, debug: bool = False):
        assert x.dim() == 4
        std = torch.std(x, dim=[2, 3])
        mask = torch.ge(std, self.threshold)
        mask = mask.unsqueeze(2).unsqueeze(3)
        x = x.masked_fill(~mask, 0)
        y = self.conv_layer(x)
        if debug:
            return y, x
        return y