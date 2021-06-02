import typing as ty

import torch
import torch.nn as nn

import brevitas
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
import brevitas.nn as qnn

from ..common import *


__all__ = [
    'DRQPredictor',
    'DRQConv2d'
]
    

     
class DRQPredictor(nn.Module):
    def __init__(self, 
                 kernel_size,
                 threshold):
        super().__init__()
        if type(kernel_size) is list:
            self._kernel_size = tuple(kernel_size)
        else:
            self._kernel_size = kernel_size
        self.mean_filtering = nn.AvgPool2d(kernel_size=self._kernel_size)
        self.threshold = threshold
        
        
    def _forward_impl(self, 
                      x: torch.Tensor) -> torch.BoolTensor:
        input_shape = x.shape[:-2]
        if type(self._kernel_size) is float:
            if input_shape[0] > self._kernel_size and input_shape[1] > self._kernel_size:
                x = self.mean_filtering(x)
        elif type(self._kernel_size) is tuple:
            if input_shape[0] > self._kernel_size[0] and input_shape[1] > self._kernel_size[1]:
                x = self.mean_filtering(x)
    
        mask = x.ge(self.threshold)
        return mask.detach()
    
    
    def forward(self, 
                x: torch.Tensor) -> torch.BoolTensor:
        return self._forward_impl(x)
    
    
    
class DRQConv2d(nn.Module):
    def __init__(self,
                 predictor_kernel_size,
                 predictor_threshold,
                 weight_scaling_per_output_channel,
                 high_bit_width,
                 low_bit_width,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1, 
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=None):
        super().__init__()
        self.predictor = DRQPredictor(predictor_kernel_size, predictor_threshold)
        
        self.conv_high = qnn.QuantConv2d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                         input_quant=Uint8ActPerTensorFloat, input_bit_width=high_bit_width,
                                         weight_bit_width=high_bit_width, weight_scaling_per_output_channel=weight_scaling_per_output_channel)
        self.conv_low = qnn.QuantConv2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                        input_quant=Uint8ActPerTensorFloat, input_bit_width=low_bit_width,
                                        weight_bit_width=low_bit_width, weight_scaling_per_output_channel=weight_scaling_per_output_channel)
        
        
    def _generate_masks(self,
                        x: torch.Tensor) -> torch.Tensor:
        mask = self.predictor(x)
        mask = nn.functional.interpolate(mask.float(), size=x.shape[-2:]).bool()
        return mask
        
        
    def _forward_impl(self, 
                      x: torch.Tensor) -> torch.Tensor:
        mask = self._generate_masks(x)
        
        x_high = x.masked_fill(~mask, 1e-5)
        y_high = self.conv_high(x_high)

        x_low = x.masked_fill(mask, 1e-5)
        y_low = self.conv_low(x_low)

        return y_high + y_low
    
    
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)