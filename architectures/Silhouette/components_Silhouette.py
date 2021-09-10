from collections import namedtuple
from typing import Union, Optional

import torch
import torch.nn as nn
import brevitas
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL

from ..common import *


__all__ = [
    'SilhouetteReLU',
    'QuantSilhouetteReLU'
]



class SilhouetteActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, theta: nn.Parameter):
        mask = torch.ge(torch.std(x, dim=[2, 3]), theta)
        x[~mask] = 0
        ctx.save_for_backward(mask, theta)
        return x
    
    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        mask, theta, = ctx.saved_tensors
        dx = dy.clone()
        dx[~mask] = 0
        dtheta = torch.mean(dx)
        return dx, dtheta
    
    
    
class SilhouetteReLU(nn.Module):
    def __init__(self,
                 theta: float = 0.1):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta, requires_grad=True))
        
    def forward(self, x: torch.Tensor):
        return SilhouetteActivation.apply(x, self.theta)
    
    
    
class QuantSilhouetteReLU(QuantNLAL):
    def __init__(self,
                 input_quant=None,
                 act_quant=Uint8ActPerTensorFloat,
                 return_quant_tensor=False,
                 **kwargs
                ):
        QuantNLAL.__init__(
            self,
            act_impl=SilhouetteReLU,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs
        )