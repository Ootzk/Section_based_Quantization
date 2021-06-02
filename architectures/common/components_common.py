import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.inject.defaults import Uint8ActPerTensorFloat
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL



class PACTClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.clamp(min=0.0).min(alpha)
    
    @staticmethod
    def backward(ctx, dLdy):
        x, alpha = ctx.saved_variables
        
        lt0      = x < 0
        gta      = x > alpha
        gi       = 1.0-lt0.float()-gta.float()
        
        dLdx     = dLdy*gi
        dLdalpha = torch.sum(dLdy*x.ge(alpha).float())
        return dLdx, dLdalpha
    
    

class PACTReLU(nn.Module):
    def __init__(self, alpha=6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, x):
        return PACTClip.apply(x, self.alpha)
    
    
    
class QuantPACTReLU(QuantNLAL):
    def __init__(self,
                 input_quant=None,
                 act_quant=Uint8ActPerTensorFloat,
                 return_quant_tensor=False,
                 **kwargs
                ):
        QuantNLAL.__init__(
            self,
            act_impl=PACTReLU,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs
        )
        

"""
class IdentityPadding(nn.Module):
	def __init__(self, in_channels, out_channels, stride):
		super().__init__()
		
		self.pooling = nn.MaxPool2d(1, stride=stride)
		self.add_channels = out_channels - in_channels
    
	def forward(self, x):
		out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
		out = self.pooling(out)
		return out
"""