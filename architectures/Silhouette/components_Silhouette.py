from collections import namedtuple

import torch
import torch.nn as nn
import brevitas
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat

from ..common import *


__all__ = [
    'SilhouetteExtractor',
    'SilhouetteEditor',
    'encode_policy',
    'SilhouetteSectionizer', 
    'SilhouettePredictor', 
    'SilhouetteConv2d'
]
        
        
        
class SilhouetteExtractor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pooling_layer, 
                 kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool = self._make_pooling_layer(pooling_layer)
        self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
        
    def _make_pooling_layer(self, pooling_layer):
        if pooling_layer == 'avg':
            return nn.AvgPool2d(kernel_size=2)
        elif pooling_layer == 'max':
            return nn.MaxPool2d(kernel_size=2)
        else:
            raise ImportError('Pooling layer type {0} is not supported'.format(pooling_layer))
            
            
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        a = x.detach()
        x = self.globalavgpool(x)
        y = x.squeeze()
        return (y, a)
    
    
    
class SilhouetteEditor(nn.Module):
    def __init__(self,
                 method,
                 **params):
        super().__init__()
        assert method in ['channelwise_maximization',
                          'topk_activation',
                          'topk_deviation_of_activation',
                          'random']
        self.method = method
        if self.method in ['topk_activation',
                           'topk_deviation_of_activation']:
            assert 'k' in params
            k = params['k']
            assert type(k) is int and k > 0
            self.k = k
            
            
    def forward(self, a):
        assert a.dim() == 4
        
        if self.method == 'channelwise_maximization':
            return torch.max(a, dim=1, keepdim=True).values
        elif self.method == 'topk_activation':
            a = torch.stack([torch.index_select(batch, dim=0, index=index) for batch, index 
                             in zip(a, torch.topk(torch.mean(a, dim=[2, 3]), k=self.k, dim=1).indices)])
            a = torch.mean(a, dim=1, keepdim=True)
            return a
        elif self.method == 'topk_deviation_of_activation':
            a = torch.stack([torch.index_select(batch, dim=0, index=index) for batch, index 
                             in zip(a, torch.topk(torch.std(a, dim=[2, 3]), k=self.k, dim=1).indices)])
            a = torch.mean(a, dim=1, keepdim=True)
            return a
        elif self.method == 'random':
            return torch.randn_like(torch.mean(a, dim=1, keepdim=True))
        
        
        
def encode_policy(policy: str):
    accum = 0
    
    nbits = []
    quantiles = [0.0]
        
    for token in policy.split(', '):
        (Nbit, Xpercentage) = token.split(' ')
        
        N = int(Nbit.replace('bit', '')); assert N > 0
        X = int(Xpercentage.replace('%', '')); assert X > 0
        
        accum += X
        nbits.append(N)
        quantiles.append(accum/100)

    assert accum == 100
    quantiles.pop() # remove last one: 1.0
    quantiles = torch.tensor(quantiles)
    
    return (nbits, quantiles)



class SilhouetteSectionizer(nn.Module):
    def __init__(self,
                 policy: str):
        super().__init__()
        self.nbits, self.quantiles = encode_policy(policy)
        
        
    def forward(self, a):
        assert a.dim() == 4
                
        masks = {nbit: [] for nbit in self.nbits}
        for i, batch in enumerate(a):
            quantile_values = torch.quantile(batch.flatten(), self.quantiles.to(batch.device), dim=0)
            for j, nbit in enumerate(self.nbits):
                if j == len(self.nbits) - 1:
                    masks[nbit].append(torch.ge(batch, quantile_values[j]))
                else:
                    masks[nbit].append(torch.ge(batch, quantile_values[j]) & torch.lt(batch, quantile_values[j + 1]))
        masks = {nbit: torch.stack(mask, dim=0) for nbit, mask in masks.items()}
        
        return masks

    
    
class SilhouettePredictor(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.extractor = SilhouetteExtractor(**config['extractor'])
        self.editor = SilhouetteEditor(**config['editor'])
        self.sectionizer = SilhouetteSectionizer(**config['sectionizer'])
        
        
    def forward(self, x):
        y, a = self.extractor(x)
        a = self.editor(a)
        m = self.sectionizer(a)
        return y, m
    
    

class SilhouetteConv2d(nn.Module):
    def __init__(self, sectioning_policy,
                 in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=None, weight_scaling_per_output_channel=False):
        super().__init__()
        self.nbits, _ = encode_policy(sectioning_policy)
        self.silhouettes = None
        self.conv_layers = self._make_conv_layers(in_channels, out_channels, kernel_size,
                                                  stride, padding, dilation, groups, bias, weight_scaling_per_output_channel)
        
    def _make_conv_layers(self, in_channels, out_channels, kernel_size,
                          stride, padding, dilation, groups, bias, weight_scaling_per_output_channel):
        conv_layers = nn.ModuleDict({})
        for nbit in self.nbits: # nbit is represented as int
            conv_layers[str(nbit)] = qnn.QuantConv2d(in_channels, out_channels, kernel_size,
                                                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                                     input_quant=Uint8ActPerTensorFloat, input_bit_width=nbit,
                                                     weight_bit_width=nbit, 
                                                     weight_scaling_per_output_channel=weight_scaling_per_output_channel)
        return conv_layers
        
            
    def update_silhouettes(self, silhouettes):
        self.silhouettes = silhouettes
    
    
    def _generate_masks(self, target_shape):
        if self.silhouettes is None: # empty tensor: no silhouette given
            return None
        else:   
            return {nbit: nn.functional.interpolate(m.float(), size=target_shape).bool()
                    for nbit, m in self.silhouettes.items()}
        
        
    def forward(self, x):
        masks = self._generate_masks(x.shape[-2:])
        
        if masks is None:
            return self.conv_layers[str(max(self.nbits))](x)
        else:
            y_quants = {}
            for nbit, conv in self.conv_layers.items():
                x_quant = x.masked_fill(~masks[int(nbit)], 1e-5)
                y_quants[nbit] = conv(x_quant)
            return sum(y_quants.values())