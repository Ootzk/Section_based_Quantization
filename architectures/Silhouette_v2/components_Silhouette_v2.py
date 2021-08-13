from collections import namedtuple
from typing import Union

import torch
import torch.nn as nn
import brevitas
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat

from ..common import *


__all__ = [
    'AttentionMapExtractor',
    'compress_attention_map',
    'AuxiliaryNet',
    'encode_policy',
    'sectionize_silhouette',
    'SilhouetteConv2d'
]
        
        
        
class AttentionMapExtractor(nn.Module):
    """
    This module will be 'installed' at the specific location of Original network, such as ResNet, MobileNet, etc.
    Feature map at that specific location will be processed into 'Attention map', by passing simple Conv-Pool-GlobalAvgPool layer combination.
    
    Good explanation for 'Attention Map': https://blog.lunit.io/2018/08/30/bam-and-cbam-self-attention-modules-for-cnn/
    
    Args:
        in_channels (int): # channels of Feature map that will be extracted.
        out_channels (int): # classes of target training dataset. ex) CIFAR10: 10, ImageNet: 1000
        conv_kernel_size (int, default=3): conv layer kernel size.
        
        pool_type (str, default='MaxPool2d'): type of pooling layer. supports ['AvgPool2d', 'MaxPool2d']
        pool_kernel_size (int, default=2): pool layer kernel size.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 conv_kernel_size: int = 3,
                 
                 pool_type: str = 'MaxPool2d',
                 pool_kernel_size: int = 2) -> None:
        super().__init__()
        
        assert pool_type in ['AvgPool2d', 'MaxPool2d']
        
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size)
        self.pool = nn.__dict__[pool_type](pool_kernel_size)
        self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
            
            
    def forward(self, x: torch.Tensor) -> tuple:
        assert x.dim() == 4
        
        x = self.conv(x)
        x = self.pool(x)
        a = x.detach()
        x = self.globalavgpool(x)
        y = x.squeeze()
        
        out = (y, a)
        return out
    
    
    
def compress_attention_map(a: torch.Tensor, method: str, **kwargs) -> torch.Tensor:
    """
    Compress attention map(extracted by AttentionMapExtractor)(size: B * C * H * W) into `silhouette`(size: B * 1 * H * W) with given method.
    
    Args:
        a (torch.Tensor): attention map extracted by AttentionMapExtractor.
        method (str): compressing method. supports ['amax', 'avg', 'topk_activation', 'topk_deviation_of_activation', 'random']
        
    Returns:
        silhouette (torch.Tensor): compressed attention map with channel size = 1.
    """
    
    assert isinstance(a, torch.Tensor) and a.dim() == 4
    assert method in ['max', 'avg', 'topk_activation', 'topk_deviation_of_activation', 'random']
    if method == 'topk_activation' or method == 'topk_deviation_of_activation':
        assert 'k' in kwargs.keys() and isinstance(kwargs['k'], int)
        assert kwargs['k'] > 0 and kwargs['k'] < a.size()[1]
        
    if method == 'max':
        return torch.amax(a, dim=1, keepdim=True)
    elif method == 'avg':
        return torch.mean(a, dim=1, keepdim=True)
    elif method == 'topk_activation':
        indices  = torch.topk(torch.mean(a, dim=[2, 3]), k=kwargs['k'], dim=1).indices
        selected = torch.stack([torch.index_select(batch, dim=0, index=index)
                                for batch, index in zip(a, indices)])
        return torch.mean(selected, dim=1, keepdim=True)
    elif method == 'topk_deviation_of_activation':
        indices  = torch.topk(torch.std(a, dim=[2, 3]), k=kwargs['k'], dim=1).indices
        selected = torch.stack([torch.index_select(batch, dim=0, index=index) 
                                for batch, index in zip(a, indices)])
        return torch.mean(selected, dim=1, keepdim=True)
    elif method == 'random':
        return torch.randn_like(torch.mean(a, dim=1, keepdim=True))
    
    
    
class AuxiliaryNet(nn.Module):
    """
    Auxiliary Network for Silhouette algorithm.
    Extract and process feature map from Original network, and keep `Silhouette`(compressed attention map) in bank for global usage of Original network.
    
    Args:
        extractor_config (dict): arguments in dict for AttentionMapExtractor.
        compress_config  (dict): arguments in dict for compress_attention_map.
        
    Returns:
        y (torch.Tensor): prediction of internal AttentionMapExtractor. size: B * nClasses.
    """
    def __init__(self,
                 extractor_config: dict, 
                 compress_config: dict) -> None:
        super().__init__()
        
        self.extractor = AttentionMapExtractor(**extractor_config)
        self.compress_config = compress_config
        
        self.register_buffer('silhouette', None, persistent=False)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, a = self.extractor(x)
        self.silhouette = compress_attention_map(a, **self.compress_config)
        return y
    
    
    
def encode_policy(policy: str):
    """
    Translate human-friendly sectioning policy to computer-friendly combination of lists.
    
    Examples:
        '2bit 50%, 3bit 30%, 4bit 20%' -> ([2, 3, 4], [0.0, 0.5, 0.8])
        
    Args:
        policy (str): human-friendly sectioning policy. percentiles in front will be assigned to low-value spectrum.
        
    Returns:
        tuple that includes...
            nbits (list): target precisions.
            quantiles (list): threshold quantile for each nbit in nbits.
    """
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



class Silhouette_v2_Conv2d(nn.Module):
    def __init__(self,
                 threshold: int, 
                
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 
                 weight_scaling_per_output_channel: bool = False,
                 interpolate_before_sectioning: bool = False):
        super().__init__()
        self.threshold = threshold
        
        
    def forward(self, x: torch.Tensor):
        mask = torch.mean(x, dim=[2, 3]).ge(self.threshold)
        
        
        
        
        for i, (batch, batch_mask) in enumerate(zip(x, mask)):
            for j, (channel, boolean) in enumerate(zip(batch, batch_mask)):
                
                
                print(f'one channel\'s size: {channel.shape}, corresponding boolean: {boolean}')
        
        
        
        return mask