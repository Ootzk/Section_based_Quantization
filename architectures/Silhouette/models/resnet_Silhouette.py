from typing import Union

import torch
import torch.nn as nn
import brevitas
import brevitas.nn as qnn

from ..components_Silhouette import *
from ...common import QuantPACTReLU


__all__ = [
    'ResNet_Silhouette',
    'resnet18_Silhouette',
    'resnet34_Silhouette',
    'resnet50_Silhouette'
]


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self,
                 inplanes: int, 
                 planes: int,
                 stride: int=1,
                 **kwargs):
        super().__init__()
        ConvLayer = SilhouetteConv2d if 'sectioning_policy' in kwargs else qnn.QuantConv2d
        activation_bit_width = kwargs['activation_bit_width'] if 'activation_bit_width' in kwargs else None
        
        self.conv1 = ConvLayer(**kwargs, in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if activation_bit_width is None:
            activation_bit_width = self.conv1.activation_bit_width
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = QuantPACTReLU(bit_width=activation_bit_width)
        
        self.conv2 = ConvLayer(**kwargs, in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential() # identity
        if stride != 1: # not identity
            self.shortcut = nn.Sequential(
                ConvLayer(**kwargs, in_channels=inplanes, out_channels=planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.relu2 = QuantPACTReLU(bit_width=activation_bit_width)
            
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        
        return out
    
    
    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4
    
    def __init__(self,
                 inplanes: int, 
                 planes: int,
                 stride: int=1,
                 **kwargs):
        super().__init__()
        ConvLayer = SilhouetteConv2d if 'sectioning_policy' in kwargs else qnn.QuantConv2d
        activation_bit_width = kwargs['activation_bit_width'] if 'activation_bit_width' in kwargs else None
        
        self.conv1 = ConvLayer(**kwargs, in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, padding=1, bias=False)
        if activation_bit_width is None:
            activation_bit_width = self.conv1.activation_bit_width
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = QuantPACTReLU(bit_width=activation_bit_width)
        
        self.conv2 = ConvLayer(**kwargs, in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = QuantPACTReLU(bit_width=activation_bit_width)
        
        self.conv3 = ConvLayer(**kwargs, in_channels=planes, out_channels=planes * self.expansion, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential() # identity
        if stride != 1: # not identity
            self.shortcut = nn.Sequential(
                ConvLayer(**kwargs, in_channels=inplanes, out_channels=planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        self.relu3 = QuantPACTReLU(bit_width=activation_bit_width)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = self.relu3(out)
        
        return out



class ResNet_Silhouette(nn.Module):
    def __init__(self, 
                 block: Union[BasicBlock, Bottleneck],
                 num_blocks: list,
                 target_dataset: str,
                 config: dict):
        super().__init__()
        
        assert len(num_blocks) == 4
        assert target_dataset in ['CIFAR10', 'CIFAR100', 'ImageNet']
        
        self.target_dataset = target_dataset
        self.config = config
        
        self.inplanes = 64
        
        self.input_stem = self._make_input_stem()
        self.layer1 = self._make_layer("layer1", block, num_blocks[0], 64, 1)
        self.layer2 = self._make_layer("layer2", block, num_blocks[1], 128, 2)
        self.layer3 = self._make_layer("layer3", block, num_blocks[2], 256, 2)
        self.layer4 = self._make_layer("layer4", block, num_blocks[3], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)
        
        self.w = nn.parameter.Parameter(data=torch.tensor([3.0]), requires_grad=True) # sigmoid(3.0) ~= 0.95
        self.register_buffer('sum_aux_nets_outputs', None)
        
        self.aux_nets = self._make_aux_nets()
        
        self._initialize_layers()
        
        
    @property
    def num_classes(self):
        if self.target_dataset == 'CIFAR10':
            return 10
        elif self.target_dataset == 'CIFAR100':
            return 100
        elif self.target_dataset == 'ImageNet':
            return 1000
        
                
    def _initialize_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
                

    def _activate_aux_net(self, x: torch.Tensor, trigger: str):
        for aux_net_name, aux_net_config in self.config['aux_nets'].items():
            if aux_net_config['trigger'] == trigger:
                if self.sum_aux_nets_outputs is None:
                    self.sum_aux_nets_outputs = self.aux_nets[aux_net_name](x)
                else:
                    self.sum_aux_nets_outputs += self.aux_nets[aux_net_name](x)
                    
                for m in getattr(self, trigger).modules():
                    if isinstance(m, SilhouetteConv2d):
                        m.silhouette = self.aux_nets[aux_net_name].silhouette
                
        
    def _make_input_stem(self) -> nn.Sequential:
        if self.target_dataset == 'ImageNet':
            return nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.inplanes),
                QuantPACTReLU(bit_width=8),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                QuantPACTReLU(bit_width=8)
            )
        
        
    def _make_layer(self, name: str, block: Union[BasicBlock, Bottleneck], num_block: int, planes: int, stride: int=1) -> nn.Sequential:
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride, **self.config['layers'][name]))
            self.inplanes = planes
        return nn.Sequential(*layers)
    
    
    def _make_aux_nets(self) -> nn.ModuleDict:
        aux_nets = nn.ModuleDict({})
        for name, aux_config in self.config['aux_nets'].items():
            aux_net = AuxiliaryNet(extractor_config=aux_config['extractor_config'], 
                                   compress_config=aux_config['compress_config'])
            aux_nets[name] = aux_net
            
        return aux_nets
    
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(x)
        
        self._activate_aux_net(x, 'layer1')
        x = self.layer1(x)
        
        self._activate_aux_net(x, 'layer2')
        x = self.layer2(x)
        
        self._activate_aux_net(x, 'layer3')
        x = self.layer3(x)
        
        self._activate_aux_net(x, 'layer4')
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y_main = self.fc(x)
        
        w = torch.sigmoid(self.w)
        y_pred = w * y_main + (1 - w) * self.sum_aux_nets_outputs
        self.sum_aux_nets_outputs = None
        
        return y_pred
    
    
    
def resnet18_Silhouette(target_dataset: str, config: dict):
    return ResNet_Silhouette(BasicBlock, [2, 2, 2, 2], target_dataset, config)



def resnet34_Silhouette(target_dataset: str, config: dict):
    return ResNet_Silhouette(BasicBlock, [3, 4, 6, 3], target_dataset, config)



def resnet50_Silhouette(target_dataset: str, config: dict):
    return ResNet_Silhouette(Bottleneck, [3, 4, 6, 3], target_dataset, config)