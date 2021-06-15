"""
analyze.py

helper functions and tools for analyze.
"""
import os
import json
import importlib
import pprint

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchvision.transforms import Normalize

import architectures

plt.rcParams['figure.facecolor']  = 'white'
plt.rcParams['savefig.facecolor'] = 'white'


def visualize_images(images, tags, applied_normalize, classes=None):
    inv_std = 1 / (torch.as_tensor(applied_normalize.std) - 1e-7)
    inv_mean = -torch.as_tensor(applied_normalize.mean) * inv_std
    unnormalize = Normalize(inv_mean, inv_std)
    
    images = unnormalize(images.detach().cpu())
    B = images.shape[0]
    fig, axes = plt.subplots(ncols=B, figsize=(B*10, 8))
    for img, tag, ax in zip(images, tags, axes):
        ax.imshow(img.permute(1, 2, 0))
        if classes:
            ax.set_title(classes[tag])
        ax.axis('off')
    plt.show(block=False)
    
    
    
def visualize_tensor(tensor):
    tensor = tensor.detach().cpu()
    assert tensor.dim() == 4 # B * C * H * W
    B, C, _, _ = tensor.shape
    
    if C == 1: # compressed channel
        tensor = tensor.squeeze() # B * H * W
        fig, axes = plt.subplots(ncols=B, figsize=(B*10, 8))
        for img, ax in zip(tensor, axes):
            ax.imshow(img)
            ax.axis('off')
    else:
        fig, axes = plt.subplots(ncols=C, nrows=B, figsize=(C*2, B*2))
        for ch_1img, axs in zip(tensor, axes):
            for img, ax in zip(ch_1img, axs):
                ax.imshow(img)
                ax.axis('off')
    plt.show(block=False)
        


def visualize_tensor_distribution(tensor, is_mask=False):
    if is_mask:
        tensor = integrate_mask(tensor)
        bit_width, counts = torch.unique(tensor, sorted=True, return_counts=True)
        ratio = counts / sum(counts)
        
        fig = plt.figure(figsize=(5, 5))
        plt.pie(ratio, labels=bit_width.tolist(), autopct='%.2f%%')
    else:
        tensor = tensor.detach().cpu()
        fig = plt.figure(figsize=(20, 10))
        sns.histplot(tensor.flatten(), kde=False, bins=1000)
        
    plt.title(f'tensor distribution\n# values: {torch.numel(tensor)}\n# unique values: {len(torch.unique(tensor))}')
    plt.show(block=False)
    
    

def integrate_mask(mask):
    if type(mask) == torch.Tensor: # DRQ mask
        integrated = torch.zeros_like(mask, dtype=torch.int8)
        integrated = integrated.masked_fill(mask, 8)
        integrated = integrated.masked_fill(~mask, 4)
        return integrated
    elif type(mask) == dict: # Silhouette mask
        integrated = None
        for nbit, m in mask.items():
            if integrated is None:
                integrated = torch.zeros_like(m, dtype=torch.int8)
            integrated = integrated.masked_fill(m, int(nbit))
        return integrated
    
    
    
def load_model(experiment_dir, checkpoint=None, verbose=False):
    if not experiment_dir.startswith('experiments/'):
        experiment_dir = f'experiments/{experiment_dir}'
        
    with open(f'{experiment_dir}/config.json', 'r') as exp_config_file:
        exp_config = json.load(exp_config_file)
    
    if verbose:
        print('experiment configuration: ')
        pprint.pprint(exp_config)
        
    try:
        archived_arch = importlib.import_module(f'{experiment_dir.replace("/", ".")}.architectures')
    except ModuleNotFoundError:
        if verbose:
            print('archived architecture: (X), archived model loader: (X)\n')
        model = architectures.get_model_skeleton(exp_config['model'], exp_config['dataloader']['dataset'])
    else:
        try:
            model = archived_arch.get_model_skeleton(exp_config['model'], exp_config['dataloader']['dataset'])
            if verbose:
                print('archived architecture: (O), archived model loader: (O)\n')
        except:
            model = architectures.get_model_skeleton(exp_config['model'], exp_config['dataloader']['dataset'])
            if verbose:
                print('archived architecture: (O), archived model loader: (X)\n')
                
    if checkpoint is None:
        if verbose:
            print(f'checkpoint: not specified')
        return model, exp_config
    else:
        for candidate in os.listdir(f'{experiment_dir}/checkpoints/'):
            if candidate.startswith(f'checkpoint_{checkpoint}_'):
                model.load_state_dict(torch.load(f'{experiment_dir}/checkpoints/{candidate}')['model'])
                if verbose:
                    print(f'checkpoint: epoch @ {checkpoint} loaded successfully')
                return model, exp_config
        
        if verbose:
            print(f'checkpoint: epoch @ {checkpoint} but not searched')
        return model, exp_config