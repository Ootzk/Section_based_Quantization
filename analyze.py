from collections import Counter
import os
import json
import importlib
import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from torchvision.transforms import ToPILImage

import ignite.distributed as idist
from ignite.engine import Engine, State
from ignite.metrics import Accuracy, TopKCategoricalAccuracy

from architectures.train import get_train_eval_datasets, get_dataloader, create_evaluator
#import architectures


__all__ = [
    'load_analyzed_model',
    'visualize_images',
    'visualize_tensor',
    'visualize_tensor_distribution', 
    'integrate_mask',
    'visualize_mask'
]

###############################################################################################################
def load_model_to_be_analyzed(experiment_dir, checkpoint=None, verbose=False):
    # to make typing experiment directory easier from the user's POV.
    exp_dir = experiment_dir if experiment_dir.startswith('experiments/') else 'experiments/{0}'.format(experiment_dir)
    
    # get experiment config 'at that time' and show it.
    with open('{0}/config.json'.format(exp_dir), 'r') as exp_config_file:
        exp_config = json.load(exp_config_file)
    if verbose:
        print('model configuration to be analyzed: ')
        pprint.pprint(exp_config)
        print()
    
    # load skeleton of model 'at that time' if save_architectures was enabled.
    try:
        arch_at_that_time = importlib.import_module('{0}.architectures'.format(exp_dir.replace('/', '.')))
    except ModuleNotFoundError:
        if verbose:
            print('WARNING! 실험 당시의 architecture가 실험 폴더에 저장되어 있지 않음!')
            print('최신 architecture와 model loader를 사용하여 불러오기 때문에 실험 당시의 model과는 다르게 작동할 가능성이 있음')
            print()
        model = architectures.get_model_skeleton(exp_config['model'], exp_config['dataloader']['dataset'])
    else:
        try:
            model = arch_at_that_time.get_model_skeleton(exp_config['model'], exp_config['dataloader']['dataset'])
            if verbose:
                print('실험 당시의 architecture가 저장되어 있고, model loader가 구현되어 있음')
                print('실험 당시의 model과 정확히 같게 행동할 것이 보장됨.')
                print()
        except:
            model = architectures.get_model_skeleton(exp_config['model'], exp_config['dataloader']['dataset'])
            if verbose:
                print('WARNING! 실험 당시의 architecture는 저장되어 있지만, model loader가 구현되어 있지 않음!')
                print('최신 model loader를 사용하여 불러오기 때문에 architecture도 최신으로 반영되며, 실험 당시의 model과는 다르게 작동할 가능성이 있음')
                print()
    
    # load checkpoint if exists.
    if checkpoint is None:
        print('checkpoint not specified, so just return not-pretrained model.')
        return model, exp_config
    else:
        for candidate in os.listdir('{0}/checkpoints/'.format(exp_dir)):
            if candidate.startswith('checkpoint_{0}_'.format(checkpoint)):
                model.load_state_dict(torch.load('{0}/checkpoints/{1}'.format(exp_dir, candidate))['model'])
                print('loaded checkpoint with epochs={0}.'.format(checkpoint))
                return model, exp_config
        
        print('no checkpoint with epochs={0}, so just return not pretrained model'.format(checkpoint))
        return model, exp_config


###############################################################################################################
def visualize_images(images, tags, classes=None):
    images = images.cpu()
    B = images.shape[0]
    fig = plt.figure(figsize=(B*10, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, B), axes_pad=0.15)
    if classes:
        for img, tag, ax in zip(images, tags, grid):
            ax.imshow(img.permute(1, 2, 0))
            ax.set_title(classes[tag])
            ax.axis('off')
    else:
        for img, tag in zip(images, tags):
            ax.imshow(img.permute(1, 2, 0))
            ax.axis('off')
    return fig



def visualize_tensor(tensor):
    tensor = tensor.cpu()
    B, C, H, W = tensor.shape
    fig, axes = plt.subplots(ncols=C, nrows=B, figsize=(C*2., B*2.))
    for i in range(B):
        for j in range(C):
            axes[i, j].imshow(tensor[i][j])
            axes[i, j].axis('off')
    return fig



def visualize_tensor_distribution(tensor):
    tensor = tensor.cpu()
    fig = plt.figure(figsize=(30, 15))
    sns.histplot(tensor.detach().flatten(), kde=False, bins=1000)
    return fig
    
    

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

    
    
def visualize_mask(mask):
    mask = integrate_mask(mask)
    mask = mask.cpu()
    B, C, H, W = mask.shape
    if C == 1: # DRQ mask
        fig, axes = plt.subplots(ncols=B, nrows=1, figsize=(B*10, 8))
        mask = mask.squeeze()
        for i in range(B):
            axes[i].imshow(mask[i])
            axes[i].axis('off')
        return fig
    else: # Silhouette mask
        fig, axes = plt.subplots(ncols=C, nrows=B, figsize=(C*10, B*8))
        for i in range(B):
            for j in range(C):
                axes[i, j].imshow(mask[i][j])
                axes[i, j].axis('off')
        return fig
        
    
def visualize_mask_ratio(mask):
    mask = integrate_mask(mask)
    mask = mask.cpu()
    nbits, counts = torch.unique(mask, sorted=True, return_counts=True)
    ratio = counts / sum(counts)
    
    fig = plt.figure(figsize=(5, 4))
    fig.patch.set_facecolor('white')
    plt.pie(ratio, labels=nbits.tolist(), autopct='%.2f%%')
    return fig

###############################################################################################################
def visualize_important_tensors(analyze_config):
    """
    visualize important tensors such as: feature maps, masks, etc...
    
    sample_analyze_config = {
        'experiment': {
            'experiment_dir': 'ResNet50(Vanilla)+CIFAR100 EffectOfQScheme/04',
            'checkpoint': 80
        },
        'target_layers': ['conv1', 'layer1.0.conv1'],
        'analytes': ['input', 'mask', 'output', 
                     'internal_act_inputs', 'internal_act_outputs', 'internal_conv_inputs', 'internal_conv_outputs'],
        'sample_size': 20,
        'only_statistics': True
    }
    """
    cache = {layername: {} for layername in analyze_config['target_layers']}
    
    ### forward hooks ###
    def hook_wrapper_interface_DRQ(layername):
        def hook(module, x, y):
            if 'input' in analyze_config['analytes']:
                cache[layername]['input'] = x[0].detach()
            if 'mask' in analyze_config['analytes']:
                cache[layername]['mask'] = module._generate_masks(x[0].detach())
            if 'output' in analyze_config['analytes']:
                cache[layername]['output'] = y.detach()
        return hook
    
    def hook_wrapper_interface_Silhouette(layername):
        def hook(module, x, y):
            if 'input' in analyze_config['analytes']:
                cache[layername]['input'] = x[0][0].detach()
            if 'mask' in analyze_config['analytes']:
                cache[layername]['mask'] = module._interpolate_masks(x[0][1], x[0][0].shape[-2:])
            if 'output' in analyze_config['analytes']:
                cache[layername]['output'] = y[0].detach()
        return hook
    
    def hook_wrapper_internal(layername, internallayername):
        def hook(module, x, y):
            analytes = analyze_config['analytes']
            if 'internal_conv_inputs' in analytes or 'internal_act_inputs' in analytes:
                cache[layername]['{0}_input'.format(internallayername)] = x[0].detach()
            if 'internal_conv_outputs' in analytes or 'internal_act_outputs' in analytes:
                cache[layername]['{0}_output'.format(internallayername)] = y.detach()
        return hook
    
    def hook_wrapper_interface_other(layername):
        def hook(module, x, y):
            if 'input' in analyze_config['analytes']:
                cache[layername]['input'] = x[0].detach()
            if 'output' in analyze_config['analytes']:
                cache[layername]['output'] = y.detach()
        return hook
    
    ### backward hooks ###
    def hook_wrapper_interface_DRQ_backward(layername):
        def hook(module, grad_y, grad_x):
            if 'gradient input' in analyze_config['analytes']:
                cache[layername]['gradient input'] = grad_y[0].detach()
            if 'gradient output' in analyze_config['analytes']:
                cache[layername]['gradient output'] = grad_x[0].detach()
        return hook
    
    def hook_wrapper_interface_Silhouette_backward(layername):
        def hook(module, grad_y, grad_x):
            if 'gradient input' in analyze_config['analytes']:
                cache[layername]['gradient input'] = grad_y[0].detach()
            if 'gradient output' in analyze_config['analytes']:
                cache[layername]['gradient output'] = grad_x[0][0].detach()
        return hook
    
    def hook_wrapper_interface_other_backward(layername):
        def hook(module, grad_y, grad_x):
            if 'gradient input' in analyze_config['analytes']:
                cache[layername]['gradient input'] = grad_y[0].detach()
            if 'gradient output' in analyze_config['analytes']:
                cache[layername]['gradient output'] = grad_x[0].detach()
        return hook
    
    
    
    # model and experiment configuration 'at that time'
    model, exp_config= load_model_to_be_analyzed(analyze_config['experiment']['experiment_dir'], checkpoint=analyze_config['experiment']['checkpoint'], verbose=True)
    model.train()
    model = model.to(idist.device())
    
    # dataloader
    dataloader = DataLoader(get_train_eval_datasets(exp_config['dataloader']['dataset'])[1], batch_size=analyze_config['sample_size'])
    x, t = next(iter(dataloader))

    # loss_function
    loss_function = nn.__dict__[exp_config['loss_function']['type']]().to(idist.device())
    
    for layername, module in model.named_modules():
        if layername in analyze_config['target_layers']:
            if exp_config['model']['variation']['type'] == 'DRQ':
                #module.act_low.register_forward_hook(hook_wrapper_internal(layername, 'act_low'))
                #module.act_high.register_forward_hook(hook_wrapper_internal(layername, 'act_high'))
                module.conv_low.register_forward_hook(hook_wrapper_internal(layername, 'conv_low'))
                module.conv_high.register_forward_hook(hook_wrapper_internal(layername, 'conv_high'))
                module.register_forward_hook(hook_wrapper_interface_DRQ(layername))
                module.register_backward_hook(hook_wrapper_interface_DRQ_backward(layername))
                if 'weight' in analyze_config['analytes']:
                    cache[layername]['weight_low'] = module.conv_low.weight.data.detach()
                    cache[layername]['weight_high'] = module.conv_high.weight.data.detach()
            elif exp_config['model']['variation']['type'] == 'Silhouette':
                for nbit, layer in module.conv_layers.items():
                    layer.register_forward_hook(hook_wrapper_internal(layername, 'act_{0}'.format(nbit)))
                    layer.register_forward_hook(hook_wrapper_internal(layername, 'conv_{0}'.format(nbit)))
                module.register_forward_hook(hook_wrapper_interface_Silhouette(layername))
                if 'weight' in analyze_config['analytes']:
                    cache[layername]['weight'] = module.weight.data.detach()
            else:
                module.register_forward_hook(hook_wrapper_interface_other(layername))
                if 'weight' in analyze_config['analytes']:
                    cache[layername]['weight'] = module.weight.data.detach()
                module.register_backward_hook(hook_wrapper_interface_other_backward(layername))
    
    """
    if exp_config['model']['variation']['type'] == 'Silhouette':
        y_pred_main, y_pred_extractor = model(x.to(idist.device()))
        loss = loss_function(y_pred_main, t.to(idist.device()))
        loss.backward()
    else:
        y_pred = model(x.to(idist.device()))
        loss = loss_function(y_pred, t.to(idist.device()))
        loss.backward()
    """
    y_pred = model(x.to(idist.device()))
    loss = loss_function(y_pred, t.to(idist.device()))
    loss.backward()
        
        
    print('model input images')
    if analyze_config['only_statistics'] is False:
        visualize_images(x, t, dataloader.dataset.classes); plt.show(); plt.close();
    for layername, cache_ in cache.items():
        for subjectname, item in cache_.items():
            print('{0} @ epochs={1} / {2} / {3}'.format(analyze_config['experiment']['experiment_dir'],
                                                        analyze_config['experiment']['checkpoint'],
                                                        layername, subjectname))
            if subjectname == 'mask':
                if analyze_config['only_statistics'] is False:
                    fig = visualize_mask(item); plt.show(); plt.close();
                fig = visualize_mask_ratio(item); plt.show(); plt.close();
            else:
                if analyze_config['only_statistics'] is False:
                    fig = visualize_tensor(item); plt.show(); plt.close();
                fig = visualize_tensor_distribution(item); plt.show(); plt.close();
                print('    number of unique values: ', len(torch.unique(item)))



def visualize_layerwise_precision_ratio(analyze_config):
    """
    visualize layerwise precision ratio.
    support multiple checkpoint at once.
    
    sample_analyze_config = {
        'experiment': {
            'experiment_dir': 'ResNet50(Vanilla)+CIFAR100 EffectOfQScheme/04',
            'checkpoint': [40, 80]
        },
        'target_layers': ['conv1', 'layer1.0.conv1']
    }
    """
    cache = {epoch: {layername: Counter() for layername in analyze_config['target_layers']}
             for epoch in analyze_config['experiment']['checkpoint']}
    
    def hook_wrapper_Silhouette(epoch, layername):
        def hook(module, x, y):
            mask = module._interpolate_masks(x[0][1], x[0][0].shape[-2:])
            mask = integrate_mask(mask)
            
            nbit, nums = torch.unique(mask, sorted=True, return_counts=True)
            count = dict(zip(nbit.tolist(), nums.tolist()))
            cache[epoch][layername].update(count)
        return hook
    
    def hook_wrapper_others(epoch, layername):
        def hook(module, x, y):
            mask = module._generate_masks(x[0].detach())
            mask = integrate_mask(mask)
            
            nbit, nums = torch.unique(mask, sorted=True, return_counts=True)
            count = dict(zip(nbit.tolist(), nums.tolist()))
            cache[epoch][layername].update(count)
        return hook
    
    for i, epoch in enumerate(analyze_config['experiment']['checkpoint']):
        if i == 0:
            model, exp_config = load_model_to_be_analyzed(analyze_config['experiment']['experiment_dir'], checkpoint=epoch, verbose=True)
        else:
            model, exp_config = load_model_to_be_analyzed(analyze_config['experiment']['experiment_dir'], checkpoint=epoch)
        dataloader = get_dataloader(exp_config)
        
        for layername, module in model.named_modules():
            if layername in analyze_config['target_layers']:
                if exp_config['model']['variation']['type'] == 'Silhouette':
                    module.register_forward_hook(hook_wrapper_Silhouette(epoch, layername))
                else:
                    module.register_forward_hook(hook_wrapper_others(epoch, layername))
                
        model = model.to(idist.device())
            
        evaluator = create_evaluator(model, exp_config)
        evaluator.run(dataloader['eval'])
    
    
    fig, axes = plt.subplots(nrows=len(cache.keys()), ncols=1, sharex=True, sharey=True, figsize=(15, 3*len(cache.keys())))
    for i, (epoch, count_by_layername) in enumerate(cache.items()):
        pd.DataFrame(count_by_layername).T.plot(kind='bar', stacked=True, rot=45, ax=axes[i],
                                                title='epoch: {0}'.format(epoch))
    plt.tight_layout()
    plt.show()
    
    
    processed = pd.DataFrame.from_dict({(epoch, layername): cache[epoch][layername] 
                                        for epoch in cache.keys() 
                                        for layername in cache[epoch].keys()},
                                       orient='index')
    return processed



def evaluate_Silhouette_model_with_train_mode(analyze_config):
    """
    Evaluate Silhouette model with train mode(...) cause they returns NaN with eval mode.
    
    sample_analyze_config = {
        'experiment': {
            'experiment_dir': 'ResNet50(Vanilla)+CIFAR100 EffectOfQScheme/09',
            'checkpoint': 75
        }
    }
    """
    model, exp_config = load_model_to_be_analyzed(analyze_config['experiment']['experiment_dir'], checkpoint=analyze_config['experiment']['checkpoint'], verbose=True)
    model = model.to(idist.device())
    model = DP(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
    dataloader = get_dataloader(exp_config)
    
    def eval_step_Silhouette(engine, batch):
        model.train() # what the...
        with torch.no_grad():
            x = batch[0].to(idist.device())
            y = batch[1].to(idist.device())
            
            y_pred_main, y_pred_extractor = model(x)
            return {
                'y_pred_main': y_pred_main,
                'y_pred_extractor': y_pred_extractor,
                'y': y
            }
        
    evaluator = Engine(eval_step_Silhouette)
    top1 = Accuracy(output_transform=lambda x: [x['y_pred_main'], x['y']])
    top5 = TopKCategoricalAccuracy(k=5, output_transform=lambda x: [x['y_pred_main'], x['y']])
    top1.attach(evaluator, 'Eval/Top-1')
    top5.attach(evaluator, 'Eval/Top-5')    
    
    evaluator.run(dataloader['eval'])
    print('experiment: ', analyze_config['experiment']['experiment_dir'])    
    print('checkpoint: ', analyze_config['experiment']['checkpoint'])
    
    print('validation set Top-1 accuracy: {0}%'.format(evaluator.state.metrics['Eval/Top-1']))
    print('validation set Top-5 accuracy: {0}%'.format(evaluator.state.metrics['Eval/Top-5']))
    
    return evaluator.state