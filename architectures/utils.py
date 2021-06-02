from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.datasets as datasets
from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, Resize, Normalize, ToTensor

from .Vanilla import *
from .QAT import *
from .DRQ import *
from .Silhouette import *

__all__ = [
    'get_model_skeleton'
]


###############################################################################################################
dataset_location = {
    'CIFAR10':  '/dataset/CIFAR10',
    'CIFAR100': '/dataset/CIFAR100',
    'ImageNet': '/dataset/ImageNet/Classification'
}

creators = {
    'resnet18_Vanilla': resnet18_Vanilla,
    'resnet34_Vanilla': resnet34_Vanilla,
    'resnet50_Vanilla': resnet50_Vanilla,
    
    'resnet18_QAT': resnet18_QAT,
    'resnet34_QAT': resnet34_QAT,
    'resnet50_QAT': resnet50_QAT,
    
    'resnet18_DRQ': resnet18_DRQ,
    'resnet34_DRQ': resnet34_DRQ,
    'resnet50_DRQ': resnet50_DRQ,
    
    'resnet18_Silhouette': resnet18_Silhouette,
    'resnet34_Silhouette': resnet34_Silhouette,
    'resnet50_Silhouette': resnet50_Silhouette
}


normalization = {
    'CIFAR10':  Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'CIFAR100': Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'ImageNet': Normalize((0.4850, 0.4560, 0.4060), (0.2290, 0.2240, 0.2250)),
    'DEBUG':    Normalize((0.5000, 0.5000, 0.5000), (0.2000, 0.2000, 0.2000))
}


def get_transform(dataset):
    if dataset == 'CIFAR10':
        train_transform = Compose([RandomCrop((32, 32), 4), RandomHorizontalFlip(), ToTensor(), normalization[dataset]])
        eval_transform  = Compose([ToTensor(), normalization[dataset]])
    elif dataset == 'CIFAR100':
        train_transform = Compose([RandomCrop((32, 32), 4), RandomHorizontalFlip(), ToTensor(), normalization[dataset]])
        eval_transform  = Compose([ToTensor(), normalization[dataset]])
    elif dataset == 'ImageNet':
        train_transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ToTensor(), normalization[dataset]])
        eval_transform  = Compose([Resize(256), CenterCrop(224), ToTensor(), normalization[dataset]])
    else: # fake dataset
        train_transform = Compose([ToTensor(), normalization[dataset]])
        eval_transform  = Compose([ToTensor(), normalization[dataset]])
    
    return train_transform, eval_transform


def get_train_eval_datasets(dataset):
    train_transform, eval_transform = get_transform(dataset)
    
    if dataset == 'DEBUG':
        train_ds = datasets.FakeData(size=640, transform=train_transform)
        eval_ds  = datasets.FakeData(size=640, transform=eval_transform)
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        train_ds = datasets.__dict__[dataset](root=dataset_location[dataset], train=True, transform=train_transform)
        eval_ds  = datasets.__dict__[dataset](root=dataset_location[dataset], train=False, transform=eval_transform)
    elif dataset == 'ImageNet':
        train_ds = datasets.ImageFolder('{0}/train'.format(dataset_location[dataset]), transform=train_transform)
        eval_ds = datasets.ImageFolder('{0}/val'.format(dataset_location[dataset]), transform=eval_transform)
    
    return train_ds, eval_ds


###############################################################################################################
def get_model_skeleton(model_config, target_dataset):
    if model_config['variation']['type'] not in [None, 'Vanilla', 'DRQ', 'QAT', 'Silhouette']:
        raise NotImplementedError('variation {0} does not supported'.format(model_config['variation']['type']))
    
    variation = 'Vanilla' if model_config['variation']['type'] is None else model_config['variation']['type']
    arch = '{0}_{1}'.format(model_config['backbone'], variation)
    return creators[arch](target_dataset, model_config['variation']['config'])