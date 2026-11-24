import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms as T
from .augmentation import RandAugment

class DoubleViews(Dataset):
    def __init__(self, base, normalize):
        self.base = base
        self.views = [T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), normalize]),
                      T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), RandAugment(2, 10), normalize])]
    def __len__(self):
        return 2*len(self.base)
    def __getitem__(self, index):
        i = index % len(self.base)
        return self.views[index//len(self.base)](Image.fromarray(self.base.data[i])), self.base.targets[i]

def evaluation_dataset(config, root, download=False):
    if config['dataset'] == 'cifar100':
        transform = T.Compose([T.ToTensor(), T.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))])
        return datasets.CIFAR100(root, train=False, download=download, transform=transform)
    if config['dataset'] != 'imagenet':
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                           T.CenterCrop(224), T.ToTensor(),
                           T.Normalize((.485,.456,.406),(.229,.224,.225))])
    test = datasets.ImageFolder(str(root)+'/val', transform)
    if len(test.classes) != 1000:
        raise ValueError('ImageNet evaluation requires 1000 class folders under val/')
    return test

def stratified_split(targets, per_class, seed):
    """Split original images before creating weak/strong augmentation views."""
    labels = np.asarray(targets)
    if not isinstance(per_class, int) or per_class < 1:
        raise ValueError('validation_per_class must be a positive integer')
    generator = np.random.default_rng(seed)
    train, validation = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if per_class >= len(indices):
            raise ValueError('Validation must leave training images in each class')
        generator.shuffle(indices)
        validation.extend(indices[:per_class].tolist())
        train.extend(indices[per_class:].tolist())
    return sorted(train), sorted(validation)


def datasets_for(config, root, download=False):
    if config['dataset'] == 'cifar100':
        clean = T.Compose([T.ToTensor(), T.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))])
        base = datasets.CIFAR100(root, train=True, download=download)
        if config.get('validation_per_class', 0):
            train_indices, val_indices = stratified_split(base.targets, config['validation_per_class'],
                                                        config.get('split_seed', 20261118))
            validation = datasets.CIFAR100(root, train=True, download=download, transform=clean)
            test = Subset(validation, val_indices)
            base.data = base.data[train_indices]
            base.targets = [base.targets[i] for i in train_indices]
        else:
            test = evaluation_dataset(config, root, download)
        if config.get('augmentation') == 'double_view':
            train = DoubleViews(base, clean)
        else:
            base.transform = T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),clean]); train=base
    else:
        if config.get('validation_per_class', 0):
            raise ValueError('Training-set holdout currently supports CIFAR-100 only')
        test = evaluation_dataset(config, root, download)
        clean = [T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))]
        train = datasets.ImageFolder(str(root)+'/train',T.Compose([T.RandomResizedCrop(224,interpolation=T.InterpolationMode.BILINEAR),T.RandomHorizontalFlip()]+clean))
        if train.classes != test.classes or len(train.classes)!=1000:
            raise ValueError('ImageNet requires matching 1000-class train/val folders')
    return train, test

def seed_worker(_):
    seed = torch.initial_seed()%2**32
    np.random.seed(seed); random.seed(seed)
