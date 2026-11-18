import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
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

def datasets_for(config, root, download=False):
    if config['dataset'] == 'cifar100':
        clean = T.Compose([T.ToTensor(), T.Normalize((.5071,.4867,.4408),(.2675,.2565,.2761))])
        base = datasets.CIFAR100(root, train=True, download=download)
        if config.get('augmentation') == 'double_view':
            train = DoubleViews(base, clean)
        else:
            base.transform = T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),clean]); train=base
        test = datasets.CIFAR100(root, train=False, download=download, transform=clean)
    else:
        clean = [T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))]
        train = datasets.ImageFolder(str(root)+'/train',T.Compose([T.RandomResizedCrop(224,interpolation=T.InterpolationMode.BILINEAR),T.RandomHorizontalFlip()]+clean))
        test = datasets.ImageFolder(str(root)+'/val',T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BILINEAR),T.CenterCrop(224)]+clean))
        if train.classes != test.classes or len(train.classes)!=1000:
            raise ValueError('ImageNet requires matching 1000-class train/val folders')
    return train, test

def seed_worker(_):
    seed = torch.initial_seed()%2**32
    np.random.seed(seed); random.seed(seed)
