import torch

import torchvision
from torchvision import transforms as T
from transforms import Compose, ToTensor, RandomHorizontalFlip

def get_tranforms(train=True):
    t = []
    t.append(ToTensor())
    if train==True:
        t.append(RandomHorizontalFlip(0.5))
    return Compose(t)

def collate_fn(batch):
    return tuple(zip(*batch))