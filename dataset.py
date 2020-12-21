import os
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Lighting:
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
    
    def __call__(self, x):
        if self.alphastd == 0:
            return x
        alpha = (np.random.randn(3) * self.alphastd).astype('float32')
        bias = self.eigvec @ (alpha * self.eigval)
        out = (x + bias).clip(0, 255).astype('uint8')
        return Image.fromarray(out, 'RGB')

def get_imagenet_transform(transform_type):
    if transform_type == 'basic':
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        crop_scale = 0.08
    elif transform_type == 'inception':
        imagenet_mean = [0.5, 0.5, 0.5]
        imagenet_std = [0.5, 0.5, 0.5]
        crop_scale = 0.08
    elif transform_type == 'mobile':
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        crop_scale = 0.25
    
    imagenet_eigval = np.array([0.2175, 0.0188, 0.0045])
    imagenet_eigvec = np.array([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203]
    ])
    
    if transform_type == "basic":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, (crop_scale, 1.0)),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            Lighting(0.1, imagenet_eigval, imagenet_eigvec),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
        val_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)
        ])
    
    return {'train': train_tf, 'val': val_tf}

def get_imagenet_loader(bs, num_workers, dir_name, transform_type='basic', mode='train'):
    tfs = get_imagenet_transform(transform_type)
    if mode == 'train':
        train_ds = datasets.ImageFolder(dir_name, tfs['train'])
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers)
        return train_dl
    
    elif mode == 'val':
        val_ds = datasets.ImageFolder(dir_name, tfs['val'])
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
        return val_dl