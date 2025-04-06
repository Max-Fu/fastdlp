"""Vision transform utilities for image processing.

This module provides functions for creating and applying vision transforms,
including default transforms and augmented transforms.
"""
import PIL
import numpy as np
from typing import Union
import torch
import torchvision
from timm.data.transforms_factory import transforms_noaug_train

class ToTensor(torch.nn.Module):
    """
    Convert a PIL image or numpy array to a torch tensor. (C, H, W)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, img : Union[PIL.Image.Image, np.ndarray, torch.Tensor]):
        if isinstance(img, PIL.Image.Image):
            img = torch.from_numpy(np.array(img))
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        # permute to (C, H, W) if not already 
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        return img
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

def default_vision_transform(
    normalize: bool = False,
):
    """Create default vision transform without augmentation.
    
    Args:
        normalize: Whether to normalize the image
        
    Returns:
        Compose transform for vision data
        Output will be uint8 tensor if normalize=False, float32 if normalize=True
    """
    transforms = [
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
    ]
    
    if normalize:
        transforms.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
    else:
        transforms.append(ToTensor())
        
    return torchvision.transforms.Compose(transforms)

def aug_vision_transform(
    brightness: float = 0.1,
    contrast: float = 0.2,
    saturation: float = 0.01,
    hue: float = 0.01,
    normalize: bool = False,
):
    """Create augmented vision transform with color jittering.
    
    Args:
        brightness: Brightness adjustment factor
        contrast: Contrast adjustment factor
        saturation: Saturation adjustment factor
        hue: Hue adjustment factor
        normalize: Whether to normalize the image
        
    Returns:
        Compose transform for vision data with augmentation
        Output will be uint8 tensor if normalize=False, float32 if normalize=True
    """
    transforms = [
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        ),
    ]
    
    if normalize:
        transforms.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
    else:
        transforms.append(ToTensor())
        
    return torchvision.transforms.Compose(transforms)