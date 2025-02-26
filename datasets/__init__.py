from typing import Union, Any, Sequence

import numpy as np
import torch
import torchvision
from torch.utils import data


def image_to_numpy(image):
    return np.array(image) / 255


# def add_channel_axis(image: np.ndarray):
#     return image[..., np.newaxis]


# def normalize_image(image, mean, std):
#     return (image - mean[None, None, :]) / std[None, None, :]


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
    """
    TODO: this might be a repeat, maybe it's ok to make it special for shapes, but needs a check
    Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    

def get_dataset(dataset_cfg):
    """ Function which gets the dataset based on the dataset config. For now it only supports CIFAR10 and the FIGURE dataset."""

    if dataset_cfg.name == "cifar10":
        from torchvision.datasets import CIFAR10

        transform = torchvision.transforms.Compose([image_to_numpy])
        train_dset = torchvision.datasets.CIFAR10(root=dataset_cfg.path, train=1, transform=transform, download=1)
        test_dset = torchvision.datasets.CIFAR10(root=dataset_cfg.path, train=0, transform=transform, download=1)
    elif dataset_cfg.name == "FIGURE":
        from datasets.figure_dset import FigureDataset

        transform = torchvision.transforms.Compose([image_to_numpy, lambda x: x.transpose(1, 2, 0)])
        train_dset = FigureDataset("FIGURE-Shape-B", split="train", color_consistency=0.9, transform=transform, download=True)  

        if dataset_cfg.swap_bias:
            test_dset = FigureDataset("FIGURE-Shape-CB", split="test-bias", color_consistency=0.9, transform=transform, download=True)
        else:
            test_dset = FigureDataset("FIGURE-Shape-B", split="test", color_consistency=0.9, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_cfg.name}")
    
    return train_dset, test_dset


def get_dataloader(dataset_cfg, shuffle_train=True):
    """ Function which gets the dataloader based on the dataset config. """

    train_dset, test_dset = get_dataset(dataset_cfg)
    
    if dataset_cfg.num_signals_train != -1:
        train_dset = data.Subset(train_dset, np.arange(0, dataset_cfg.num_signals_train))
    if dataset_cfg.num_signals_test != -1:
        test_dset = data.Subset(test_dset, np.arange(0, dataset_cfg.num_signals_test))

    train_loader = data.DataLoader(
        train_dset,
        batch_size=dataset_cfg.batch_size,
        shuffle=shuffle_train,
        num_workers=dataset_cfg.num_workers,
        collate_fn=numpy_collate,
        persistent_workers=False,
        drop_last=True
    )

    test_loader = data.DataLoader(
        test_dset,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        collate_fn=numpy_collate,
        persistent_workers=False,
        drop_last=True
    )
    return train_loader, test_loader
