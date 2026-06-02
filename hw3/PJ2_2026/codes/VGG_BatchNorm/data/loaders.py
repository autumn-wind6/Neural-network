"""
Data loaders.
"""
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision.datasets as datasets


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = min(n_items, len(dataset))

    def __getitem__(self, index):
        if index >= self.n_items:
            raise IndexError(index)
        return self.dataset[index]

    def __len__(self):
        return self.n_items


def _default_root():
    return Path(__file__).resolve().parent


def cifar_transforms(train=False, augment=False):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    steps = []
    if train and augment:
        steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    steps.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(steps)


def get_cifar_loader(root=None, batch_size=128, train=True, shuffle=True,
                     num_workers=4, n_items=-1, download=True, augment=False,
                     drop_last=False):
    if root is None:
        root = _default_root()

    dataset = datasets.CIFAR10(
        root=str(root),
        train=train,
        download=download,
        transform=cifar_transforms(train=train, augment=augment),
    )
    if n_items > 0:
        dataset = Subset(dataset, range(min(n_items, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=drop_last,
    )

    return loader

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break
