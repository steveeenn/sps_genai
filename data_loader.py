import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_loader(data_dir="./data", batch_size=32, train=True, dataset_name="cifar10"):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        try:
            dataset = torchvision.datasets.CIFAR10(
                root=data_dir,
                train=train,
                download=False,
                transform=transform
            )
        except RuntimeError:
            dataset = torchvision.datasets.CIFAR10(
                root=data_dir,
                train=train,
                download=True,
                transform=transform
            )

    elif dataset_name == "fashionmnist":
        def preprocess(img):
            img = np.pad(img, ((2, 2), (2, 2)), constant_values=0.0)
            return img

        transform = transforms.Compose([
            transforms.Lambda(preprocess),
            transforms.ToTensor()
        ])

        try:
            dataset = torchvision.datasets.FashionMNIST(
                root=data_dir,
                train=train,
                download=False,
                transform=transform
            )
        except RuntimeError:
            dataset = torchvision.datasets.FashionMNIST(
                root=data_dir,
                train=train,
                download=True,
                transform=transform
            )

    else:
        raise ValueError("dataset_name must be one of: cifar10, fashionmnist")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )