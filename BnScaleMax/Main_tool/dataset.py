import torch
from torchvision import datasets, transforms


def data_set(cuda, dataset, batch_size, test_batch_size):
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
    if dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('Data/cifar10', train=True, download=True,
                             transform=transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])), batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('Data/cifar10', train=False,
                             transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])), batch_size=test_batch_size, shuffle=True, **kwargs)
    elif dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('Data/cifar100', train=True, download=True,
                              transform=transforms.Compose([
                               transforms.Pad(4),
                               transforms.RandomCrop(32),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                              ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('Data/cifar100', train=False,
                              transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                              ])), batch_size=test_batch_size, shuffle=True, **kwargs)
    elif dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('Data/mnist', train=True, download=True,
                           transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('Data/mnist', train=False,
                           transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    return train_loader, test_loader
