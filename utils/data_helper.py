import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


class DataHelper:
    DATA_PATH = './data/'
    transform_raw = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_flip = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.RandomHorizontalFlip()
    ])
    transform_rotate = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.RandomRotation(10)
    ])
    train_set_raw = None
    train_set_flip = None
    train_set_rotate = None

    def __init__(self):
        self.train_set_raw = torchvision.datasets.FashionMNIST(root=self.DATA_PATH,
                                                               train=True,
                                                               download=True,
                                                               transform=self.transform_raw)
        self.train_set_flip = torchvision.datasets.FashionMNIST(root=self.DATA_PATH,
                                                                train=True,
                                                                download=True,
                                                                transform=self.transform_flip)
        self.train_set_rotate = torchvision.datasets.FashionMNIST(root=self.DATA_PATH,
                                                                  train=True,
                                                                  download=True,
                                                                  transform=self.transform_rotate)
        self.test_set_raw = torchvision.datasets.FashionMNIST(root=self.DATA_PATH,
                                                              train=False,
                                                              download=True,
                                                              transform=self.transform_raw)

    def get_loader_train(self, train_set):
        return torch.utils.data.DataLoader(train_set,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)

    def get_loader_test(self, test_set):
        return torch.utils.data.DataLoader(test_set,
                                           batch_size=100,
                                           shuffle=False,
                                           num_workers=1)

    def split(self, train_set):
        if len(train_set) != 60000:
            raise ValueError

        return torch.utils.data.random_split(train_set,
                                             [50000, 10000],
                                             generator=torch.Generator().manual_seed(1))
