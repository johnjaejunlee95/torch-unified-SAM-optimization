import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
# from utils.cutout import Cutout


class TinyImageNet:
    def __init__(self, batch_size, num_workers):
        mean, std = self._get_statistics()
        

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(64, 64), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            torchvision.transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = ImageFolder(root='/dataset/tinyIN/train', transform=train_transform) # /dataset/tinyIN/train --> path to tinyImageNet train folder
        test_set = ImageFolder(root='/dataset/tinyIN/val', transform=test_transform)  # /dataset/tinyIN/val --> path to tinyImageNet val folder
        

        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    def _get_statistics(self):
        train_set = ImageFolder(root='/dataset/tinyIN/train', transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set, batch_size=512)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
