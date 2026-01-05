import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np 

class CIFAR:
    def __init__(self, batch_size, num_workers, cifar_type=100):
        self.cifar_type = cifar_type
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std= torch.tensor([0.2470, 0.2435, 0.2616])
        
        print(f"CIFAR-{cifar_type} Mean: {np.round(mean.numpy(), 4)}, Std: {np.round(std.numpy(), 4)}")
        

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        if cifar_type == 100:
            train_set = torchvision.datasets.CIFAR100(root='src/data', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root='src/data', train=False, download=True, transform=test_transform)
        else:
            train_set = torchvision.datasets.CIFAR10(root='src/data', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root='src/data', train=False, download=True, transform=test_transform)
      
        
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    