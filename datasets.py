import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

train_data = datasets.CIFAR10(root="./data", download=True, train=True)
test_data = datasets.CIFAR10(root="./data", train=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=150, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=150)
