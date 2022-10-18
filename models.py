import torch
import torch.nn as nn

from .layers import SELayer


class ConvNet(nn.Module):
    def __init__(self, in_channels, mum_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.maxpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.maxpool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.maxpool3 = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.bn8 = nn.BatchNorm2d(128)

        self.global_avgpool = nn.AvgPool2d(4)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(128, mum_classes)

    def forward(self, x):
        x = self.relu(self.bn1(DropBlock2d(self.conv1(x), block_size=3, p=0.4)))
        x = self.relu(self.bn2(DropBlock2d(self.conv2(x), block_size=3, p=0.4)))
        x = self.maxpool1(x)
        x = self.relu(self.bn3(DropBlock2d(self.conv3(x), block_size=3, p=0.4)))
        x = self.relu(self.bn4(DropBlock2d(self.conv4(x), block_size=3, p=0.4)))
        x = self.maxpool2(x)
        x = self.relu(self.bn5(DropBlock2d(self.conv5(x), block_size=3, p=0.4)))
        x = self.relu(self.bn6(DropBlock2d(self.conv6(x), block_size=3, p=0.4)))
        x = self.maxpool3(x)
        x = self.relu(self.bn7(DropBlock2d(self.conv7(x), block_size=3, p=0.4)))
        x = self.relu(self.bn8(DropBlock2d(self.conv8(x), block_size=3, p=0.4)))
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SENet(nn.Module):
    def __init__(self, in_channels, reduction_ratio, num_classes):
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.se_layer1 = SELayer(
            in_channels=16, kernel_size=32, reduction_ratio=reduction_ratio
        )
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.se_layer2 = SELayer(
            in_channels=16, kernel_size=32, reduction_ratio=reduction_ratio
        )
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.se_layer3 = SELayer(
            in_channels=32, kernel_size=16, reduction_ratio=reduction_ratio
        )
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.se_layer4 = SELayer(
            in_channels=32, kernel_size=16, reduction_ratio=reduction_ratio
        )
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.se_layer5 = SELayer(
            in_channels=64, kernel_size=8, reduction_ratio=reduction_ratio
        )
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.se_layer6 = SELayer(
            in_channels=64, kernel_size=8, reduction_ratio=reduction_ratio
        )
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.se_layer7 = SELayer(
            in_channels=128, kernel_size=4, reduction_ratio=reduction_ratio
        )
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.se_layer8 = SELayer(
            in_channels=128, kernel_size=4, reduction_ratio=reduction_ratio
        )
        self.global_avgpool = nn.AvgPool2d(4)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.se_layer1(
            self.relu(self.bn1(DropBlock2d(self.conv1(x), block_size=3, p=0.4)))
        )
        x = self.se_layer2(
            self.relu(self.bn2(DropBlock2d(self.conv2(x), block_size=3, p=0.4)))
        )
        x = self.maxpool1(x)
        x = self.se_layer3(
            self.relu(self.bn3(DropBlock2d(self.conv3(x), block_size=3, p=0.4)))
        )
        x = self.se_layer4(
            self.relu(self.bn4(DropBlock2d(self.conv4(x), block_size=3, p=0.4)))
        )
        x = self.maxpool2(x)
        x = self.se_layer5(
            self.relu(self.bn5(DropBlock2d(self.conv5(x), block_size=3, p=0.4)))
        )
        x = self.se_layer6(
            self.relu(self.bn6(DropBlock2d(self.conv6(x), block_size=3, p=0.4)))
        )
        x = self.maxpool3(x)
        x = self.se_layer7(
            self.relu(self.bn7(DropBlock2d(self.conv7(x), block_size=3, p=0.4)))
        )
        x = self.se_layer8(
            self.relu(self.bn8(DropBlock2d(self.conv8(x), block_size=3, p=0.4)))
        )
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
