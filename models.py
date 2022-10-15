import torch
import torch.nn as nn
from .layers import DenseBlock, TransitionLayer


class DenseNet(nn.Module):
    def __init__(self, in_channel, dense_out_channels=12):
        super(DenseNet, self).__init__()
        self.in_channel = in_channel
        self.dense_out_channels = dense_out_channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.dense_block1 = DenseBlock(16, 4, dense_out_channels)
        self.dense_block2 = DenseBlock(
            self.dense_block1.output_channels(), 4, dense_out_channels
        )
        self.dense_block3 = DenseBlock(
            self.dense_block2.output_channels(), 4, dense_out_channels
        )
        self.transition1 = TransitionLayer(self.dense_block3.output_channels(), 0.4)
        self.dense_block4 = DenseBlock(
            self.transition1.output_channels(), 6, dense_out_channels
        )
        self.dense_block5 = DenseBlock(
            self.dense_block4.output_channels(), 6, dense_out_channels
        )
        self.dense_block6 = DenseBlock(
            self.dense_block5.output_channels(), 6, dense_out_channels
        )
        self.transition2 = TransitionLayer(self.dense_block6.output_channels(), 0.4)
        self.dense_block7 = DenseBlock(
            self.transition2.output_channels(), 8, dense_out_channels
        )
        self.dense_block8 = DenseBlock(
            self.dense_block7.output_channels(), 8, dense_out_channels
        )
        self.avgpool = nn.AvgPool2d(4)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.dense_block8.output_channels(), 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = self.dense_block3(x)
        x = self.transition1(x)
        x = self.dense_block4(x)
        x = self.dense_block5(x)
        x = self.dense_block6(x)
        x = self.transition2(x)
        x = self.dense_block7(x)
        x = self.dense_block8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
