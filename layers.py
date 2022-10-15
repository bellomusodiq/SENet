import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ConvLayer(in_channels, out_channels):

    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, 4, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
    ).to(device)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers=8, dense_out_channel=12):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.dense_out_channel = dense_out_channel
        self.dense_layers = []
        for i in range(1, num_layers + 1):
            dense_in_channel = in_channels + dense_out_channel * (i - 1)
            conv_layer = ConvLayer(dense_in_channel, dense_out_channel)
            self.dense_layers.append(conv_layer)

    def forward(self, x):
        input_layer = x
        output_layer = None
        for i, dense_layer in enumerate(self.dense_layers):
            output_layer = dense_layer(input_layer)
            input_layer = torch.cat((output_layer, input_layer), dim=1)
        return input_layer

    def output_channels(self):
        return self.in_channels + self.dense_out_channel * self.num_layers


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression_ratio=0.5):
        super(TransitionLayer, self).__init__()
        self.out_channels = math.floor(in_channels * compression_ratio)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels, self.out_channels, kernel_size=1, stride=1, bias=False
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.maxpool(x)
        return x

    def output_channels(self):
        return self.out_channels
