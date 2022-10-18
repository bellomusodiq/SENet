import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SELayer(nn.Module):
    def __init__(self, in_channels, kernel_size, reduction_ratio):
        super(SELayer, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.reduction_ratio = reduction_ratio
        self.avgpool = nn.AvgPool2d(kernel_size)
        self.fcsq = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fcex = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tr = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fcsq(x)
        x = self.relu(x)
        x = self.fcex(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        scale = tr * x
        return scale
