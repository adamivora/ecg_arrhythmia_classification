import torch
import torch.nn as nn


class InConv(nn.Module):
    def __init__(self, in_channels, start_channels, dropout):
        super().__init__()

        self.in_conv = nn.Sequential(
            make_conv_layer(in_channels, start_channels),
            nn.BatchNorm1d(start_channels),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            make_conv_layer(start_channels),
            ConvBlock(start_channels, start_channels, dropout)
        )
        self.pool = nn.MaxPool1d(1)

    def forward(self, x):
        x = self.in_conv(x)
        x2 = self.conv(x)
        x = self.pool(x)
        return x + x2


class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=in_channels, out_features=num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, stride=1):
        super().__init__()

        self.bn = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.conv = make_conv_layer(in_channels=in_channels, out_channels=out_channels, stride=stride)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, subsample_factor):
        super().__init__()

        self.block1 = ConvBlock(in_channels, out_channels, dropout, stride=subsample_factor)
        self.block2 = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool1d(subsample_factor, stride=subsample_factor)

    def forward(self, x):
        x2 = self.block1(x)
        x2 = self.block2(x2)
        x = self.pool(x)
        return x + x2


def make_conv_layer(in_channels=1, out_channels=None, kernel_size=15, **kwargs):
    if out_channels is None:
        out_channels = in_channels
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     padding=kernel_size // 2, **kwargs)


class StanfordECG(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, start_channels=32, num_layers=16, dropout=0.2):
        super().__init__()
        self.start_channels = start_channels
        self.in_conv = InConv(in_channels, start_channels, dropout)
        self.conv = nn.ModuleList()

        subsample_factor = 2
        for i in range(1, num_layers):  # one layer is already done
            self.conv.append(ResnetBlock(
                self.get_channels(i - 1),
                self.get_channels(i),
                dropout,
                subsample_factor
            ))
            subsample_factor = 1 if subsample_factor != 1 else 2

        self.out_conv = OutConv(self.get_channels(num_layers), num_classes)

    def get_channels(self, index):
        return self.start_channels * (2 ** (index // 4))

    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv(x)
        x = self.out_conv(x)

        return x
