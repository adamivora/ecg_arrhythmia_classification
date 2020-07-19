import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet18

__all__ = ['CnnGru']


def backbone_ResNet18():
    return ResNet18(head_only=True, num_classes=1)


class CnnGru(nn.Module):
    def __init__(self, num_classes, backbone=backbone_ResNet18, gru_hidden_size=128, gru_layers=2, gru_dropout=0):
        super().__init__()

        self.conv = backbone()
        self.conv2 = nn.Conv1d(512, gru_hidden_size, kernel_size=1)
        self.gru = nn.GRU(input_size=1, hidden_size=gru_hidden_size, bidirectional=True, num_layers=gru_layers,
                          dropout=gru_dropout)
        self.fc = nn.Linear(2 * gru_hidden_size * gru_layers, num_classes)

    def forward(self, input):
        x = self.conv(input)
        x = F.relu(self.conv2(x))
        x = x.transpose(0, 1)
        _, x = self.gru(x)
        x = x.transpose(1, 0).flatten(start_dim=1)
        x = self.fc(x)
        return x
