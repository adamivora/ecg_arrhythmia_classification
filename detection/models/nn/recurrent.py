import torch.nn as nn

from .resnet import resnet18


def backbone_resnet18():
    return resnet18(head_only=True, num_classes=1)


class RNN(nn.Module):
    def __init__(self, num_classes, backbone=backbone_resnet18, gru_hidden_size=256, gru_layers=2):
        super().__init__()

        self.conv = backbone()
        self.gru = nn.GRU(input_size=1, hidden_size=gru_hidden_size, bidirectional=True, num_layers=gru_layers)
        self.fc = nn.Linear(2 * gru_hidden_size * gru_layers, num_classes)

    def forward(self, input):
        x = self.conv(input)
        x = x.transpose(0, 1)
        _, x = self.gru(x)
        x = x.transpose(1, 0).flatten(start_dim=1)
        x = self.fc(x)
        return x
