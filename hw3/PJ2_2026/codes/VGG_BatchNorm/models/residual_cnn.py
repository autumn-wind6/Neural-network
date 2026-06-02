"""
Residual CNN for CIFAR-10 Project 2 experiments.
"""
import torch
from torch import nn


def make_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="relu",
                 dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = make_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = make_activation(activation)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act2(out)
        return out


class ResidualCifarNet(nn.Module):
    """Small residual CNN containing Conv2d, BN, Dropout, pooling and FC layers."""

    def __init__(self, num_classes=10, base_channels=32, activation="relu",
                 dropout=0.2):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            make_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(c1, c1, activation=activation, dropout=dropout),
            ResidualBlock(c1, c1, activation=activation, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2, activation=activation, dropout=dropout),
            ResidualBlock(c2, c2, activation=activation, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2, activation=activation, dropout=dropout),
            ResidualBlock(c3, c3, activation=activation, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def first_conv_weight(self):
        return self.stem[0].weight.detach().cpu()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    model = ResidualCifarNet()
    x = torch.randn(4, 3, 32, 32)
    print(model(x).shape)
    print(count_parameters(model))
