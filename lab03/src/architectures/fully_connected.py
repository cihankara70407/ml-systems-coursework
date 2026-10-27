import torch
import torch.nn as nn

class CihanDeepMLP(nn.Module):
    """
    Multilayer Perceptron with Batch Normalization
    Input: (B, 1, 28, 28) -> Flatten(784) -> Linear(784, 256) -> BN -> ReLU -> Linear(256, 128) -> BN -> ReLU -> Linear(128, 10)
    """
    def __init__(self, in_features=784, h1=256, h2=128, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x):
        return self.encoder(x)
