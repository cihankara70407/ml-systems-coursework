import torch
import torch.nn as nn

class FeedForwardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.encoder(x)

class ConvNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.head = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        return self.head(self.conv_backbone(x))

class VisionTransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(49, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)
    def forward(self, x):
        B = x.size(0)
        p = x.unfold(2, 7, 7).unfold(3, 7, 7).contiguous().view(B, 16, 49)
        tokens = self.patch_embed(p)
        hidden = self.encoder(tokens)
        pooled = torch.mean(hidden, dim=1)
        return self.classifier(pooled)
