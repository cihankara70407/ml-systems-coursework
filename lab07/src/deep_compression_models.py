"""
Deep Compression Model Topologies: Teacher & Sub-100k Student
Author: Cihan Karayigit (Student ID: 70407)
Academic Program: Computer Engineering, Vistula University
Coursework: Machine Learning Systems (Laboratory 07)
Reference: Vijay Janapa Reddi, Machine Learning Systems (Chapters 9 & 10)
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class DeepFashionTeacherNet(nn.Module):
    """
    Oversized Teacher Convolutional Neural Network.
    Target Parameter Scale: ~1.0M parameters.
    Topology:
      Stage 1: Conv2d(1 -> 48, 3x3, pad 1) -> BatchNorm -> LeakyReLU(0.1) -> MaxPool(2) [14x14]
      Stage 2: Conv2d(48 -> 96, 3x3, pad 1) -> BatchNorm -> LeakyReLU(0.1) -> MaxPool(2) [7x7]
      Stage 3: Conv2d(96 -> 128, 3x3, pad 1) -> BatchNorm -> LeakyReLU(0.1) [7x7]
      Dense Head: Flatten (6272) -> Linear(6272 -> 160) -> LeakyReLU(0.1) -> Dropout(0.25) -> Linear(160 -> 10)
    """
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.25):
        super().__init__()
        self.conv_trunk = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)  # 7x7
        )
        self.linear_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 160),  # 6272 * 160 + 160 = 1,003,680 params
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(160, num_classes)   # 160 * 10 + 10 = 1,610 params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_trunk(x)
        return self.linear_head(features)


class Sub100kStudentNet(nn.Module):
    """
    Lightweight Student Network for Micro-Class Hardware Deployment.
    Target Parameter Scale: ~98k parameters (~11x compression factor).
    Topology:
      Stage 1: Conv2d(1 -> 16, 3x3, pad 1) -> BatchNorm -> LeakyReLU(0.1) -> MaxPool(2) [14x14]
      Stage 2: Conv2d(16 -> 24, 3x3, pad 1) -> BatchNorm -> LeakyReLU(0.1) -> MaxPool(2) [7x7]
      Stage 3: Conv2d(24 -> 32, 3x3, pad 1) -> BatchNorm -> LeakyReLU(0.1) [7x7]
      Dense Head: Flatten (1568) -> Linear(1568 -> 55) -> LeakyReLU(0.1) -> Linear(55 -> 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv_trunk = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14

            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7

            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)  # 7x7
        )
        self.linear_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 55),  # 1568 * 55 + 55 = 86,295 params
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(55, num_classes)  # 55 * 10 + 10 = 560 params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_trunk(x)
        return self.linear_head(features)


def inspect_model_capacity(net: nn.Module) -> Dict[str, Any]:
    param_counts = {name: p.numel() for name, p in net.named_parameters()}
    total = sum(param_counts.values())
    return {
        'total_params': total,
        'breakdown': param_counts
    }


if __name__ == '__main__':
    t = DeepFashionTeacherNet()
    s = Sub100kStudentNet()
    print("Teacher Total:", sum(p.numel() for p in t.parameters()))
    print("Student Total:", sum(p.numel() for p in s.parameters()))
