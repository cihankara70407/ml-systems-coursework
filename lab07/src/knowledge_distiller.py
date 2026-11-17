"""
Knowledge Distillation Mechanics: Teacher Soft Targets vs. Vanilla Supervised Student
Author: Cihan Karayigit (Student ID: 70407)
Laboratory 07: Model Compression & Knowledge Distillation
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any


class SoftLabelKDLoss(nn.Module):
    """
    Kullback-Leibler divergence with temperature scaling combined with task Cross-Entropy:
      Loss = alpha * CE(logits_s, y) + (1 - alpha) * T^2 * KL(LogSoftmax(logits_s / T) || Softmax(logits_t / T))
    """
    def __init__(self, temperature: float = 4.0, alpha_factor: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha_factor
        self.ce_criterion = nn.CrossEntropyLoss()
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        hard_objective = self.ce_criterion(student_out, ground_truth)

        log_p_student = F.log_softmax(student_out / self.temperature, dim=1)
        p_teacher = F.softmax(teacher_out / self.temperature, dim=1)
        soft_objective = self.kl_criterion(log_p_student, p_teacher) * (self.temperature ** 2)

        return self.alpha * hard_objective + (1.0 - self.alpha) * soft_objective


def train_distilled_student_pipeline(
    student_net: nn.Module,
    teacher_net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 5,
    temp: float = 4.0,
    alpha: float = 0.5,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> List[float]:
    student_net.train()
    student_net.to(device)
    teacher_net.eval()
    teacher_net.to(device)

    kd_loss = SoftLabelKDLoss(temperature=temp, alpha_factor=alpha)
    optimizer = torch.optim.Adam(student_net.parameters(), lr=lr)
    history = []

    for ep in range(num_epochs):
        epoch_loss = 0.0
        steps = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            s_logits = student_net(x)
            with torch.no_grad():
                t_logits = teacher_net(x)

            loss = kd_loss(s_logits, t_logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        history.append(epoch_loss / max(1, steps))

    return history


def train_vanilla_student_pipeline(
    student_net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 5,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> List[float]:
    student_net.train()
    student_net.to(device)

    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student_net.parameters(), lr=lr)
    history = []

    for ep in range(num_epochs):
        epoch_loss = 0.0
        steps = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            s_logits = student_net(x)
            loss = ce(s_logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        history.append(epoch_loss / max(1, steps))

    return history
