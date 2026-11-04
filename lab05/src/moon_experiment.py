"""
Empirical Systems Benchmark: Custom AutoDiff vs. PyTorch on make_moons
Author: Cihan Karayigit (Student ID: 70407)
Laboratory 05: Frameworks and Automatic Differentiation
Network: 2 -> 8 -> 1 (ReLU hidden, Sigmoid output, BCE criterion)
"""

import time
import random
import statistics
import json
import os
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from autodiff import AutoDiffScalar


class AffineProjection:
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True):
        # Glorot uniform initialization
        limit = (6.0 / (in_features + out_features)) ** 0.5
        self.weights = [
            [AutoDiffScalar(random.uniform(-limit, limit)) for _ in range(in_features)]
            for _ in range(out_features)
        ]
        self.biases = [AutoDiffScalar(0.0) for _ in range(out_features)] if use_bias else []
        self.use_bias = use_bias

    def forward(self, x):
        outputs = []
        for j, row in enumerate(self.weights):
            dot = self.biases[j] if self.use_bias else AutoDiffScalar(0.0)
            for w_jk, x_k in zip(row, x):
                dot = dot + (w_jk * x_k)
            outputs.append(dot)
        return outputs

    def get_learnable_parameters(self):
        params = []
        for row in self.weights:
            params.extend(row)
        params.extend(self.biases)
        return params


class FeedForwardMoons:
    def __init__(self):
        # 2 -> 8 (ReLU) -> 1 (Sigmoid)
        self.proj1 = AffineProjection(2, 8)
        self.proj2 = AffineProjection(8, 1)

    def forward(self, x):
        h = self.proj1.forward(x)
        h_act = [node.relu() for node in h]
        out_linear = self.proj2.forward(h_act)
        return out_linear[0].sigmoid()

    def get_parameters(self):
        return self.proj1.get_learnable_parameters() + self.proj2.get_learnable_parameters()

    def clear_adjoints(self):
        for p in self.get_parameters():
            p.grad = 0.0


class PyTorchReferenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def compute_binary_cross_entropy(y_targets, y_probs):
    n = len(y_targets)
    losses = []
    for y, p in zip(y_targets, y_probs):
        if y == 1:
            loss = -p.log()
        else:
            loss = -(AutoDiffScalar(1.0) - p).log()
        losses.append(loss)
    
    total = losses[0]
    for l in losses[1:]:
        total = total + l
    return total * (1.0 / n)


def train_custom_system(X, y, epochs=40, lr=0.25):
    random.seed(1337)
    model = FeedForwardMoons()
    X_scalars = [[AutoDiffScalar(v) for v in row] for row in X]
    
    t0 = time.perf_counter()
    loss_history = []
    
    for epoch in range(epochs):
        predictions = [model.forward(x) for x in X_scalars]
        batch_loss = compute_binary_cross_entropy(y, predictions)
        
        model.clear_adjoints()
        batch_loss.backward()
        
        for param in model.get_parameters():
            param.data -= lr * param.grad
            
        loss_history.append(batch_loss.data)
        
    duration = time.perf_counter() - t0
    correct = sum((1 if p.data >= 0.5 else 0) == target for p, target in zip(predictions, y))
    acc = correct / len(y)
    
    return duration, loss_history[-1], acc


def train_pytorch_system(X, y, epochs=40, lr=0.25):
    torch.manual_seed(1337)
    model = PyTorchReferenceNet()
    criterion = nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    t_X = torch.tensor(X, dtype=torch.float32)
    t_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    t0 = time.perf_counter()
    loss_history = []
    
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(t_X)
        loss = criterion(out, t_y)
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
        
    duration = time.perf_counter() - t0
    with torch.no_grad():
        preds = (model(t_X) >= 0.5).float()
        acc = (preds == t_y).float().mean().item()
        
    return duration, loss_history[-1], acc


def execute_comparison():
    print("[MOONS BENCHMARK] Executing 5-run comparison for Cihan Karayigit...")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    custom_runtimes = []
    pytorch_runtimes = []
    custom_final_losses = []
    pytorch_final_losses = []
    
    for trial in range(5):
        c_time, c_loss, _ = train_custom_system(X, y, epochs=35, lr=0.25)
        p_time, p_loss, _ = train_pytorch_system(X, y, epochs=35, lr=0.25)
        
        custom_runtimes.append(c_time * 1000.0)
        pytorch_runtimes.append(p_time * 1000.0)
        custom_final_losses.append(c_loss)
        pytorch_final_losses.append(p_loss)
        print(f"  Trial {trial+1} | Custom: {custom_runtimes[-1]:.2f} ms | PyTorch: {pytorch_runtimes[-1]:.2f} ms")
        
    custom_median = statistics.median(custom_runtimes)
    pytorch_median = statistics.median(pytorch_runtimes)
    speedup = custom_median / pytorch_median
    
    telemetry = {
        'student': 'Cihan Karayigit (70407)',
        'trials': 5,
        'epochs': 35,
        'custom_runtimes_ms': custom_runtimes,
        'pytorch_runtimes_ms': pytorch_runtimes,
        'custom_median_ms': custom_median,
        'pytorch_median_ms': pytorch_median,
        'runtime_ratio': speedup,
        'custom_final_loss': statistics.median(custom_final_losses),
        'pytorch_final_loss': statistics.median(pytorch_final_losses)
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/moons_empirical_comparison.json', 'w') as f:
        json.dump(telemetry, f, indent=2)
        
    print(f"\n[SUMMARY] Custom Median: {custom_median:.2f} ms | PyTorch Median: {pytorch_median:.2f} ms | Ratio: {speedup:.1f}x")
    return telemetry


if __name__ == '__main__':
    execute_comparison()
