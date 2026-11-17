"""
Pruning Protocols: Global/Layer-wise Magnitude Pruning & Structured Filter Reduction
Author: Cihan Karayigit (Student ID: 70407)
Laboratory 07: Model Compression & Knowledge Distillation
"""

import os
import copy
import time
import gzip
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import numpy as np


def extract_weight_tensors(module: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Gathers all prunable 2D/4D weight parameters from Conv2d and Linear blocks."""
    params = []
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            params.append((f"{name}.weight", layer.weight))
    return params


def apply_global_magnitude_mask(network: nn.Module, target_sparsity: float) -> Dict[str, torch.Tensor]:
    """
    Ranks weight parameters globally across all layers and zeros out lowest absolute weights.
    Returns dictionary mapping layer names to their binary tensor masks.
    """
    weight_entries = extract_weight_tensors(network)
    tensor_masks = {}

    if target_sparsity <= 0.0:
        for name, p in weight_entries:
            tensor_masks[name] = torch.ones_like(p.data)
        return tensor_masks

    all_magnitudes = torch.cat([p.data.abs().view(-1) for _, p in weight_entries])
    cutoff_idx = int(target_sparsity * all_magnitudes.numel())
    if cutoff_idx >= all_magnitudes.numel():
        cutoff_idx = all_magnitudes.numel() - 1

    threshold = torch.kthvalue(all_magnitudes, cutoff_idx)[0].item()

    for name, p in weight_entries:
        binary_mask = (p.data.abs() > threshold).float()
        p.data.mul_(binary_mask)
        tensor_masks[name] = binary_mask

    return tensor_masks


def apply_layerwise_magnitude_mask(network: nn.Module, target_sparsity: float) -> Dict[str, torch.Tensor]:
    """
    Applies magnitude-based thresholding independently per weight tensor.
    """
    weight_entries = extract_weight_tensors(network)
    tensor_masks = {}

    if target_sparsity <= 0.0:
        for name, p in weight_entries:
            tensor_masks[name] = torch.ones_like(p.data)
        return tensor_masks

    for name, p in weight_entries:
        flat_abs = p.data.abs().view(-1)
        cutoff_idx = int(target_sparsity * flat_abs.numel())
        if cutoff_idx >= flat_abs.numel():
            cutoff_idx = flat_abs.numel() - 1

        if cutoff_idx > 0:
            threshold = torch.kthvalue(flat_abs, cutoff_idx)[0].item()
            binary_mask = (p.data.abs() > threshold).float()
        else:
            binary_mask = torch.ones_like(p.data)

        p.data.mul_(binary_mask)
        tensor_masks[name] = binary_mask

    return tensor_masks


def apply_structured_l2_filter_pruning(network: nn.Module, channel_ratio: float) -> Dict[str, torch.Tensor]:
    """
    Structured channel pruning: calculates Euclidean norm per output channel and zeros weak filters.
    """
    filter_masks = {}
    if channel_ratio <= 0.0:
        return filter_masks

    for name, layer in network.named_modules():
        if isinstance(layer, nn.Conv2d):
            w = layer.weight.data  # (C_out, C_in, kH, kW)
            c_out = w.size(0)
            l2_norms = torch.norm(w.view(c_out, -1), p=2, dim=1)

            num_to_prune = int(channel_ratio * c_out)
            if num_to_prune >= c_out:
                num_to_prune = c_out - 1

            if num_to_prune > 0:
                l2_threshold = torch.kthvalue(l2_norms, num_to_prune)[0].item()
                c_mask = (l2_norms > l2_threshold).float()
            else:
                c_mask = torch.ones(c_out, device=w.device)

            expanded = c_mask.view(c_out, 1, 1, 1).expand_as(w)
            w.mul_(expanded)
            filter_masks[f"{name}.weight"] = expanded
            if layer.bias is not None:
                layer.bias.data.mul_(c_mask)

    return filter_masks


def finetune_with_static_mask(
    network: nn.Module,
    masks: Dict[str, torch.Tensor],
    train_dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 3,
    compute_device: str = 'cpu'
) -> List[float]:
    """
    Fine-tunes remaining unpruned parameters while holding pruned zero weights strictly static.
    """
    network.train()
    network.to(compute_device)
    loss_curve = []

    for ep in range(num_epochs):
        epoch_loss = 0.0
        step_count = 0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(compute_device), targets.to(compute_device)
            optimizer.zero_grad()
            preds = network(inputs)
            loss = criterion(preds, targets)
            loss.backward()

            # Nullify gradients for pruned weights
            for p_name, param in network.named_parameters():
                if p_name in masks:
                    param.grad.data.mul_(masks[p_name].to(compute_device))

            optimizer.step()

            # Re-enforce zero invariants post-optimizer step
            for p_name, param in network.named_parameters():
                if p_name in masks:
                    param.data.mul_(masks[p_name].to(compute_device))

            epoch_loss += loss.item()
            step_count += 1

        loss_curve.append(epoch_loss / max(1, step_count))

    return loss_curve


def calculate_top1_accuracy(network: nn.Module, test_dataloader: torch.utils.data.DataLoader, device: str = 'cpu') -> float:
    network.eval()
    network.to(device)
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            pred = network(x).argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += y.size(0)
    return total_correct / max(1, total_samples)


def profile_hardware_latency(
    network: nn.Module,
    batch_tensor: torch.Tensor,
    warmup_steps: int = 25,
    benchmark_steps: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    network.eval()
    network.to(device)
    x = batch_tensor.to(device)
    bs = x.size(0)

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = network(x)

    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_steps):
            start = time.perf_counter()
            _ = network(x)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000.0)

    return {
        'mean_latency_ms': float(np.mean(latencies)),
        'median_latency_ms': float(np.median(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'throughput_fps': float(bs / (np.mean(latencies) / 1000.0))
    }


def audit_sparse_storage(network: nn.Module, directory: str, identifier: str = 'checkpoint') -> Dict[str, Any]:
    os.makedirs(directory, exist_ok=True)
    dense_file = os.path.join(directory, f"{identifier}_dense.bin")
    compressed_file = os.path.join(directory, f"{identifier}_sparse.gz")

    torch.save(network.state_dict(), dense_file)
    raw_bytes = os.path.getsize(dense_file)

    with open(dense_file, 'rb') as f_in, gzip.open(compressed_file, 'wb') as f_out:
        f_out.writelines(f_in)
    compressed_bytes = os.path.getsize(compressed_file)

    weight_entries = extract_weight_tensors(network)
    total_w = sum(p.numel() for _, p in weight_entries)
    zero_w = sum((p.data == 0).sum().item() for _, p in weight_entries)
    effective_sparsity = zero_w / max(1, total_w)

    return {
        'raw_dense_kb': raw_bytes / 1024.0,
        'compressed_sparse_kb': compressed_bytes / 1024.0,
        'storage_compression_ratio': raw_bytes / max(1, compressed_bytes),
        'measured_sparsity': effective_sparsity
    }
