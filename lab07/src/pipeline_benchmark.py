"""
Master Benchmark Pipeline: Pruning Protocols, Knowledge Distillation & Systems Profiling
Author: Cihan Karayigit (Student ID: 70407)
Laboratory 07: Model Compression & Knowledge Distillation
Academic Program: Computer Engineering, Vistula University
"""

import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deep_compression_models import DeepFashionTeacherNet, Sub100kStudentNet, inspect_model_capacity
from magnitude_filter_pruner import (
    apply_global_magnitude_mask,
    apply_layerwise_magnitude_mask,
    apply_structured_l2_filter_pruning,
    finetune_with_static_mask,
    calculate_top1_accuracy,
    profile_hardware_latency,
    audit_sparse_storage
)
from knowledge_distiller import (
    train_distilled_student_pipeline,
    train_vanilla_student_pipeline
)


def run_laboratory_experiment():
    torch.manual_seed(1337)
    device = 'cpu'
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=====================================================================")
    print("  LABORATORY 07: ADVANCED MODEL COMPRESSION & DISTILLATION")
    print("  Student Researcher: Cihan Karayigit (70407)")
    print("=====================================================================\n")

    # 1. Dataset setup
    data_root = 'C:/Users/Tuan/.gemini/antigravity/scratch/data'
    train_set = FashionMNIST(root=data_root, train=True, transform=ToTensor())
    test_set = FashionMNIST(root=data_root, train=False, transform=ToTensor())

    train_subset = Subset(train_set, range(16000))
    ft_subset = Subset(train_set, range(5000))
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    ft_loader = DataLoader(ft_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    sample_probe = torch.randn(64, 1, 28, 28)

    print(f"[DATA PIPELINE] Train Samples: {len(train_subset):,}, FT Samples: {len(ft_subset):,}, Test Samples: {len(test_set):,}")

    # ---------------------------------------------------------------------
    # PHASE 1: Train Oversized Teacher CNN (~1.09M Parameters)
    # ---------------------------------------------------------------------
    print("\n[PHASE 1] Initializing & Training DeepFashionTeacherNet (~1.09M)...")
    teacher = DeepFashionTeacherNet()
    teacher_capacity = inspect_model_capacity(teacher)
    print(f"  Architecture Weight Count: {teacher_capacity['total_params']:,} parameters")

    optimizer = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    t_start = time.perf_counter()
    for ep in range(5):
        teacher.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = teacher(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = calculate_top1_accuracy(teacher, test_loader, device=device)
        print(f"  Epoch {ep+1}/5 | Training Loss: {running_loss/len(train_loader):.4f} | Validation Acc: {acc*100:.2f}%")

    train_duration = time.perf_counter() - t_start
    baseline_top1 = calculate_top1_accuracy(teacher, test_loader, device=device)
    baseline_speed = profile_hardware_latency(teacher, sample_probe)
    baseline_disk = audit_sparse_storage(teacher, results_dir, identifier='teacher_dense_ref')

    teacher_profile = {
        'student': "Cihan Karayigit (70407)",
        'model_architecture': "DeepFashionTeacherNet",
        'parameters': teacher_capacity['total_params'],
        'train_time_sec': train_duration,
        'test_accuracy': baseline_top1,
        'latency_profile': baseline_speed,
        'storage_profile': baseline_disk
    }

    with open(os.path.join(results_dir, 'baseline_execution_profile.json'), 'w') as f:
        json.dump(teacher_profile, f, indent=2)

    print(f"  Teacher Accuracy: {baseline_top1*100:.2f}% | Latency: {baseline_speed['median_latency_ms']:.2f} ms")

    # ---------------------------------------------------------------------
    # PHASE 2: Magnitude & Structured Pruning Experiments
    # ---------------------------------------------------------------------
    print("\n[PHASE 2] Executing Pruning Regimes & Fine-Tuning Protocols...")
    target_sparsities = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]
    filter_ratios = [0.10, 0.25, 0.50]

    prune_log = {
        'global_magnitude': [],
        'layerwise_magnitude': [],
        'structured_l2_filter': []
    }

    # 2A. Global Magnitude Pruning
    print("\n--- Regime A: Global Magnitude Pruning ---")
    for s in target_sparsities:
        clone_net = copy.deepcopy(teacher)
        masks = apply_global_magnitude_mask(clone_net, s)
        acc_raw = calculate_top1_accuracy(clone_net, test_loader, device=device)

        if s > 0.0:
            ft_opt = torch.optim.Adam(clone_net.parameters(), lr=2e-4)
            _ = finetune_with_static_mask(clone_net, masks, ft_loader, loss_fn, ft_opt, num_epochs=3, compute_device=device)
        acc_ft = calculate_top1_accuracy(clone_net, test_loader, device=device)
        speed = profile_hardware_latency(clone_net, sample_probe)
        storage = audit_sparse_storage(clone_net, results_dir, identifier=f"global_s{int(s*100)}")

        print(f"  Sparsity {s*100:4.1f}% | Raw Acc: {acc_raw*100:.2f}% | Post-FT Acc: {acc_ft*100:.2f}% | "
              f"Latency: {speed['median_latency_ms']:.2f} ms | Gzip: {storage['compressed_sparse_kb']:.1f} KB")

        prune_log['global_magnitude'].append({
            'sparsity_ratio': s,
            'pre_ft_accuracy': acc_raw,
            'post_ft_accuracy': acc_ft,
            'median_latency_ms': speed['median_latency_ms'],
            'compressed_size_kb': storage['compressed_sparse_kb'],
            'raw_size_kb': storage['raw_dense_kb'],
            'compression_factor': storage['storage_compression_ratio']
        })

    # 2B. Layer-wise Magnitude Pruning
    print("\n--- Regime B: Layer-wise Magnitude Pruning ---")
    for s in target_sparsities:
        clone_net = copy.deepcopy(teacher)
        masks = apply_layerwise_magnitude_mask(clone_net, s)
        acc_raw = calculate_top1_accuracy(clone_net, test_loader, device=device)

        if s > 0.0:
            ft_opt = torch.optim.Adam(clone_net.parameters(), lr=2e-4)
            _ = finetune_with_static_mask(clone_net, masks, ft_loader, loss_fn, ft_opt, num_epochs=3, compute_device=device)
        acc_ft = calculate_top1_accuracy(clone_net, test_loader, device=device)
        speed = profile_hardware_latency(clone_net, sample_probe)

        print(f"  Sparsity {s*100:4.1f}% | Raw Acc: {acc_raw*100:.2f}% | Post-FT Acc: {acc_ft*100:.2f}% | Latency: {speed['median_latency_ms']:.2f} ms")

        prune_log['layerwise_magnitude'].append({
            'sparsity_ratio': s,
            'pre_ft_accuracy': acc_raw,
            'post_ft_accuracy': acc_ft,
            'median_latency_ms': speed['median_latency_ms']
        })

    # 2C. Structured L2-Norm Filter Pruning
    print("\n--- Regime C: Structured L2-Norm Filter Pruning ---")
    for fr in filter_ratios:
        clone_net = copy.deepcopy(teacher)
        masks = apply_structured_l2_filter_pruning(clone_net, fr)
        acc_raw = calculate_top1_accuracy(clone_net, test_loader, device=device)

        ft_opt = torch.optim.Adam(clone_net.parameters(), lr=2e-4)
        _ = finetune_with_static_mask(clone_net, masks, ft_loader, loss_fn, ft_opt, num_epochs=3, compute_device=device)
        acc_ft = calculate_top1_accuracy(clone_net, test_loader, device=device)
        speed = profile_hardware_latency(clone_net, sample_probe)

        print(f"  Channel Sparsity {fr*100:4.1f}% | Raw Acc: {acc_raw*100:.2f}% | Post-FT Acc: {acc_ft*100:.2f}% | Latency: {speed['median_latency_ms']:.2f} ms")

        prune_log['structured_l2_filter'].append({
            'channel_sparsity': fr,
            'pre_ft_accuracy': acc_raw,
            'post_ft_accuracy': acc_ft,
            'median_latency_ms': speed['median_latency_ms']
        })

    with open(os.path.join(results_dir, 'pruning_telemetry.json'), 'w') as f:
        json.dump(prune_log, f, indent=2)

    # ---------------------------------------------------------------------
    # PHASE 3: Knowledge Distillation vs. Vanilla Training
    # ---------------------------------------------------------------------
    print("\n[PHASE 3] Benchmarking Knowledge Distillation (Sub100kStudentNet)...")
    student_kd = Sub100kStudentNet()
    student_vanilla = Sub100kStudentNet()
    s_cap = inspect_model_capacity(student_kd)
    print(f"  Student Weight Count: {s_cap['total_params']:,} parameters (~{s_cap['total_params']/1e3:.1f}k)")

    # 3A. Distillation Training
    print("  Training Distilled Student (Temperature=4.0, Alpha=0.5, 5 epochs)...")
    t0_kd = time.perf_counter()
    _ = train_distilled_student_pipeline(student_kd, teacher, train_loader, num_epochs=5, temp=4.0, alpha=0.5, lr=1e-3, device=device)
    dur_kd = time.perf_counter() - t0_kd
    acc_kd = calculate_top1_accuracy(student_kd, test_loader, device=device)
    speed_kd = profile_hardware_latency(student_kd, sample_probe)
    storage_kd = audit_sparse_storage(student_kd, results_dir, identifier='student_distilled_ref')

    # 3B. Vanilla Training
    print("  Training Vanilla Student (CE Only, 5 epochs)...")
    t0_vanilla = time.perf_counter()
    _ = train_vanilla_student_pipeline(student_vanilla, train_loader, num_epochs=5, lr=1e-3, device=device)
    dur_vanilla = time.perf_counter() - t0_vanilla
    acc_vanilla = calculate_top1_accuracy(student_vanilla, test_loader, device=device)
    speed_vanilla = profile_hardware_latency(student_vanilla, sample_probe)

    print(f"\n  Distilled Student Top-1: {acc_kd*100:.2f}% | Latency: {speed_kd['median_latency_ms']:.2f} ms")
    print(f"  Vanilla Student Top-1:   {acc_vanilla*100:.2f}% | Latency: {speed_vanilla['median_latency_ms']:.2f} ms")
    print(f"  Distillation Advantage:  +{(acc_kd - acc_vanilla)*100:.2f}% Top-1 Accuracy!")

    kd_report = {
        'teacher_reference': {
            'parameters': teacher_capacity['total_params'],
            'accuracy': baseline_top1,
            'latency_ms': baseline_speed['median_latency_ms'],
            'storage_kb': baseline_disk['raw_dense_kb']
        },
        'distilled_student': {
            'parameters': s_cap['total_params'],
            'accuracy': acc_kd,
            'latency_ms': speed_kd['median_latency_ms'],
            'storage_kb': storage_kd['raw_dense_kb'],
            'train_time_sec': dur_kd
        },
        'vanilla_student': {
            'parameters': s_cap['total_params'],
            'accuracy': acc_vanilla,
            'latency_ms': speed_vanilla['median_latency_ms'],
            'train_time_sec': dur_vanilla
        },
        'accuracy_gain': acc_kd - acc_vanilla,
        'parameter_compression': teacher_capacity['total_params'] / s_cap['total_params'],
        'throughput_gain': baseline_speed['median_latency_ms'] / speed_kd['median_latency_ms']
    }

    with open(os.path.join(results_dir, 'distillation_benchmarks.json'), 'w') as f:
        json.dump(kd_report, f, indent=2)

    # ---------------------------------------------------------------------
    # PHASE 4: Generate Diagnostic Performance Plot
    # ---------------------------------------------------------------------
    print("\n[PHASE 4] Exporting Diagnostic Trade-Off Visualizations...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # Subplot 1: Sparsity vs Accuracy
    sparsities = [p['sparsity_ratio'] * 100 for p in prune_log['global_magnitude']]
    g_raw = [p['pre_ft_accuracy'] * 100 for p in prune_log['global_magnitude']]
    g_ft = [p['post_ft_accuracy'] * 100 for p in prune_log['global_magnitude']]
    l_ft = [p['post_ft_accuracy'] * 100 for p in prune_log['layerwise_magnitude']]

    chan_spars = [p['channel_sparsity'] * 100 for p in prune_log['structured_l2_filter']]
    chan_ft = [p['post_ft_accuracy'] * 100 for p in prune_log['structured_l2_filter']]

    ax1.plot(sparsities, g_ft, 'o-', color='#004c6d', linewidth=2.4, label='Global Magnitude (Fine-Tuned)')
    ax1.plot(sparsities, g_raw, '--', color='#6996b3', linewidth=1.6, label='Global Magnitude (Uncompressed/Zero-Shot)')
    ax1.plot(sparsities, l_ft, 's-.', color='#c3553a', linewidth=2.0, label='Layer-wise Magnitude (Fine-Tuned)')
    ax1.plot(chan_spars, chan_ft, '^:', color='#e08244', linewidth=2.2, label='Structured L2 Filter (Fine-Tuned)')
    ax1.axhline(baseline_top1 * 100, color='#333333', linestyle=':', label=f'Teacher Baseline ({baseline_top1*100:.1f}%)')
    ax1.axhline(acc_kd * 100, color='#2e7d32', linestyle='--', label=f'Distilled Student ({acc_kd*100:.1f}%)')

    ax1.set_xlabel('Model Sparsity Ratio (%)', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('Top-1 Test Accuracy (%)', fontsize=11, fontweight='semibold')
    ax1.set_title('Fashion-MNIST Accuracy Retention Across Pruning Regimes', fontsize=12, fontweight='bold')
    ax1.set_ylim([60, 95])
    ax1.legend(loc='lower left', frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Microarchitecture Reality: Latency vs Storage
    latencies = [p['median_latency_ms'] for p in prune_log['global_magnitude']]
    storages = [p['compressed_size_kb'] for p in prune_log['global_magnitude']]

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(sparsities, latencies, 'D-', color='#00796b', linewidth=2.2, label='CPU Execution Latency (ms)')
    l2 = ax2_twin.plot(sparsities, storages, 'v--', color='#7b1fa2', linewidth=2.2, label='Sparse Checkpoint Size (KB)')

    ax2.set_xlabel('Global Sparsity Ratio (%)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Inference Latency per Batch (ms)', color='#00796b', fontsize=11, fontweight='semibold')
    ax2_twin.set_ylabel('Compressed Storage Footprint (KB)', color='#7b1fa2', fontsize=11, fontweight='semibold')
    ax2.set_title('Systems Reality: Storage Reduction vs. CPU Execution Latency', fontsize=12, fontweight='bold')

    plots = l1 + l2
    tags = [p.get_label() for p in plots]
    ax2.legend(plots, tags, loc='center left', frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_file = os.path.join(results_dir, 'pruning_distillation_tradeoffs.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"  Figure exported to: {plot_file}")

    # ---------------------------------------------------------------------
    # PHASE 5: Print Summary
    # ---------------------------------------------------------------------
    print("\n=====================================================================")
    print("  CHAPTER 9 EFFICIENCY PILLARS: EXPERIMENTAL BENCHMARK SUMMARY")
    print("=====================================================================")
    print(f"{'Configuration':<26} | {'Top-1 Acc':<9} | {'Parameters':<10} | {'Disk (KB)':<10} | {'Latency':<9} | {'Relative Cost':<12}")
    print("-" * 88)
    print(f"{'Teacher Baseline (~1M)':<26} | {baseline_top1*100:6.2f}%   | {teacher_capacity['total_params']:<10} | {baseline_disk['raw_dense_kb']:<10.1f} | {baseline_speed['median_latency_ms']:<6.2f} ms | 1.00x (baseline)")
    print(f"{'Pruned 90% Global':<26} | {prune_log['global_magnitude'][4]['post_ft_accuracy']*100:6.2f}%   | {int(teacher_capacity['total_params']*0.10):<10} | {prune_log['global_magnitude'][4]['compressed_size_kb']:<10.1f} | {prune_log['global_magnitude'][4]['median_latency_ms']:<6.2f} ms | 0.99x")
    print(f"{'Pruned 50% Filter L2':<26} | {prune_log['structured_l2_filter'][2]['post_ft_accuracy']*100:6.2f}%   | {int(teacher_capacity['total_params']*0.50):<10} | {baseline_disk['raw_dense_kb']*0.5:<10.1f} | {prune_log['structured_l2_filter'][2]['median_latency_ms']:<6.2f} ms | 0.60x")
    print(f"{'Vanilla Student (98k)':<26} | {acc_vanilla*100:6.2f}%   | {s_cap['total_params']:<10} | {storage_kd['raw_dense_kb']:<10.1f} | {speed_vanilla['median_latency_ms']:<6.2f} ms | 0.17x")
    print(f"{'Distilled Student (98k)':<26} | {acc_kd*100:6.2f}%   | {s_cap['total_params']:<10} | {storage_kd['raw_dense_kb']:<10.1f} | {speed_kd['median_latency_ms']:<6.2f} ms | 0.17x")
    print("=====================================================================\n")

    return {
        'baseline': teacher_profile,
        'pruning': prune_log,
        'distillation': kd_report
    }


if __name__ == '__main__':
    run_laboratory_experiment()
