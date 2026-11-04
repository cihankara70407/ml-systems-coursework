"""
PyTorch Execution Modes Profiling: Eager vs. Traced/Compiled & Memory Contiguity
Author: Cihan Karayigit (Student ID: 70407)
Laboratory 05: Frameworks and Automatic Differentiation
"""

import time
import json
import os
import torch
import torch.nn as nn


class DifferentiatedMoonsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine_1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.affine_2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.affine_1(x))
        return self.sigmoid(self.affine_2(h))


def profile_execution_and_memory():
    print("[PYTORCH PROFILER] Starting Execution and Contiguity Audit for Cihan Karayigit...")
    model = DifferentiatedMoonsModel().eval()
    sample_batch = torch.randn(32, 2, dtype=torch.float32)

    # 1. Traced Graph Generation
    traced_graph = torch.jit.trace(model, sample_batch)
    graph_representation = str(traced_graph.inlined_graph)

    # 2. Warm-up and Timed Execution (1000 iterations)
    warmup_cycles = 150
    test_cycles = 1000

    for _ in range(warmup_cycles):
        _ = model(sample_batch)
        _ = traced_graph(sample_batch)

    # Eager Mode
    t_start = time.perf_counter()
    for _ in range(test_cycles):
        _ = model(sample_batch)
    eager_elapsed = (time.perf_counter() - t_start) / test_cycles * 1e6

    # Traced Mode
    t_start = time.perf_counter()
    for _ in range(test_cycles):
        _ = traced_graph(sample_batch)
    traced_elapsed = (time.perf_counter() - t_start) / test_cycles * 1e6

    # Compiled Mode (Inductor)
    compiled_elapsed = None
    try:
        compiled_net = torch.compile(model)
        for _ in range(warmup_cycles):
            _ = compiled_net(sample_batch)
        t_start = time.perf_counter()
        for _ in range(test_cycles):
            _ = compiled_net(sample_batch)
        compiled_elapsed = (time.perf_counter() - t_start) / test_cycles * 1e6
        compiled_status = "Available and functional"
    except Exception as exc:
        compiled_status = f"Unsupported ({type(exc).__name__})"

    print(f"  Eager Latency:   {eager_elapsed:.2f} µs")
    print(f"  Traced Latency:  {traced_elapsed:.2f} µs (Speedup: {eager_elapsed / traced_elapsed:.2f}x)")
    if compiled_elapsed:
        print(f"  Compiled Latency:{compiled_elapsed:.2f} µs (Speedup: {eager_elapsed / compiled_elapsed:.2f}x)")

    # 3. Contiguity and Stride Memory Investigation
    print("\n[TENSOR STRIDE BENCHMARK] Evaluating physical layout effects...")
    DIM = 2048
    contiguous_tensor = torch.randn(DIM, DIM, dtype=torch.float32)
    # Transpose produces a non-contiguous view with swapped strides
    transposed_tensor = contiguous_tensor.t()

    c_meta = {
        'shape': list(contiguous_tensor.shape),
        'dtype': str(contiguous_tensor.dtype),
        'stride': list(contiguous_tensor.stride()),
        'is_contiguous': contiguous_tensor.is_contiguous()
    }
    nc_meta = {
        'shape': list(transposed_tensor.shape),
        'dtype': str(transposed_tensor.dtype),
        'stride': list(transposed_tensor.stride()),
        'is_contiguous': transposed_tensor.is_contiguous()
    }

    # Measure reduction across columns (dim=0)
    repeat_ops = 250
    for _ in range(25):
        _ = torch.sum(contiguous_tensor, dim=1)
        _ = torch.sum(transposed_tensor, dim=1)

    t0 = time.perf_counter()
    for _ in range(repeat_ops):
        _ = torch.sum(contiguous_tensor, dim=1)
    time_contig = ((time.perf_counter() - t0) / repeat_ops) * 1000.0

    t0 = time.perf_counter()
    for _ in range(repeat_ops):
        _ = torch.sum(transposed_tensor, dim=1)
    time_non_contig = ((time.perf_counter() - t0) / repeat_ops) * 1000.0

    t0 = time.perf_counter()
    for _ in range(repeat_ops):
        _ = transposed_tensor.contiguous()
    time_copy = ((time.perf_counter() - t0) / repeat_ops) * 1000.0

    slowdown = time_non_contig / time_contig
    print(f"  Contiguous Sum:     {time_contig:.3f} ms")
    print(f"  Non-Contiguous Sum: {time_non_contig:.3f} ms")
    print(f"  Stride Penalty:     {slowdown:.2f}x")
    print(f"  Memory Clone Cost:  {time_copy:.3f} ms")

    audit = {
        'student': 'Cihan Karayigit (70407)',
        'latencies_microsec': {
            'eager': eager_elapsed,
            'traced': traced_elapsed,
            'compiled': compiled_elapsed,
            'speedup_traced': eager_elapsed / traced_elapsed,
            'speedup_compiled': (eager_elapsed / compiled_elapsed) if compiled_elapsed else None
        },
        'memory_strides': {
            'contiguous': c_meta,
            'non_contiguous': nc_meta,
            'reduction_time_contiguous_ms': time_contig,
            'reduction_time_non_contiguous_ms': time_non_contig,
            'stride_slowdown_factor': slowdown,
            'contiguous_reallocation_cost_ms': time_copy
        },
        'traced_ir_snippet': graph_representation[:800]
    }

    os.makedirs('results', exist_ok=True)
    with open('results/pytorch_runtime_audit.json', 'w') as f:
        json.dump(audit, f, indent=2)

    return audit


if __name__ == '__main__':
    profile_execution_and_memory()
