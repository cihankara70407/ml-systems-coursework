"""
Computational Latency Profiler for Inference and Backpropagation
Author: Cihan Karayigit (ID: 70407)
"""
import time
from typing import Dict, Any
import numpy as np
from .computation_graph import forward_pass, backward_pass

def profile_latency_breakdown(
    params: Dict[str, np.ndarray],
    X_batch: np.ndarray,
    Y_batch: np.ndarray,
    iterations: int = 100
) -> Dict[str, Any]:
    """
    Task 7: Benchmarks forward pass and backward pass latency separately.
    Reports median duration across 100 iterations and links theoretical GEMM counts to observed timings.
    """
    print("="*70)
    print(">> Task 7: Forward vs Backward Pass Latency Profiling (100 Iterations)")
    print("="*70)

    # Warm-up phase to eliminate JIT/cache cold start effects
    for _ in range(10):
        p, c = forward_pass(X_batch, params)
        _ = backward_pass(params, c, Y_batch)

    forward_durations = []
    backward_durations = []

    for _ in range(iterations):
        # 1. Profile Forward Pass
        t0 = time.perf_counter()
        probs, cache = forward_pass(X_batch, params)
        t1 = time.perf_counter()
        forward_durations.append((t1 - t0) * 1000.0)

        # 2. Profile Backward Pass
        t2 = time.perf_counter()
        _ = backward_pass(params, cache, Y_batch)
        t3 = time.perf_counter()
        backward_durations.append((t3 - t2) * 1000.0)

    median_fwd = float(np.median(forward_durations))
    median_bwd = float(np.median(backward_durations))
    ratio = median_bwd / median_fwd if median_fwd > 0 else 0.0

    print(f"Median Forward Latency  : {median_fwd:.4f} ms")
    print(f"Median Backward Latency : {median_bwd:.4f} ms")
    print(f"Backward-to-Forward Ratio: {ratio:.2f}x\n")

    explanation = (
        "Computational Complexity Analysis (GEMM Operations):\n"
        "The backward pass performs approximately double the dense matrix multiplications (GEMM) of the forward pass:\n"
        "  - Forward Pass requires 3 Matrix Multiplications:\n"
        "      1. Z1 = X @ W1\n"
        "      2. Z2 = A1 @ W2\n"
        "      3. Z3 = A2 @ W3\n"
        "  - Backward Pass requires 5 Matrix Multiplications:\n"
        "      1. dW3 = A2.T @ dZ3   (gradient w.r.t. parameters W3)\n"
        "      2. dA2 = dZ3 @ W3.T   (gradient backpropagated to activations A2)\n"
        "      3. dW2 = A1.T @ dZ2   (gradient w.r.t. parameters W2)\n"
        "      4. dA1 = dZ2 @ W2.T   (gradient backpropagated to activations A1)\n"
        "      5. dW1 = X.T @ dZ1    (gradient w.r.t. parameters W1)\n"
        "Because backpropagation must evaluate both parameter gradients (for weight updates) and activation gradients\n"
        "(for error transmission across layers), it requires significantly more floating-point operations (FLOPs)."
    )
    print(explanation)

    return {
        "median_forward_ms": median_fwd,
        "median_backward_ms": median_bwd,
        "ratio": ratio,
        "iterations": iterations,
        "explanation": explanation
    }
