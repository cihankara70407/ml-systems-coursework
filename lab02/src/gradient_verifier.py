"""
Numerical Gradient Verifier for Backpropagation Integrity
Author: Cihan Karayigit (ID: 70407)
"""
from typing import Dict, List, Any
import numpy as np
from .computation_graph import forward_pass, cross_entropy_objective, backward_pass

def verify_analytical_gradients(
    params: Dict[str, np.ndarray],
    X_sample: np.ndarray,
    Y_sample: np.ndarray,
    epsilon: float = 1e-5,
    num_tests: int = 10,
    sampling_seed: int = 8
) -> Dict[str, Any]:
    """
    Task 5 Verification:
    Nudges 10 random parameters by +/- epsilon, evaluates two-sided finite difference slopes,
    and compares them to analytical backpropagation gradients.
    Enforces relative error < 1e-6 for at least 9 of 10 sampled parameters.
    """
    print("\n" + "="*70)
    print(">> Task 5: Numerical Gradient Consistency Verification")
    print("="*70)
    print(f"Finite difference step (epsilon) : {epsilon}")
    print(f"Batch sample count               : {X_sample.shape[0]}")
    
    # 1. Compute analytical gradients on the intact model
    probs, cache = forward_pass(X_sample, params)
    analytical_grads = backward_pass(params, cache, Y_sample)

    target_layers = ["W1", "W2", "W3"]
    records = []
    passed_count = 0

    rng = np.random.RandomState(sampling_seed)

    for idx in range(num_tests):
        # Rotate among target weight tensors
        layer_name = target_layers[idx % len(target_layers)]
        W = params[layer_name]
        dW = analytical_grads[f"d{layer_name}"]

        # Sample random entry coordinates
        row = rng.randint(0, W.shape[0])
        col = rng.randint(0, W.shape[1])

        analytic_val = float(dW[row, col])

        # Positive perturbation: +epsilon
        W[row, col] += epsilon
        probs_pos, _ = forward_pass(X_sample, params)
        loss_pos = cross_entropy_objective(probs_pos, Y_sample)

        # Negative perturbation: -epsilon
        W[row, col] -= 2.0 * epsilon
        probs_neg, _ = forward_pass(X_sample, params)
        loss_neg = cross_entropy_objective(probs_neg, Y_sample)

        # Restore original parameter value
        W[row, col] += epsilon

        # Central difference slope approximation
        numeric_val = float((loss_pos - loss_neg) / (2.0 * epsilon))

        # Relative difference metric
        normalizer = max(abs(analytic_val), abs(numeric_val), 1e-12)
        rel_diff = abs(analytic_val - numeric_val) / normalizer

        is_passed = bool(rel_diff < 1e-6)
        if is_passed:
            passed_count += 1

        records.append({
            "test_id": idx + 1,
            "parameter": f"{layer_name}[{row}, {col}]",
            "analytical": analytic_val,
            "numerical": numeric_val,
            "relative_error": float(rel_diff),
            "status": "PASS" if is_passed else "FAIL"
        })

        flag = "[PASS]" if is_passed else "[WARN]"
        print(f"Test {idx+1:02d} | {layer_name}[{row:2d}, {col:2d}] | Analytic: {analytic_val:+.7e} | Numeric: {numeric_val:+.7e} | RelErr: {rel_diff:.2e} {flag}")

    print(f"\nVerification Summary: {passed_count}/{num_tests} passed tolerance (< 1e-6).")
    assert passed_count >= 9, f"Gradient verification failed! Only {passed_count}/10 passed."
    print(">> [SUCCESS] Analytical backpropagation mathematically confirmed via finite differences!\n")

    return {
        "passed_tests": passed_count,
        "total_tests": num_tests,
        "details": records
    }
