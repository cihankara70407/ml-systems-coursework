"""
Mathematical Verification: Custom AutoDiff vs. PyTorch Autograd & Symmetric Finite Differences
Author: Cihan Karayigit (Student ID: 70407)
Laboratory 05: Frameworks and Automatic Differentiation
"""

import json
import os
import torch
from autodiff import AutoDiffScalar


def evaluate_custom_graph(a_list, b_list, c_list):
    terms = []
    for a, b, c in zip(a_list, b_list, c_list):
        # ((a * b + c) ** 2).relu()
        prod = a * b
        lin = prod + c
        sq = lin ** 2
        act = sq.relu()
        terms.append(act)
    
    total = terms[0]
    for t in terms[1:]:
        total = total + t
    return total


def symmetric_finite_difference(a_raw, b_raw, c_raw, target_arr, index, h=1e-7):
    # f(x + h) - f(x - h) / (2 * h)
    def calc_forward(a, b, c):
        acc = 0.0
        for ai, bi, ci in zip(a, b, c):
            val = (ai * bi + ci) ** 2
            acc += max(0.0, val)
        return acc

    a_plus, a_minus = list(a_raw), list(a_raw)
    b_plus, b_minus = list(b_raw), list(b_raw)
    c_plus, c_minus = list(c_raw), list(c_raw)

    if target_arr == 'a':
        a_plus[index] += h
        a_minus[index] -= h
    elif target_arr == 'b':
        b_plus[index] += h
        b_minus[index] -= h
    elif target_arr == 'c':
        c_plus[index] += h
        c_minus[index] -= h

    f_hi = calc_forward(a_plus, b_plus, c_plus)
    f_lo = calc_forward(a_minus, b_minus, c_minus)
    return (f_hi - f_lo) / (2.0 * h)


def verify_all():
    print("[AUTODIFF VERIFICATION] Starting verification for Cihan Karayigit (70407)...")
    # Multi-element test values covering distinct quadrants
    a_init = [2.25, -1.75, 0.50, -3.10]
    b_init = [-1.50, 3.20, -2.40, 1.10]
    c_init = [0.85, -2.10, 1.30, 4.25]

    # 1. Custom Reverse-Mode AD
    a_nodes = [AutoDiffScalar(v, annotation=f'a_{i}') for i, v in enumerate(a_init)]
    b_nodes = [AutoDiffScalar(v, annotation=f'b_{i}') for i, v in enumerate(b_init)]
    c_nodes = [AutoDiffScalar(v, annotation=f'c_{i}') for i, v in enumerate(c_init)]

    custom_loss = evaluate_custom_graph(a_nodes, b_nodes, c_nodes)
    custom_loss.backward()

    # 2. PyTorch Autograd Engine
    t_a = torch.tensor(a_init, dtype=torch.float64, requires_grad=True)
    t_b = torch.tensor(b_init, dtype=torch.float64, requires_grad=True)
    t_c = torch.tensor(c_init, dtype=torch.float64, requires_grad=True)

    t_loss = (((t_a * t_b + t_c) ** 2).relu()).sum()
    t_loss.backward()

    # 3. Symmetric Finite Difference Calculations
    max_torch_err = 0.0
    max_num_err = 0.0
    records = []

    for name, nodes, torch_t in [('a', a_nodes, t_a), ('b', b_nodes, t_b), ('c', c_nodes, t_c)]:
        for i in range(len(a_init)):
            c_grad = nodes[i].grad
            p_grad = torch_t.grad[i].item()
            num_grad = symmetric_finite_difference(a_init, b_init, c_init, name, i)

            torch_delta = abs(c_grad - p_grad)
            num_delta = abs(c_grad - num_grad)

            max_torch_err = max(max_torch_err, torch_delta)
            max_num_err = max(max_num_err, num_delta)

            records.append({
                'symbol': f"{name}_{i}",
                'custom_grad': c_grad,
                'pytorch_grad': p_grad,
                'finite_diff_grad': num_grad,
                'abs_diff_pytorch': torch_delta,
                'abs_diff_fd': num_delta
            })

    passed = (max_torch_err < 1e-6)
    print(f"  Max Absolute Difference vs PyTorch: {max_torch_err:.2e}")
    print(f"  Max Absolute Difference vs FD:      {max_num_err:.2e}")
    print(f"  Verification Result:                {'PASS (< 1e-6)' if passed else 'FAIL'}")

    summary = {
        'student': 'Cihan Karayigit (70407)',
        'expression': '((a * b + c) ** 2).relu().sum()',
        'forward_val_custom': custom_loss.data,
        'forward_val_pytorch': t_loss.item(),
        'max_pytorch_delta': max_torch_err,
        'max_finite_difference_delta': max_num_err,
        'threshold': 1e-6,
        'status': 'PASSED' if passed else 'FAILED',
        'telemetry': records
    }

    os.makedirs('results', exist_ok=True)
    with open('results/gradient_audit.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == '__main__':
    verify_all()
