"""
Experiment Runner & Complete Evaluation Pipeline
Author: Cihan Karayigit (ID: 70407)
"""
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from engine import (
    initialize_network_parameters,
    forward_pass,
    cross_entropy_objective,
    backward_pass,
    apply_gradients,
    predict_classes,
    calculate_accuracy,
    verify_analytical_gradients,
    profile_latency_breakdown
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_and_preprocess_dataset():
    """
    Task 1: Load load_digits, scale inputs / 16.0, one-hot encode labels,
    and split 70/30 stratified with random_state=42.
    """
    logger.info("Loading scikit-learn digits dataset...")
    X_raw, y_raw = load_digits(return_X_y=True)

    # Scale inputs by dividing by 16.0
    X_scaled = X_raw.astype(np.float64) / 16.0

    # One-hot encode targets
    num_classes = 10
    Y_onehot = np.eye(num_classes)[y_raw]

    # Stratified 70/30 split
    X_train, X_test, Y_train, Y_test, y_train_idx, y_test_idx = train_test_split(
        X_scaled, Y_onehot, y_raw, test_size=0.30, random_state=42, stratify=y_raw
    )

    logger.info(f"Dataset partitioned: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples.")
    return X_train, X_test, Y_train, Y_test, y_train_idx, y_test_idx


def train_single_configuration(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    y_test_labels: np.ndarray,
    hidden_width: int,
    learning_rate: float,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42
):
    """
    Executes mini-batch SGD training loop for 50 epochs with shuffling and per-epoch logging.
    """
    logger.info(f"\n--- Initiating Run: Width={hidden_width}, Learning Rate={learning_rate} ---")
    params = initialize_network_parameters(
        input_features=64, hidden_dim=hidden_width, output_classes=10, seed=seed
    )

    num_samples = X_train.shape[0]
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    rng = np.random.RandomState(seed)

    for epoch in range(1, epochs + 1):
        # Mini-batch shuffling
        indices = rng.permutation(num_samples)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        # Mini-batch gradient descent
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            Y_batch = Y_shuffled[start_idx:end_idx]

            probs_b, cache_b = forward_pass(X_batch, params)
            grads = backward_pass(params, cache_b, Y_batch)
            params = apply_gradients(params, grads, learning_rate)

        # Full-dataset epoch metrics evaluation
        train_probs, _ = forward_pass(X_train, params)
        train_loss = cross_entropy_objective(train_probs, Y_train)

        test_probs, _ = forward_pass(X_test, params)
        test_loss = cross_entropy_objective(test_probs, Y_test)
        test_preds = np.argmax(test_probs, axis=1)
        test_acc = calculate_accuracy(y_test_labels, test_preds)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            logger.info(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    return params, history


def main():
    os.makedirs("results", exist_ok=True)

    # 1. Dataset loading & stratified 70/30 split
    X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels = load_and_preprocess_dataset()

    # 2. Sanity Checks: Forward Pass Integrity & Untrained Cross-Entropy Loss
    print("\n" + "="*70)
    print(">> Tasks 3 & 4: Forward Pass Shapes & Untrained Loss Validation")
    print("="*70)
    test_params = initialize_network_parameters(input_features=64, hidden_dim=16, output_classes=10, seed=42)
    sample_X = X_train[:32]
    sample_Y = Y_train[:32]

    probs, _ = forward_pass(sample_X, test_params)
    print(f"Probabilities matrix shape: {probs.shape} (Expected: (32, 10))")
    row_sums = np.sum(probs, axis=1)
    print(f"Softmax row sums range   : [{np.min(row_sums):.8f}, {np.max(row_sums):.8f}]")
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Softmax probabilities do not sum to 1!"
    print("[PASS] Forward pass layer-by-layer evaluation verified.")

    untrained_loss = cross_entropy_objective(probs, sample_Y)
    expected_untrained_loss = np.log(10.0)
    print(f"Untrained network loss   : {untrained_loss:.6f} (Theoretical ln(10) = {expected_untrained_loss:.6f})")
    assert abs(untrained_loss - expected_untrained_loss) < 0.05, "Untrained loss deviated from theoretical ln(10)"
    print("[PASS] Untrained loss sanity check successfully verified.")

    # 3. Task 5: Numerical Gradient Verification
    grad_results = verify_analytical_gradients(test_params, sample_X, sample_Y, epsilon=1e-5, num_tests=10, sampling_seed=8)

    # 4. Task 7: Latency Profiling (100 runs)
    profiling_results = profile_latency_breakdown(test_params, sample_X, sample_Y, iterations=100)

    # 5. Task 6: Grid Search (3 Learning Rates x 2 Hidden Widths)
    learning_rates = [0.001, 0.01, 0.1]
    hidden_widths = [16, 64]

    experiment_records = {}

    for width in hidden_widths:
        for lr in learning_rates:
            key = f"Width_{width}__LR_{lr}"
            params, hist = train_single_configuration(
                X_train, Y_train, X_test, Y_test, y_test_labels,
                hidden_width=width, learning_rate=lr, epochs=50, batch_size=32, seed=42
            )
            experiment_records[key] = {
                "hidden_width": width,
                "learning_rate": lr,
                "final_train_loss": float(hist["train_loss"][-1]),
                "final_test_loss": float(hist["test_loss"][-1]),
                "final_test_accuracy": float(hist["test_acc"][-1]),
                "history": {
                    "train_loss": [float(v) for v in hist["train_loss"]],
                    "test_loss": [float(v) for v in hist["test_loss"]],
                    "test_acc": [float(v) for v in hist["test_acc"]]
                }
            }

    # 6. Plotting Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    colors = {
        0.001: "#2563EB", # Blue
        0.01:  "#059669", # Emerald
        0.1:   "#DC2626"  # Crimson
    }

    for idx, width in enumerate(hidden_widths):
        ax = axes[idx]
        ax.set_title(f"Hidden Width W = {width} (Categorical Cross-Entropy)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch Number", fontsize=11)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax.set_ylim(-0.05, 2.5)
        ax.grid(True, linestyle=":", alpha=0.6)

        epochs_axis = range(1, 51)
        for lr in learning_rates:
            run_key = f"Width_{width}__LR_{lr}"
            run_hist = experiment_records[run_key]["history"]
            c = colors[lr]
            ax.plot(epochs_axis, run_hist["train_loss"], label=f"Train (lr={lr})", color=c, linewidth=2.0)
            ax.plot(epochs_axis, run_hist["test_loss"], label=f"Test (lr={lr})", color=c, linestyle="--", linewidth=1.6, alpha=0.85)

        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

    plt.suptitle("Deep Learning from Scratch: Training & Testing Loss Curves (70407 Cihan Karayigit)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    chart_output = "results/loss_curves.png"
    plt.savefig(chart_output, dpi=300)
    plt.close()
    logger.info(f"Loss curves saved to: {chart_output}")

    # 7. Print Final Performance Summary Table
    print("\n" + "="*80)
    print(">> Evaluation Grid Summary: 6 Architectural Configurations")
    print("="*80)
    print(f"{'Config ID':22s} | {'Width':6s} | {'Learning Rate':14s} | {'Final Test Loss':15s} | {'Test Accuracy':12s}")
    print("-" * 80)
    for k, v in experiment_records.items():
        print(f"{k:22s} | {v['hidden_width']:<6d} | {v['learning_rate']:<14.3f} | {v['final_test_loss']:<15.4f} | {v['final_test_accuracy']*100:.2f}%")

    # 8. Save Machine-Readable JSON Report
    report_data = {
        "student_info": {
            "name": "Cihan Karayigit",
            "student_id": "70407",
            "assignment": "Lab 2: A neural network from scratch (Ch. 3)"
        },
        "gradient_verification": grad_results,
        "latency_profiling": profiling_results,
        "grid_experiments": experiment_records
    }
    with open("results/metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    logger.info("Complete metrics report saved to: results/metrics_report.json")


if __name__ == "__main__":
    main()
