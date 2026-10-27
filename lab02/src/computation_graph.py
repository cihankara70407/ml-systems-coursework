"""
Vectorized Computational Graph & Core Primitives
Author: Cihan Karayigit (ID: 70407)
"""
from typing import Dict, Tuple
import numpy as np

def initialize_network_parameters(
    input_features: int = 64,
    hidden_dim: int = 16,
    output_classes: int = 10,
    std_dev: float = 0.01,
    seed: int = None
) -> Dict[str, np.ndarray]:
    """
    Allocates and initializes network weight tensors and bias vectors.
    Weights ~ N(0, 0.01^2), Biases = 0.
    Shapes:
      W1: (64, W), b1: (W,)
      W2: (W, W),  b2: (W,)
      W3: (W, 10), b3: (10,)
    """
    if seed is not None:
        np.random.seed(seed)
        
    parameters = {
        "W1": np.random.normal(0.0, std_dev, size=(input_features, hidden_dim)),
        "b1": np.zeros(shape=(hidden_dim,)),
        "W2": np.random.normal(0.0, std_dev, size=(hidden_dim, hidden_dim)),
        "b2": np.zeros(shape=(hidden_dim,)),
        "W3": np.random.normal(0.0, std_dev, size=(hidden_dim, output_classes)),
        "b3": np.zeros(shape=(output_classes,))
    }
    return parameters


def forward_pass(
    X: np.ndarray,
    params: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Executes sequential forward evaluation through affine and nonlinear transformations:
      Z1 = X @ W1 + b1
      A1 = ReLU(Z1)
      Z2 = A1 @ W2 + b2
      A2 = ReLU(Z2)
      Z3 = A2 @ W3 + b3
      P  = Softmax(Z3)
    Returns probabilities P and computational cache for backpropagation.
    """
    # Affine Layer 1 & ReLU Activation
    Z1 = np.dot(X, params["W1"]) + params["b1"]
    A1 = np.maximum(0.0, Z1)

    # Affine Layer 2 & ReLU Activation
    Z2 = np.dot(A1, params["W2"]) + params["b2"]
    A2 = np.maximum(0.0, Z2)

    # Affine Layer 3 (Output Logits)
    Z3 = np.dot(A2, params["W3"]) + params["b3"]

    # Numerically Stable Softmax: subtract row-wise max to prevent exp overflow
    shifted_logits = Z3 - np.max(Z3, axis=-1, keepdims=True)
    unnormalized_exp = np.exp(shifted_logits)
    sum_exp = np.sum(unnormalized_exp, axis=-1, keepdims=True)
    probabilities = unnormalized_exp / sum_exp

    # Check softmax integrity: each row must sum to 1 within 1e-6
    row_totals = np.sum(probabilities, axis=1)
    assert np.allclose(row_totals, 1.0, atol=1e-6), "Softmax row sum violation (> 1e-6)"

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "probabilities": probabilities
    }
    return probabilities, cache


def cross_entropy_objective(
    probabilities: np.ndarray,
    Y_targets: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Computes mean categorical cross-entropy loss:
      L = - (1 / N) * sum_i sum_k Y_{ik} * log(P_{ik} + eps)
    """
    batch_size = probabilities.shape[0]
    clipped_probs = np.clip(probabilities, eps, 1.0 - eps)
    total_loss = -np.sum(Y_targets * np.log(clipped_probs))
    return float(total_loss / batch_size)


def backward_pass(
    params: Dict[str, np.ndarray],
    cache: Dict[str, np.ndarray],
    Y_targets: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculates exact analytical gradients via matrix calculus chain rule:
      dZ3 = (P - Y) / N
      dW3 = A2.T @ dZ3, db3 = sum_rows(dZ3)
      dA2 = dZ3 @ W3.T
      dZ2 = dA2 * (Z2 > 0)
      dW2 = A1.T @ dZ2, db2 = sum_rows(dZ2)
      dA1 = dZ2 @ W2.T
      dZ1 = dA1 * (Z1 > 0)
      dW1 = X.T @ dZ1, db1 = sum_rows(dZ1)
    """
    N = Y_targets.shape[0]
    P = cache["probabilities"]
    A2 = cache["A2"]
    A1 = cache["A1"]
    X = cache["X"]
    Z2 = cache["Z2"]
    Z1 = cache["Z1"]

    # Gradient of loss w.r.t. pre-softmax logits Z3
    dZ3 = (P - Y_targets) / N  # shape: (N, 10)

    # Layer 3 parameter gradients
    dW3 = np.dot(A2.T, dZ3)    # shape: (W, 10)
    db3 = np.sum(dZ3, axis=0)  # shape: (10,)

    # Backpropagate into Hidden Layer 2
    dA2 = np.dot(dZ3, params["W3"].T) # shape: (N, W)
    dZ2 = dA2 * (Z2 > 0.0)             # ReLU derivative mask

    # Layer 2 parameter gradients
    dW2 = np.dot(A1.T, dZ2)    # shape: (W, W)
    db2 = np.sum(dZ2, axis=0)  # shape: (W,)

    # Backpropagate into Hidden Layer 1
    dA1 = np.dot(dZ2, params["W2"].T) # shape: (N, W)
    dZ1 = dA1 * (Z1 > 0.0)             # ReLU derivative mask

    # Layer 1 parameter gradients
    dW1 = np.dot(X.T, dZ1)     # shape: (64, W)
    db1 = np.sum(dZ1, axis=0)  # shape: (W,)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3
    }
    return gradients


def apply_gradients(
    params: Dict[str, np.ndarray],
    gradients: Dict[str, np.ndarray],
    lr: float
) -> Dict[str, np.ndarray]:
    """Applies functional stochastic gradient descent update step."""
    updated = {}
    for key in params:
        grad_key = f"d{key}"
        updated[key] = params[key] - lr * gradients[grad_key]
    return updated


def predict_classes(X: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """Computes discrete categorical predictions for input samples."""
    probs, _ = forward_pass(X, params)
    return np.argmax(probs, axis=1)


def calculate_accuracy(y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> float:
    """Computes categorical classification accuracy ratio."""
    return float(np.mean(y_true_labels == y_pred_labels))
