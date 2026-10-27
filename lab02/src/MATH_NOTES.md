# Mathematical Foundations: Matrix Calculus & Backpropagation

**Author:** Cihan Karayigit (Student ID: 70407)  
**Topic:** Exact Derivation of Analytical Gradients for a Two-Hidden-Layer Multilayer Perceptron  

---

## 1. Network Topology & Forward Mapping

Let $X \in \mathbb{R}^{N \times D}$ denote the mini-batch input matrix ($N = 32, D = 64$). The network defines a sequence of alternating affine maps and nonlinear point-wise activations:

$$Z_1 = X W_1 + \mathbf{1}_N b_1^T, \quad W_1 \in \mathbb{R}^{D \times W}, \quad b_1 \in \mathbb{R}^W$$
$$A_1 = \sigma(Z_1) = \max(0, Z_1)$$
$$Z_2 = A_1 W_2 + \mathbf{1}_N b_2^T, \quad W_2 \in \mathbb{R}^{W \times W}, \quad b_2 \in \mathbb{R}^W$$
$$A_2 = \sigma(Z_2) = \max(0, Z_2)$$
$$Z_3 = A_2 W_3 + \mathbf{1}_N b_3^T, \quad W_3 \in \mathbb{R}^{W \times C}, \quad b_3 \in \mathbb{R}^C$$

where $C = 10$ denotes the number of target classes, and $\mathbf{1}_N \in \mathbb{R}^N$ is a column vector of ones.

The predictive probability distribution is obtained via the row-wise softmax transformation:
$$P_{ik} = \frac{\exp(Z_{3, ik})}{\sum_{j=1}^C \exp(Z_{3, ij})}, \quad \forall i \in \{1, \dots, N\}, \, k \in \{1, \dots, C\}$$

The scalar objective is the mean categorical cross-entropy loss:
$$\mathcal{L}(P, Y) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^C Y_{ik} \ln(P_{ik})$$

---

## 2. Gradient of Objective w.r.t. Pre-Softmax Logits $Z_3$

We employ the matrix differential approach. For a single observation $i$:
$$d\mathcal{L}_i = -\sum_{k=1}^C \frac{Y_{ik}}{P_{ik}} dP_{ik}$$

The differential of the softmax probability $P_{ik}$ with respect to $Z_{3, ij}$ satisfies:
$$\frac{\partial P_{ik}}{\partial Z_{3, ij}} = P_{ik} (\delta_{kj} - P_{ij})$$

Substituting this relation into the chain rule:
$$\frac{\partial \mathcal{L}_i}{\partial Z_{3, ij}} = -\sum_{k=1}^C \frac{Y_{ik}}{P_{ik}} P_{ik} (\delta_{kj} - P_{ij}) = -\sum_{k=1}^C Y_{ik} (\delta_{kj} - P_{ij}) = -Y_{ij} + P_{ij} \sum_{k=1}^C Y_{ik}$$

Since the one-hot target satisfies the probability axiom $\sum_{k=1}^C Y_{ik} = 1$:
$$\frac{\partial \mathcal{L}_i}{\partial Z_{3, ij}} = P_{ij} - Y_{ij}$$

Summing over the full batch and dividing by $N$:
$$\frac{\partial \mathcal{L}}{\partial Z_3} = \frac{1}{N} (P - Y) \triangleq \delta_3 \in \mathbb{R}^{N \times C}$$

---

## 3. Backpropagation into Layer 3 Parameters $(W_3, b_3)$

Using the differential form of the affine transformation $dZ_3 = (dA_2) W_3 + A_2 (dW_3) + \mathbf{1}_N (db_3)^T$:
$$d\mathcal{L} = \operatorname{tr}\left( \delta_3^T dZ_3 \right) = \operatorname{tr}\left( \delta_3^T A_2 dW_3 \right) + \operatorname{tr}\left( \delta_3^T \mathbf{1}_N db_3^T \right) + \operatorname{tr}\left( \delta_3^T dA_2 W_3 \right)$$

Applying cyclic permutation of the matrix trace:
1. **Weight Gradient:**
   $$\operatorname{tr}\left( (A_2^T \delta_3)^T dW_3 \right) \implies \frac{\partial \mathcal{L}}{\partial W_3} = A_2^T \delta_3 \in \mathbb{R}^{W \times C}$$
2. **Bias Gradient:**
   $$\operatorname{tr}\left( (\delta_3^T \mathbf{1}_N)^T db_3 \right) \implies \frac{\partial \mathcal{L}}{\partial b_3} = \delta_3^T \mathbf{1}_N = \sum_{i=1}^N \delta_{3, i, :} \in \mathbb{R}^C$$
3. **Activation Gradient:**
   $$\operatorname{tr}\left( (W_3 \delta_3^T)^T dA_2^T \right) = \operatorname{tr}\left( (\delta_3 W_3^T)^T dA_2 \right) \implies \frac{\partial \mathcal{L}}{\partial A_2} = \delta_3 W_3^T \in \mathbb{R}^{N \times W}$$

---

## 4. Backpropagation Through Nonlinear Activation & Hidden Layer 2

For the ReLU activation function $A_2 = \max(0, Z_2)$, the subgradient operator acts element-wise:
$$\frac{\partial A_{2, ij}}{\partial Z_{2, ij}} = \mathbb{I}(Z_{2, ij} > 0)$$

Using the Hadamard product ($\odot$):
$$\delta_2 \triangleq \frac{\partial \mathcal{L}}{\partial Z_2} = \frac{\partial \mathcal{L}}{\partial A_2} \odot \mathbb{I}(Z_2 > 0) = (\delta_3 W_3^T) \odot \mathbb{I}(Z_2 > 0) \in \mathbb{R}^{N \times W}$$

Repeating the affine differential identity for Layer 2:
$$\frac{\partial \mathcal{L}}{\partial W_2} = A_1^T \delta_2 \in \mathbb{R}^{W \times W}$$
$$\frac{\partial \mathcal{L}}{\partial b_2} = \sum_{i=1}^N \delta_{2, i, :} \in \mathbb{R}^W$$
$$\frac{\partial \mathcal{L}}{\partial A_1} = \delta_2 W_2^T \in \mathbb{R}^{N \times W}$$

---

## 5. Backpropagation into Hidden Layer 1

Propagating through the first ReLU activation:
$$\delta_1 \triangleq \frac{\partial \mathcal{L}}{\partial Z_1} = \frac{\partial \mathcal{L}}{\partial A_1} \odot \mathbb{I}(Z_1 > 0) = (\delta_2 W_2^T) \odot \mathbb{I}(Z_1 > 0) \in \mathbb{R}^{N \times W}$$

Finally, computing Layer 1 parameter gradients:
$$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \delta_1 \in \mathbb{R}^{D \times W}$$
$$\frac{\partial \mathcal{L}}{\partial b_1} = \sum_{i=1}^N \delta_{1, i, :} \in \mathbb{R}^W$$

---

## 6. Matrix Multiplication (GEMM) Counting

Counting dense matrix multiplications highlights the fundamental runtime discrepancy between inference and backpropagation:

| Phase | Operation | Dimension 1 | Dimension 2 | Product Dimension | GEMM Count |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Forward** | $Z_1 = X W_1$ | $(N \times 64)$ | $(64 \times W)$ | $(N \times W)$ | 1 |
| **Forward** | $Z_2 = A_1 W_2$ | $(N \times W)$ | $(W \times W)$ | $(N \times W)$ | 1 |
| **Forward** | $Z_3 = A_2 W_3$ | $(N \times W)$ | $(W \times 10)$ | $(N \times 10)$ | 1 |
| **Forward Total** | — | — | — | — | **3 GEMMs** |
| **Backward** | $dW_3 = A_2^T \delta_3$ | $(W \times N)$ | $(N \times 10)$ | $(W \times 10)$ | 1 |
| **Backward** | $dA_2 = \delta_3 W_3^T$ | $(N \times 10)$ | $(10 \times W)$ | $(N \times W)$ | 1 |
| **Backward** | $dW_2 = A_1^T \delta_2$ | $(W \times N)$ | $(N \times W)$ | $(W \times W)$ | 1 |
| **Backward** | $dA_1 = \delta_2 W_2^T$ | $(N \times W)$ | $(W \times W)$ | $(N \times W)$ | 1 |
| **Backward** | $dW_1 = X^T \delta_1$ | $(64 \times N)$ | $(N \times W)$ | $(64 \times W)$ | 1 |
| **Backward Total**| — | — | — | — | **5 GEMMs** |

**Theoretical Conclusion:** The backward pass performs $5$ GEMMs compared to $3$ in the forward pass, naturally demanding more than $1.67\times$ the matrix multiplication floating-point operations.
