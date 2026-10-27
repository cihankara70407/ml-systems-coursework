# Lab 03: Comparative DNN Architectures and Their Costs

**Author:** Cihan Karayigit (Student ID: 70407)  
**Assigned Readings:** Reddi, Chapter 4 DNN Architectures: §§4.2, 4.3, 4.5, 4.6, 4.7  

---

## 1. Goal
This laboratory investigates the trade-off surface between predictive accuracy and systems execution cost across three distinct neural network paradigms (Multi-Layer Perceptron, Convolutional Neural Network, and Patch-based Vision Transformer) on the Fashion-MNIST dataset. The module hand-calculates theoretical FLOPs, measures profiler estimates (`thop.profile`), evaluates inference latencies across batch sizes 1 and 128, trains each architecture for 5 epochs, and analyzes memory access patterns to determine architectural efficiency.

---

## 2. Method
1. **Dataset Pipeline:** Fashion-MNIST dataset preprocessed with `ToTensor()` and normalization (mean = 0.2860, std = 0.3530). Official 60,000 training images partitioned into 55,000 train and 5,000 validation sets (`seed=101`). Evaluated on official 10,000 test set.
2. **Architecture Configurations (Target ~100k-150k params):**
   - **MLP:** Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10 (235,146 parameters).
   - **CNN:** Conv2d(1, 16, 3, pad=1) -> ReLU -> MaxPool2d(2) -> Conv2d(16, 32, 3, pad=1) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(32*7*7, 10) (20,490 parameters).
   - **Vision Transformer:** 16 patches (7x7) -> Linear 64 -> 2 Transformer encoder layers (d_model=64, nhead=4, dim_feedforward=128) -> Mean pool -> Linear 64->10 (70,794 parameters).
3. **FLOPs Derivation vs. Profiling:**
   - Hand-calculated theoretical FLOPs using Chapter 4 formulas (2 * in * out for linear, 2 * K * K * C_in * C_out * H_out * W_out for conv, 2 * seq * d_model * 3 * d_model for attention QKV).
   - Evaluated thop.profile MACs and bridged the 2x multiplier gap (1 MAC = 2 FLOPs).
4. **Latency Benchmarks:**
   - Single-sample inference (batch 1): Median of 100 runs.
   - Batched inference (batch 128): Median of 20 runs.
5. **Training Protocol:** Trained all models for 5 epochs using Adam (learning rate 1e-3, batch size 128).
6. **Trade-off Visualizations:** Plotted accuracy vs parameters and accuracy vs latency to `results/tradeoff_plots.png`.

---

## 3. Results Table or Plot

### Multi-Architecture Systems Cost and Accuracy Matrix

| Model Class | Architecture Category | Parameter Count | Hand Layer FLOPs | `thop` MACs (b=1) | Median Latency (b=1) | Median Latency (b=128) | 5-Epoch Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | Fully Connected | **235,146** | 401,408 (Layer 1) | 234,752 | **0.133 ms** | **0.72 ms** | **87.47%** |
| **CNN** | Convolutional | **20,490** | 225,792 (Conv 1) | 1,031,744 | **0.240 ms** | **6.01 ms** | **89.68%** |
| **Attention (ViT)** | Transformer | **70,794** | 393,216 (QKV Proj) | 591,488 | **1.263 ms** | **4.57 ms** | **83.79%** |

### Pareto Frontier Visualizations
![Tradeoff Plots](results/tradeoff_plots.png)

### Primary Handwritten Mathematical Derivations & Diagrams
Below are the primary handwritten derivations for parameter totals and FLOPs across the model architectures:

![Handwritten FLOPs and Parameter Derivations](results/handwritten_calculations.jpg)

#### Transcribed Mathematical Step-by-Step Breakdown

1. **Multi-Layer Perceptron (Input $28 \times 28 = 784$):**
   - **Layer 1 (Linear: $784 \rightarrow 128$):**
     - Weights: $784 \times 128 = 100,352$, Biases: $128$. (Subtotal: $100,480$)
     - FLOPs: $2 \times 784 \times 128 = 200,704 \text{ FLOPs}$
   - **Layer 2 (Linear: $128 \rightarrow 64$):**
     - Weights: $128 \times 64 = 8,192$, Biases: $64$. (Subtotal: $8,256$)
     - FLOPs: $2 \times 128 \times 64 = 16,384 \text{ FLOPs}$
   - **Layer 3 (Output: $64 \rightarrow 10$):**
     - Weights: $64 \times 10 = 640$, Biases: $10$. (Subtotal: $650$)
     - FLOPs: $2 \times 64 \times 10 = 1,280 \text{ FLOPs}$
   - **Total Parameters:** $100,480 + 8,256 + 650 = 109,386 \text{ parameters}$
   - **Total Network Forward FLOPs:** $200,704 + 16,384 + 1,280 = 218,368 \text{ FLOPs}$

2. **Convolutional Neural Network (Conv2D MNIST-like):**
   - **Input Specification:** $28 \times 28 \times 1$
   - **Conv Configuration:** 32 filters, $3 \times 3$ kernel, padding = 1
   - **Output Dimensions:** $28 \times 28 \times 32$
   - **Parameter Count:** $(3 \times 3 \times 1 \times 32) + 32 = 320 \text{ parameters}$
   - **FLOPs:** $2 \times 28 \times 28 \times 1 \times 32 \times 3 \times 3 = 451,584 \text{ FLOPs}$

3. **Architectural Schematics & Learning Curves:**
   - Topology graph illustrating dense interconnected feedforward layers ($784 \rightarrow 128 \rightarrow 64 \rightarrow 10$).
   - Empirical loss trajectory analysis highlighting non-linear activation stability.

### Explaining Profiler Gaps (FLOPs vs. MACs)
The mathematical difference between hand derivations and `thop.profile` is explained by:
1. **Multiply-Accumulate Convention:** `thop` measures MACs. Each MAC comprises an addition and multiplication, equaling 2 FLOPs ($2 \times \text{MACs} = \text{FLOPs}$).
2. **Omission of Auxiliary Layers:** Non-linear activation operators (ReLU) and pooling mechanisms (MaxPool2d) are excluded from `thop` GEMM counting, creating minor offsets from raw operation totals.

---

## 4. Three Conclusions in Your Own Words
1. **Weight Sharing Dominates Spatial Vision Classification:** The Convolutional Network demonstrates unmatched parameter efficiency, capturing spatial hierarchies to achieve 89.68% accuracy with only 20,490 parameters—substantially outperforming the 235,146-parameter MLP (87.47% accuracy).
2. **Self-Attention Demands Higher Computational Bandwidth:** The Vision Transformer achieved 83.79% accuracy, but required higher per-sample inference latency (1.263 ms) due to sequence patch unfolding, projection layers, and multi-head attention reshaping.
3. **Hardware GEMM Alignment Accelerates Dense Topologies:** At batch size 128, the MLP achieves rapid throughput (0.72 ms) because large 2D matrix multiplications map directly to optimized BLAS linear algebra primitives with high arithmetic intensity.
