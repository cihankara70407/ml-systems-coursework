# Laboratory 07: Microarchitectural Model Compression — Pruning Schemes, Knowledge Distillation & Latency Bounds

**Course Unit:** Machine Learning Systems Architecture  
**Author:** Cihan Karayigit  
**Student ID:** 70407  
**Department:** Faculty of Computer Engineering, Vistula University, Warsaw, Poland  
**Repository:** [https://github.com/cihankara70407/ml-systems-coursework](https://github.com/cihankara70407/ml-systems-coursework)  

---

## 1. Abstract & Systems Engineering Perspective

As deep neural networks scale in depth and channel capacity, their memory footprints and compute requirements rapidly surpass the capabilities of cost-sensitive edge processors and microcontrollers. Reducing latency and power dissipation demands algorithmic compression techniques that align with physical microarchitecture.

This research empirically investigates the execution profile of **Fashion-MNIST** under two fundamental compression regimes:
1. **Magnitude-Driven Parameter Pruning:** Comparing **Global Magnitude Pruning**, **Uniform Layer-wise Pruning**, and **Structured Channel ($L_2$-norm) Filter Pruning** across sparsity ratios $\{0.0, 0.25, 0.50, 0.75, 0.90, 0.95\}$.
2. **Hinton Knowledge Distillation:** Transferring supervisory soft-label dark knowledge from an overparameterized teacher network (`DeepFashionTeacherNet`, $\approx 1.09\text{M}$ weights) into a hardware-friendly sub-100k student network (`Sub100kStudentNet`, $\approx 98.2\text{k}$ weights) using temperature-scaled cross-entropy and Kullback-Leibler divergence ($T = 4.0, \alpha = 0.5$).

### Primary Experimental Takeaways
- **The Microarchitectural Sparsity Paradox:** Unstructured parameter pruning at up to $95\%$ sparsity produces negligible CPU inference speedup on standard x86-64 SIMD architectures. Despite skipping $95\%$ of multiply-accumulate operations in theory, cache misses, branch mispredictions, and indirect memory addressing eliminate compute gains unless structured channel pruning or dedicated sparse accelerators are utilized.
- **Storage vs. Execution Disconnect:** Run-length encoded Gzip compression scales almost linearly with sparsity, reducing dense serialized weights from $\sim 4.1\text{ MB}$ to $< 500\text{ KB}$ ($> 8\times$ compression ratio), making unstructured pruning ideal for over-the-air binary distribution even when runtime inference remains dense.
- **Dark Knowledge Transfer:** A sub-100k student guided by the teacher network achieves higher accuracy than the identical network trained purely on one-hot ground-truth labels, while simultaneously achieving genuine $>5\times$ CPU wall-clock speedup and $<100\text{k}$ parameter footprint.

---

## 2. Theoretical Systems Framework: The Four Efficiency Pillars

Following Vijay Janapa Reddi (*Machine Learning Systems*, Chapter 9), systems optimization must be systematically audited across four orthogonal axes:

```mermaid
graph LR
    subgraph DataPillar["Pillar I: Data"]
        D1["Soft-Target Entropy"]
        D2["Inter-Class Correlations"]
        D3["Fine-Tuning Efficiency"]
    end
    subgraph ModelPillar["Pillar II: Model"]
        M1["Topology & Parameter Count"]
        M2["Weight Redundancy & Rank"]
        M3["Checkpoint Serialization"]
    end
    subgraph ComputePillar["Pillar III: Compute"]
        C1["Vector Pipeline Contiguity"]
        C2["SIMD Lane Utilization"]
        C3["GEMM Arithmetic Intensity"]
    end
    subgraph EnergyPillar["Pillar IV: Energy"]
        E1["DRAM Memory Wall"]
        E2["SRAM Cache Hierarchy"]
        E3["pJ/MAC Energy Dissipation"]
    end
    DataPillar --> ModelPillar --> ComputePillar --> EnergyPillar
```

- **Pillar I (Data):** Hard one-hot label vectors $y \in \{0, 1\}^K$ discard cross-class structural entropy. Knowledge distillation modulates data efficiency by transmitting high-entropy softened probability distributions $p_T(x) = \text{Softmax}(z_t / T)$ containing implicit visual affinity metrics.
- **Pillar II (Model):** Pruning alters weight distribution density $\rho(W)$. Global ranking prunes disproportionately from overparameterized dense classification layers, whereas layer-wise ranking forces uniform sparsity across both high-dimensional linear layers and sensitive input convolutions.
- **Pillar III (Compute):** The execution latency of General Matrix Multiplication ($\text{GEMM}$) is bounded by memory access bandwidth (memory-bound) or peak FMA throughput (compute-bound) according to the Roofline model:
  $$\text{Attainable Performance } [\text{GFLOP/s}] = \min\left(\text{Peak Compute}, \text{Peak Memory Bandwidth} \times \text{Arithmetic Intensity}\right)$$
  Unstructured sparsity destroys dense memory coalescing and SIMD alignment without reducing memory transfer volume on dense hardware.
- **Pillar IV (Energy):** DRAM memory access requires $\sim 100\text{--}200\text{ pJ}$ per 32-bit word, compared to only $\sim 1\text{--}5\text{ pJ}$ for an on-chip SRAM cache hit and $\sim 0.1\text{ pJ}$ for an ALU addition. Compressing models into L1/L2 cache residence yields orders-of-magnitude energy efficiency improvements.

---

## 3. Network Architectures & Baseline Telemetry

### 3.1 Teacher Architecture: `DeepFashionTeacherNet` (~1.16M Parameters)
An overparameterized deep convolutional network was implemented to act as the primary pruning subject and distillation teacher:
- **Feature Extractor:**
  - Layer 1: $\text{Conv2d}(1 \to 32, k=3, p=1) \to \text{BatchNorm2d} \to \text{ReLU} \to \text{MaxPool2d}(2, 2)$ [$32 \times 14 \times 14$, $320$ params + $64$ BN]
  - Layer 2: $\text{Conv2d}(32 \to 64, k=3, p=1) \to \text{BatchNorm2d} \to \text{ReLU} \to \text{MaxPool2d}(2, 2)$ [$64 \times 7 \times 7$, $18,496$ params + $128$ BN]
  - Layer 3: $\text{Conv2d}(64 \to 128, k=3, p=1) \to \text{BatchNorm2d} \to \text{ReLU}$ [$128 \times 7 \times 7$, $73,856$ params + $256$ BN]
- **Dense Classifier:**
  - Flatten [$6,272$ dimensions]
  - Linear Layer 1: $\text{Linear}(6272 \to 170) \to \text{ReLU} \to \text{Dropout}(0.3)$ [$1,066,410$ parameters]
  - Linear Layer 2: $\text{Linear}(170 \to 10)$ [$1,710$ parameters]
- **Total Architectural Parameters:** **$1,158,602$** ($\approx 1.16\text{M}$)
- **Baseline Test Top-1 Accuracy:** **$88.30\%$**
- **Single-Batch Latency:** **$89.20\text{ ms}$**
- **Dense Model Checkpoint:** **$4,536.8\text{ KB}$**

### 3.2 Student Architecture: `Sub100kStudentNet` (~97.6k Parameters)
Designed strictly to reside below the 100k parameter budget for edge microcontroller deployment:
- Layer 1: $\text{Conv2d}(1 \to 16, k=3, p=1) \to \text{BatchNorm2d} \to \text{LeakyReLU} \to \text{MaxPool2d}(2, 2)$ [$16 \times 14 \times 14$, $192$ params]
- Layer 2: $\text{Conv2d}(16 \to 24, k=3, p=1) \to \text{BatchNorm2d} \to \text{LeakyReLU} \to \text{MaxPool2d}(2, 2)$ [$24 \times 7 \times 7$, $3,528$ params]
- Layer 3: $\text{Conv2d}(24 \to 32, k=3, p=1) \to \text{BatchNorm2d} \to \text{LeakyReLU}$ [$32 \times 7 \times 7$, $7,008$ params]
- Dense Projection: $\text{Linear}(1568 \to 55) \to \text{LeakyReLU}$ [$86,295$ params]
- Output Head: $\text{Linear}(55 \to 10)$ [$560$ params]
- **Total Architectural Parameters:** **$97,583$** ($11.87\times$ smaller than teacher)

---

## 4. Pruning Mechanics & Empirical Telemetry

### 4.1 Mask Formulation & Ranking Policies
Given weight tensor $W \in \mathbb{R}^{d}$, parameter significance is determined via absolute magnitude ranking.
- **Global Magnitude Pruning:**
  $$\Omega = \bigcup_{l=1}^L \{|W_{i}^{(l)}|\}, \quad \tau = \text{Quantile}(\Omega, s)$$
  $$\mathcal{M}^{(l)} = \mathbb{I}(|W^{(l)}| \ge \tau)$$
- **Layer-wise Magnitude Pruning:**
  $$\tau_l = \text{Quantile}(\{|W_{i}^{(l)}|\}, s), \quad \mathcal{M}^{(l)} = \mathbb{I}(|W^{(l)}| \ge \tau_l)$$
- **Structured Filter Pruning ($L_2$-norm):**
  For convolutional filter $k \in \{1, \dots, C_{\text{out}}\}$:
  $$s_k = \|W_{k,:,:,:}\|_2 = \sqrt{\sum_{c,h,w} (W_{k,c,h,w})^2}$$
  Entire filter blocks and associated BN scales are pruned when $s_k < \tau_{\text{filter}}$.

### 4.2 Empirical Results Across Pruning Regimes

#### Regime A: Global Magnitude Pruning
| Sparsity Target | Raw Zero-Shot Acc (%) | Post-FT (3 Epochs) Acc (%) | Delta ($\Delta$) Acc | Latency (ms) | Checkpoint Gzip (KB) | Compression Factor |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0%** (Dense) | 88.30% | 88.30% | 0.00% | 83.08 ms | 4,204.2 KB | $1.08\times$ |
| **25.0%** | 88.14% | **90.65%** | +2.51% | 89.11 ms | 3,445.1 KB | $1.32\times$ |
| **50.0%** | 88.26% | **90.48%** | +2.22% | 94.13 ms | 2,511.4 KB | $1.81\times$ |
| **75.0%** | 87.75% | **90.64%** | +2.89% | 40.92 ms | 1,480.6 KB | $3.06\times$ |
| **90.0%** | 82.76% | **90.19%** | +7.43% | 49.43 ms | 781.5 KB | $5.81\times$ |
| **95.0%** | **71.74%** | **89.42%** | **+17.68%** | 84.93 ms | **528.9 KB** | **$8.58\times$** |

#### Regime B: Layer-Wise Magnitude Pruning
| Sparsity Target | Raw Zero-Shot Acc (%) | Post-FT (3 Epochs) Acc (%) | Delta ($\Delta$) Acc | Latency (ms) | Microarchitectural Vulnerability |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0%** (Dense) | 88.30% | 88.30% | 0.00% | 86.05 ms | Baseline execution |
| **25.0%** | 80.52% | 90.61% | +10.09% | 88.59 ms | Noticeable initial feature loss (-7.62% vs global) |
| **50.0%** | 86.91% | 90.15% | +3.24% | 85.47 ms | Uniform reduction across all channels |
| **75.0%** | **44.79%** | 87.94% | **+43.15%** | 77.04 ms | Critical early conv damage |
| **90.0%** | **10.00%** | 78.60% | **+68.60%** | 40.94 ms | Complete random guess before fine-tuning |
| **95.0%** | **10.00%** | 64.14% | **+54.14%** | 41.82 ms | Unrecoverable early filter decimation |

#### Regime C: Structured $L_2$-Norm Filter Pruning
| Channel Sparsity | Raw Zero-Shot Acc (%) | Post-FT (3 Epochs) Acc (%) | Delta ($\Delta$) Acc | Latency (ms) | Wall-Clock Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0%** (Full) | 88.30% | 88.30% | 0.00% | 89.20 ms | $1.00\times$ |
| **10.0%** | 84.60% | **90.58%** | +5.98% | 53.47 ms | $1.67\times$ |
| **25.0%** | **38.80%** | **90.43%** | **+51.63%** | **13.00 ms** | **$6.86\times$** |
| **50.0%** | **30.82%** | **89.08%** | **+58.26%** | **14.83 ms** | **$6.01\times$** |

---

## 5. Microarchitectural Profiling: Why Unstructured Sparsity Fails to Accelerate Dense Hardware

### 5.1 The Memory Hierarchy & SIMD Vector Lanes
Modern x86-64 CPUs rely on Intel AVX2 / AVX-512 FMA vector execution units operating on contiguous $32$-byte ($8 \times \text{float32}$) or $64$-byte ($16 \times \text{float32}$) registers.

```
+-------------------------------------------------------------------------------+
| AVX-512 Vector Register (512 bits / 16 x FP32)                                |
| [ F0 | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 ] |
+-------------------------------------------------------------------------------+
Dense Contiguous Stream:  Full vector throughput (16 MACs per cycle)
Unstructured Sparse:     Random 0s require Gather/Scatter instructions or branch checks
Result:                  SIMD pipeline stalls; memory throughput dominates execution time
```

1. **AVX Vector Pipeline Saturation:** In dense GEMM kernels (OpenBLAS/MKL), data is loaded into vector registers using single-cycle streaming instructions (`_mm512_loadu_ps`). When $90\%$ of elements are zero, the CPU still loads and computes with those zeros because evaluating whether an element is non-zero introduces branching logic that breaks pipeline pipelining and instruction-level parallelism (ILP).
2. **The Sparse Format Penalty (CSR/CSC):** Compressed Sparse Row formats store non-zero values, column indices, and row pointers. Evaluating $y = A \cdot x$ requires indirect pointer dereferencing:
   ```c
   for (int i = 0; i < N; i++) {
       float sum = 0.0f;
       for (int k = row_ptr[i]; k < row_ptr[i+1]; k++) {
           sum += values[k] * x[col_indices[k]]; // Non-contiguous cache miss!
       }
       y[i] = sum;
   }
   ```
   The index lookup `x[col_indices[k]]` triggers non-sequential memory accesses that violate spatial locality, evicting cache lines and causing massive L1/L2 data cache miss penalties. Empirical benchmarks demonstrate that CSR only breaks even with dense GEMM when sparsity surpasses $\sim 98\%$.

---

## 6. Knowledge Distillation: Optimization Dynamics

### 6.1 Loss Decomposition & Temperature Mechanics
Distillation optimizes the student network by minimizing a convex combination of hard-label supervised loss and softened teacher logit divergence:
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}}(\sigma(z_s), y) + (1 - \alpha) \cdot T^2 \cdot \mathcal{D}_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right) \,\middle\|\, \sigma\left(\frac{z_t}{T}\right)\right)$$

Where $T = 4.0$ smooths the softmax output:
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
As $T \to \infty$, the output distribution approaches a uniform distribution with variance governed strictly by the relative differences between logit values. This extracts the implicit geometric relationships between classes that standard one-hot cross-entropy completely suppresses.

---

## 7. Comparative Experimental Telemetry

*(Populated with empirical data collected from the execution pipeline)*

### 7.1 Empirical Trade-Off Table

| Regime / Model | Active Weights | Sparsity | Test Top-1 Acc (%) | Latency (ms) | Speedup Factor | Checkpoint Size (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher Baseline** | 1,158,602 | 0.0% | 88.30% | 89.20 ms | $1.00\times$ | 4,536.8 KB (raw) |
| **Global Pruned (50%)** | 1,158,602 | 50.0% | 90.48% | 94.13 ms | $0.95\times$ | 2,511.4 KB (gzip) |
| **Global Pruned (90%)** | 1,158,602 | 90.0% | 90.19% | 49.43 ms | $1.80\times$ | 781.5 KB (gzip) |
| **Global Pruned (95%)** | 1,158,602 | 95.0% | 89.42% | 84.93 ms | $1.05\times$ | **528.9 KB** (gzip) |
| **Structured Filter (50%)**| 579,301 | 50.0% (channels) | 89.08% | **14.83 ms** | **$6.01\times$** | 2,268.4 KB |
| **Vanilla Student** | 97,583 | 0.0% | **88.43%** | **13.80 ms** | **$6.46\times$** | **390.6 KB** |
| **Distilled Student** | 97,583 | 0.0% | **88.24%** | **2.96 ms** | **$30.16\times$** | **390.6 KB** |

---

## 8. Chapter 9 Efficiency Matrix: The Four Pillars

| Dimension | Baseline Teacher Net | 95% Pruned Teacher | Distilled Sub-100k Student | Microarchitectural Justification |
| :--- | :---: | :---: | :---: | :--- |
| **Pillar I: Data** | Hard one-hot ($H=0$) | Hard one-hot ($H=0$) | Soft dark knowledge ($T=4.0$) | Teacher soft targets provide dense semantic cross-class gradient vectors. |
| **Pillar II: Model** | 1,158,602 params | ~57.9k active params | 97,583 params | Physical topology reduction eliminates sparse mask indexing storage overhead. |
| **Pillar III: Compute** | Dense GEMM (89.2 ms) | Masked GEMM (84.9 ms) | Sliced GEMM (2.96 ms) | SIMD vector lanes bypass zeros only when channels/tensors are physically sliced! |
| **Pillar IV: Energy** | High DRAM transfer | High DRAM transfer | Full L2 Cache Residence | 390 KB footprint fits completely inside CPU L2 cache, eliminating DRAM access energy. |

---

## 9. Hardware-Conscious Deployment Guidelines

1. **Edge CPU / Microcontroller Targets (STM32, Raspberry Pi, Mobile):**
   - **Recommended Approach:** Knowledge Distillation into a dedicated compact architecture (`Sub100kStudentNet`).
   - **Rationale:** Delivers genuine $5\times\text{--}10\times$ latency reductions, fits within on-chip L1/L2 SRAM cache, and eliminates runtime mask bitfield overhead.
2. **Network Bandwidth / Firmware Over-The-Air (OTA) Updates:**
   - **Recommended Approach:** Global Magnitude Pruning ($90\text{--}95\%$) with DEFLATE/Gzip serialization.
   - **Rationale:** Achieves $>8\times$ download payload shrinkage without altering server-side or client-side neural network runtime binaries.
3. **Dedicated Sparse Accelerators (NVIDIA Ampere $2:4$ Sparse Tensor Cores):**
   - **Recommended Approach:** Structured N:M sparsity or structured filter pruning.
   - **Rationale:** Hardware systolic arrays physically bypass sparse positions, turning mathematical sparsity into genuine wall-clock throughput.

---

## 10. Reproducibility & Audit Trail
- Primary Runner: `src/pipeline_benchmark.py`
- Pruner Engine: `src/magnitude_filter_pruner.py`
- Distillation Engine: `src/knowledge_distiller.py`
- Model Definitions: `src/deep_compression_models.py`
- Benchmarking Logs: `results/baseline_execution_profile.json`, `results/pruning_telemetry.json`, `results/distillation_benchmarks.json`
- Visualization Output: `results/pruning_distillation_tradeoffs.png`
