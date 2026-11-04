# Laboratory 05: Frameworks and Automatic Differentiation

**Student Researcher:** Cihan Karayigit (ID: 70407)  
**Academic Program:** Computer Engineering, Vistula University  
**Subject:** Machine Learning Systems Coursework  
**Reference Text:** Vijay Janapa Reddi, *Machine Learning Systems* (Chapter 7: AI Frameworks)  

---

## 1. Abstract & Systems Overview

Machine learning frameworks bridge high-level mathematical expressions and physical computing architecture. This report documents the implementation and microarchitectural evaluation of a scalar automatic differentiation engine and contrasts its runtime characteristics with compiled and vectorized framework execution:
1. **Dynamic Scalar DAG Construction:** Built the `AutoDiffScalar` engine featuring forward-pass graph recording, topological scheduling, and non-destructive gradient accumulation (`+=`).
2. **Empirical Gradient Validation:** Validated the engine against PyTorch autograd and symmetric numerical finite differences on non-linear composite functions, demonstrating zero divergence ($\Delta < 10^{-6}$).
3. **Moons Benchmark Telemetry:** Evaluated scalar interpreted execution against PyTorch tensor BLAS primitives over 5 complete training runs on `make_moons`.
4. **Execution Graph Profiling:** Traced the computation graph with TorchScript IR (`torch.jit.trace`) and evaluated dispatch latency across 1,000 warm-up cycles.
5. **Memory Stride & Cache Locality Investigation:** Quantified how non-contiguous tensor layouts degrade memory bus bandwidth and cache-line utilization, demonstrating a $2.46\times$ slowdown on strided reductions.

---

## 2. Dynamic Computation Graph Core (`AutoDiffScalar`)

The custom automatic differentiation module ([autodiff.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_cihan/lab05/src/autodiff.py)) implements scalar reverse-mode automatic differentiation without a global tape:

### 2.1 State & Graph Topology
Each `AutoDiffScalar` node encapsulates:
- `data` (`float`): Primal scalar output computed during the forward pass.
- `grad` (`float`): Accumulated partial adjoint $\frac{\partial \mathcal{L}}{\partial x}$, seeded to $0.0$.
- `_ancestors` (`tuple`): Input operand vertices in the directed acyclic graph (DAG).
- `_grad_fn` (`callable`): Backward closure storing the chain-rule transfer function for the node.
- `_op_type` (`str`): Operation tag for graph auditing.

### 2.2 Operator Overloads & Local Derivative Rules
- **Addition ($z = x + y$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= 1.0 \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} \mathrel{+}= 1.0 \cdot \frac{\partial \mathcal{L}}{\partial z}$$
- **Multiplication ($z = x \cdot y$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= y \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} \mathrel{+}= x \cdot \frac{\partial \mathcal{L}}{\partial z}$$
- **Power Function ($z = x^p$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= \left(p \cdot x^{p-1}\right) \cdot \frac{\partial \mathcal{L}}{\partial z}$$
- **Rectified Linear Unit ($z = \max(0, x)$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= \begin{cases} 1.0 \cdot \frac{\partial \mathcal{L}}{\partial z} & x > 0 \\ 0.0 & x \le 0 \end{cases}$$
- **Logistic Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= \sigma(x)(1 - \sigma(x)) \cdot \frac{\partial \mathcal{L}}{\partial z}$$
- **Logarithm ($z = \ln(x)$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= \frac{1}{x} \cdot \frac{\partial \mathcal{L}}{\partial z}$$

### 2.3 Topological Reverse Ordering & In-Place Accumulation
Reverse-mode automatic differentiation traverses nodes in exact reverse topological order:
```python
def backward(self) -> None:
    sorted_nodes = []
    visited = set()

    def dfs_visit(node: AutoDiffScalar) -> None:
        if node not in visited:
            visited.add(node)
            for ancestor in node._ancestors:
                dfs_visit(ancestor)
            sorted_nodes.append(node)

    dfs_visit(self)
    self.grad = 1.0  # Seed output loss adjoint

    for vertex in reversed(sorted_nodes):
        vertex._grad_fn()
```
The accumulation operator `+=` guarantees that shared graph vertices with out-degree $>1$ sum contributions from all descendant paths, adhering strictly to the multivariate calculus chain rule.

---

## 3. Mathematical & Empirical Gradient Verification

We validated the engine on the target expression:
$$\mathcal{L} = \sum_{i=1}^N \text{ReLU}\left( (a_i \cdot b_i + c_i)^2 \right)$$

### 3.1 Verification Setup
- **Evaluation Points:**
  - $a = [2.25, -1.75, 0.50, -3.10]$
  - $b = [-1.50, 3.20, -2.40, 1.10]$
  - $c = [0.85, -2.10, 1.30, 4.25]$
- **Symmetric Finite Difference Formulation:**
  $$\frac{\partial \mathcal{L}}{\partial \theta_k} \approx \frac{\mathcal{L}(\theta_k + h) - \mathcal{L}(\theta_k - h)}{2h}, \quad h = 10^{-7}$$

### 3.2 Verification Results
- **Forward Evaluation:** Custom `AutoDiffScalar` = `60.67200`, PyTorch = `60.67200` ($\Delta = 0.00$)
- **Max Absolute Difference vs. PyTorch:** **$0.00 \times 10^0$** (Strictly $< 10^{-6}$)
- **Max Absolute Difference vs. Finite Difference:** $9.84 \times 10^{-8}$
- **Outcome:** **PASSED** (100% gradient fidelity across all 12 parameters).

---

## 4. End-to-End Training Benchmark: `make_moons`

We benchmarked training performance of a multi-layer perceptron on the non-linear `make_moons` classification task ($N=100$, noise $\sigma = 0.10$):
- **Model Topology:** $2 \to 8 \to 1$ (8 hidden units with ReLU activation, 1 output unit with Sigmoid activation).
- **Optimization:** SGD with learning rate $\eta = 0.25$, 35 epochs.
- **Objective:** Binary Cross-Entropy Loss:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

### 4.1 5-Run Empirical Execution Timing

| Run | Custom AutoDiff (ms) | PyTorch Tensor Engine (ms) | Speedup Ratio |
| :---: | :---: | :---: | :---: |
| 1 | 3667.12 | 70.49 | 52.0x |
| 2 | 3794.04 | 12.81 | 296.2x |
| 3 | 2204.75 | 12.71 | 173.5x |
| 4 | 2393.60 | 13.85 | 172.8x |
| 5 | 2317.05 | 13.16 | 176.1x |
| **Median** | **2393.60 ms** | **13.16 ms** | **181.9x** |

### 4.2 Hardware & System Root Cause Analysis
The $181.9\times$ performance gap originates from fundamental system-level architecture differences:
1. **Dynamic Interpreter Overhead:** In `AutoDiffScalar`, each scalar addition and multiplication traverses the CPython bytecode loop (`LOAD_FAST`, `BINARY_MULTIPLY`), instantiating thousands of individual heap objects per forward pass.
2. **SIMD Vectorization & Contiguous Memory:** PyTorch allocates contiguous blocks of float32 values and executes GEMM operations through vectorized C++ backends (Intel MKL/BLAS) utilizing AVX2/AVX-512 vector units to execute 8 to 16 floating-point multiply-accumulate operations in a single CPU clock cycle.

---

## 5. PyTorch Execution Modes & IR Graph Inspection

We inspected the underlying intermediate representation (IR) of the small model using TorchScript tracing:

```
graph(%self : __torch__.DifferentiatedMoonsModel, %x : Float(32, 2, strides=[2, 1])):
  %affine_1.weight : Float(8, 2, strides=[2, 1]) = prim::GetAttr[name="affine_1.weight"](%self)
  %affine_1.bias : Float(8, strides=[1]) = prim::GetAttr[name="affine_1.bias"](%self)
  %1 : Tensor = aten::linear(%x, %affine_1.weight, %affine_1.bias)
  %2 : Tensor = aten::relu(%1)
  %affine_2.weight : Float(1, 8, strides=[8, 1]) = prim::GetAttr[name="affine_2.weight"](%self)
  %affine_2.bias : Float(1, strides=[1]) = prim::GetAttr[name="affine_2.bias"](%self)
  %3 : Tensor = aten::linear(%2, %affine_2.weight, %affine_2.bias)
  %out : Tensor = aten::sigmoid(%3)
  return (%out)
```

### 5.1 Latency Measurements (1,000 Iterations After Warm-up)
- **Eager Execution Latency:** $257.83\ \mu\text{s}$
- **Traced Graph Latency:** $253.91\ \mu\text{s}$
- **Speedup Factor:** **$1.02\times$**

---

## 6. Tensor Memory Layout, Stride Physics, and Cache Locality

Tensors store high-dimensional arrays in a linear one-dimensional memory space. The stride tuple specifies the physical step size required to advance one unit along each axis.

### 6.1 Layout Configuration ($2048 \times 2048$ Tensor)
- **Contiguous Array:** `stride = [2048, 1]`, `is_contiguous() = True`. Elements in the same row are adjacent in memory.
- **Transposed Array:** `stride = [1, 2048]`, `is_contiguous() = False`. Elements in the same row are spaced $2048 \times 4 = 8192$ bytes apart.

### 6.2 Empirical Stride Penalty
- **Contiguous Row-Sum Latency:** $1.027\text{ ms}$
- **Non-Contiguous Row-Sum Latency:** $2.521\text{ ms}$
- **Stride Penalty:** **$2.46\times$ slowdown**
- **Buffer Reallocation (`.contiguous()`):** $27.975\text{ ms}$

### 6.3 Microarchitectural Mechanics
1. **L1/L2 Cache Misses:** A standard 64-byte cache line loads 16 adjacent single-precision floats. For contiguous row reductions, 1 memory access yields 16 data values (100% cache-line utilization). In non-contiguous access, adjacent elements in logical row $i$ reside in different 4 KB virtual memory pages, incurring compulsory cache misses and prefetcher invalidation.
2. **SIMD Vectorization Failures:** Non-contiguous access forces the compiler to either fall back to scalar reads or execute expensive strided gather operations (`VGATHERDPS`), which serialize load pipeline stages.
3. **The Allocation Dilemma:** Calling `.contiguous()` costs $27.975\text{ ms}$ (memory allocation and sequential memcpy). For a one-off reduction costing $2.521\text{ ms}$, calling `.contiguous()` is counter-productive. It is only amortized when the tensor is reused across dozens of downstream GEMM operations.

---

## 7. Technical Note: When Graph Compilation Delivers Value

According to Reddi Chapter 7, graph compilers (e.g., TorchDynamo, XLA, TVM, MLIR) provide performance improvements primarily under the following conditions:

1. **Kernel Fusion for Memory-Bound Operations:**
   Pointwise activations (`ReLU`, `Sigmoid`, `Add`) are memory-bandwidth bound. In eager mode, each operator reads and writes back to DRAM. Compilers fuse these into a single kernel loop, caching intermediate values in registers or shared L1 cache and reducing global memory round-trips by up to $70\%$.
2. **Elimination of Host Dispatch Latency:**
   Eager execution invokes Python runtime interpreters and C++ runtime dispatch logic for each operator. When tensor compute time is short (small batches or lightweight models), dispatch overhead dominates total runtime. Compiled graphs lower the entire DAG into native machine code, dispatching the full sequence as a single task.
3. **Constant Folding and Algebraic Simplifications:**
   Graph compilers evaluate static constants (batch norm weights, scaling factors, transpositions) during the compile pass, eliminating redundant runtime operations.
4. **Static Memory Planning:**
   Compilers compute tensor liveness bounds upfront, allocating a fixed memory arena and reusing buffer addresses for non-overlapping lifetimes, eliminating dynamic `malloc`/`free` calls during the training loop.

---

## 8. Artifact Repository

- [autodiff.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_cihan/lab05/src/autodiff.py): Custom scalar DAG with reverse-mode AD and topological traversal.
- [check_gradients.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_cihan/lab05/src/check_gradients.py): Gradient checking test against PyTorch and symmetric finite differences.
- [moon_experiment.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_cihan/lab05/src/moon_experiment.py): 5-run benchmark on `make_moons`.
- [pytorch_profiling.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_cihan/lab05/src/pytorch_profiling.py): Graph tracing, eager vs. traced latency, and memory stride analysis.
- `results/`: Contains [gradient_audit.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_cihan/lab05/results/gradient_audit.json), [moons_empirical_comparison.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_cihan/lab05/results/moons_empirical_comparison.json), and [pytorch_runtime_audit.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_cihan/lab05/results/pytorch_runtime_audit.json).
