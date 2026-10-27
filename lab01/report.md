# Lab 01: Environment and First System Measurements

**Author:** Cihan Karayigit (Student ID: 70407)  
**Assigned Readings:** Reddi, Chapters 1, 2, and 12  

---

## 1. Goal
This laboratory module validates the local machine learning runtime environment, trains linear and ensemble predictive baselines on the Breast Cancer Wisconsin diagnostic dataset under a deterministic stratified split, benchmarks system-level cost vectors (training duration, single-sample inference latency, resident process memory, and disk footprint), and evaluates model deployment compatibility across four discrete hardware tiers: Cloud, Edge, Mobile, and TinyML.

---

## 2. Method
1. **Isolated Runtime Verification:** Verified the pinned dependency stack according to course specifications, storing environment metadata in `results/versions.txt`.
2. **Deterministic Partitioning:** Loaded Breast Cancer Wisconsin data (569 records, 30 continuous features) and stratified 70/30 with `random_state=42` (398 train, 171 test).
3. **Model Selection:**
   - Linear Baseline: `LogisticRegression(max_iter=1000, random_state=42)`
   - Tree Ensemble: `RandomForestClassifier(n_estimators=100, random_state=42)`
4. **Benchmarking Protocol:**
   - Training latency: 1 warm-up run followed by 5 repetitions; recorded median duration.
   - Inference latency: 100 consecutive predictions on a single sample vector; recorded median duration.
   - Footprint: Serialized estimators with `joblib.dump` to determine exact byte and kilobyte requirements.
   - Memory profile: Sampled resident set size (RSS) via `psutil`.
5. **Deployment Suitability Matrix:** Mapped empirical footprints against standardized hardware constraints.

---

## 3. Results Table or Plot

### Empirical Systems Profiling Table

| Performance Vector | Linear Classifier (`LogisticRegression`) | Ensemble (`RandomForestClassifier`) | Measurement Protocol |
| :--- | :---: | :---: | :--- |
| **Classification Accuracy** | **0.9415** (94.15%) | **0.9357** (93.57%) | Stratified Test Split (30%) |
| **Training Latency (Median)** | **145.65 ms** | **177.59 ms** | Median of 5 runs (post-warmup) |
| **Inference Latency (Median)** | **0.1804 ms** | **7.2101 ms** | Median of 100 single-sample passes |
| **Serialized Footprint** | **1,103 bytes** (1.08 KB) | **290,889 bytes** (284.07 KB) | On-disk `joblib` binary |
| **Process RSS Memory** | **~430.3 MB** | **~430.3 MB** | Host process resident set size |

### Hardware Tier Feasibility Assessment

| Target Tier | Memory Limit | Latency Limit | Storage Limit | Logistic Regression Status | Random Forest Status | Architectural Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cloud** | >= 1 GB | <= 100 ms | <= 500 MB | **Supported** | **Supported** | Negligible load; both models operate without infrastructure constraints. |
| **Edge Gateway** | 256-1024 MB | <= 50 ms | <= 50 MB | **Supported** | **Supported** | Excellent throughput with sub-millisecond per-sample latency. |
| **Mobile Device** | 64-256 MB | <= 20 ms | <= 10 MB | **Supported** | **Supported** | Compact storage (< 284.1 KB) well within mobile memory bounds. |
| **TinyML Microcontroller** | <= 256 KB | <= 10 ms | <= 100 KB | **Supported** (1.08 KB) | **Unfeasible** (284.07 KB) | Linear weights fit into embedded SRAM/Flash; ensemble trees exceed 100 KB limit. |

---

## 4. Three Conclusions in Your Own Words
1. **Model Simplicity Unlocks Embedded & Microcontroller Deployments:** The linear model requires only 1.08 KB of storage while achieving superior test accuracy (94.15%) compared to the ensemble, confirming that simpler inductive biases can dominate on structured low-dimensional diagnostic data while enabling deployment on TinyML chips.
2. **Decision Tree Depth Imposes Exponential Storage and Latency Costs:** Constructing 100 decision trees expands storage by 264x and increases single-sample inference time by 40.0x due to pointer chasing across branching conditions.
3. **Statistical Median Reporting Guarantees Deterministic Baselines:** Profiling inference over 100 repetitions and adopting median filtering eliminated operating system context-switching outliers, establishing a solid baseline for hardware deployment budgets.
