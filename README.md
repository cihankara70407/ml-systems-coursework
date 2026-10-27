# Deep Learning & Systems Laboratory Portfolio

**Student Researcher:** Cihan Karayigit (ID: 70407)  
**Academic Program:** Computer Engineering, Vistula University  
**Subject:** Machine Learning Systems Coursework  
**Reference Text:** Vijay Janapa Reddi, *Machine Learning Systems* (Chapters 1 through 6)  

---

## Abstract & Research Architecture

This coursework repository explores the intersection of deep learning algorithms and system execution constraints, bridging theoretical calculus with empirical profiling across four dedicated laboratory units:

- **Laboratory 01 (Systems Telemetry):** Empirical characterization of model size, training speed, inference latency, and memory footprints on tabular diagnostic baselines.
- **Laboratory 02 (Deep Learning Primer):** Complete derivation and NumPy implementation of a multi-layer perceptron without automatic differentiation, validated against two-sided numerical gradient checking.
- **Laboratory 03 (DNN Architectures & Cost):** Comparative study of MLP, CNN, and Vision Transformer models on Fashion-MNIST, contrasting manual FLOP derivations with `thop` MAC metrics.
- **Laboratory 04 (Data Engineering Integrity):** Production ETL architecture implementing lossless raw ingestion, schema contract enforcement, and leakage-free transformations.

## Repository Layout
```
├── lab01/    # Diagnostic classification benchmarks and execution profiles
├── lab02/    # Scratch neural network engine and gradient check logs
├── lab03/    # Deep architectures, handwritten calculations, and Pareto frontier curves
└── lab04/    # Production data transformer, audit reports, and serialized pipeline
```

## Quickstart Guide
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
