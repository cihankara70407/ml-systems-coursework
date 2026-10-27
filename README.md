# Machine Learning Systems & Architectures — Laboratory Coursework

**Author:** Cihan Karayigit  
**Student ID:** 70407  
**Coursework Text:** *Vijay Janapa Reddi, Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems.*  
**Environment Target:** Python 3.11.8 (Standard Isolated Stack)  

---

## Directory Organization

This repository implements the four foundational laboratory modules prescribed in the Machine Learning Systems syllabus:

```
ml-systems-coursework/
├── README.md                 # Primary portfolio index and documentation
├── requirements.txt          # Frozen dependency specifications
├── .gitignore                # Repository hygiene and artifact exclusions
├── lab01/                    # Module 1: Environment & Systems Measurement (Ch. 1, 2, 12)
│   ├── report.md
│   ├── src/
│   └── results/
├── lab02/                    # Module 2: Neural Network from First Principles (Ch. 3)
│   ├── report.md
│   ├── src/
│   └── results/
├── lab03/                    # Module 3: Comparative DNN Architectures & Computational Costs (Ch. 4)
│   ├── report.md
│   ├── src/
│   └── results/
└── lab04/                    # Module 4: Resilient Data Pipeline & Quality Engineering (Ch. 5-6)
    ├── report.md
    ├── src/
    └── results/
```

---

## Modules Directory

| Module | Topic & Objectives | Reddi Textbook | Delivered Evidence | Completion |
| :---: | :--- | :--- | :--- | :---: |
| **Lab 01** | **Systems Resource Measurement & Budgets** | Ch. 1, 2, 12 | Pinned runtime report, latency distributions, RSS memory, edge/TinyML budget matrix | Complete |
| **Lab 02** | **Neural Network from Scratch (Pure NumPy)** | Ch. 3 | Matrix calculus chain rule, 10/10 finite-difference slope check, GEMM breakdown | Complete |
| **Lab 03** | **Comparative DNN Architectures & Costs** | Ch. 4 | PyTorch MLP, CNN, and Patch ViT; hand vs thop FLOPs derivation, Pareto curves | Complete |
| **Lab 04** | **Data Pipeline Quality & Integrity** | Ch. 5, 6 | Lossless ingestion, zero-leakage transformation, feature synthesis, label noise audit | Complete |

---

## Autonomous Execution Guide

```bash
# Clone the repository
git clone https://github.com/cihankara70407/ml-systems-coursework.git
cd ml-systems-coursework

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run all modules
python lab01/src/evaluate_baselines.py
python lab02/src/main.py
python lab03/src/evaluate_architectures.py
python lab04/src/main.py
```
