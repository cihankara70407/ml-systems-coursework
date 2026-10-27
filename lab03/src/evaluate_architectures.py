import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from architectures.fully_connected import CihanDeepMLP
from architectures.convolutional import CihanSpatialCNN
from architectures.transformer import CihanSelfAttentionViT
from architectures.latency_benchmark import profile_efficiency

def main():
    print("[INFO] Auditing Cihan Karayigit Architectures...")
    models = {
        "MLP (BatchNormalized)": CihanDeepMLP(),
        "CNN (Spatial Convolutions)": CihanSpatialCNN(),
        "ViT (Self-Attention)": CihanSelfAttentionViT()
    }
    
    metrics = {}
    for name, m in models.items():
        res = profile_efficiency(m)
        metrics[name] = res
        print(f"[{name}] Params: {res['parameters']:,} | MACs: {res['macs']:,} | Latency: {res['latency_ms']:.3f} ms")
        
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmarks.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SUCCESS] Exported results to {out_dir}")

if __name__ == "__main__":
    main()
