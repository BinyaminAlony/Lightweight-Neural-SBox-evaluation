"""Minimal runner: load known model and run on input [0]*32.

Usage:
  python run_model_on_zero_input.py
  python run_model_on_zero_input.py path/to/model.pt

This script assumes the saved file is either a TorchScript module or a saved
nn.Module instance (torch.save(model)). If the file is a state_dict only,
the script will exit with a short message.
"""
from pathlib import Path
import sys
import torch
import torch.nn as nn


MODEL_DIR = Path(__file__).parent / "models"
COMMON = [
    MODEL_DIR / "BNN_Model_50_10_Million_samples_LP_DEG_SAC_full.pt",
    MODEL_DIR / "BNN_Model_50_10_Million_samples_LP_DEG_SAC.pt",
]


def pick_model_path():
    # CLI override
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return p
        else:
            print(f"Provided path does not exist: {p}")
            return None

    for p in COMMON:
        if p.exists():
            return p

    # try any .pt/.pth in models
    if MODEL_DIR.exists():
        for ext in ("*.pt", "*.pth"):
            for f in MODEL_DIR.glob(ext):
                return f

    print("No model file found in ./models/. Please provide a path as the first argument.")
    return None


def main():
    path = pick_model_path()
    if path is None:
        return

    device = torch.device('cpu')
    # try TorchScript first
    model = None
    try:
        model = torch.jit.load(str(path), map_location=device)
        print(f"Loaded TorchScript model from {path}")
    except Exception:
        try:
            loaded = torch.load(str(path), map_location=device)
            if isinstance(loaded, nn.Module):
                model = loaded
                print(f"Loaded nn.Module instance from {path}")
            else:
                print(f"Loaded object is a {type(loaded)}; this script expects a saved nn.Module or TorchScript module.")
                return
        except Exception as e:
            print(f"Failed to load model: {e}")
            return

    model.eval()

    inp = torch.zeros((1, 32), dtype=torch.float32, device=device)
    print(f"Running model on input shape {tuple(inp.shape)}")
    with torch.no_grad():
        out = model(inp)

    print("Output type:", type(out))
    if isinstance(out, torch.Tensor):
        print("Output shape:", tuple(out.shape))
        print("Output (first 20 values):", out.flatten()[:20])
    else:
        print("Model returned non-tensor output:", out)


if __name__ == '__main__':
    main()
