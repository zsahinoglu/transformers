"""Demo script to run a Vision Transformer forward pass (if ML deps installed).

Run with:
    python examples/demo_vit.py

This script is intentionally safe — it prints helpful messages if `torch` or
`timm` are not installed rather than crashing on import.
"""
import sys

from transformers.vit import create_vit, forward_dummy


def main():
    try:
        model = create_vit(pretrained=False)
    except Exception as e:
        print("Cannot create ViT model:", e)
        print("Install ML dependencies: pip install -r requirements-ml.txt")
        sys.exit(1)

    shape = forward_dummy(model)
    print("Forward pass output shape:", shape)


if __name__ == "__main__":
    main()
