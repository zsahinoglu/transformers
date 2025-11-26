"""Vision Transformer helpers for the Transformers project.

This module provides a small wrapper to create a Vision Transformer model
using `timm`. Imports `torch`/`timm` lazily so the package can be imported
without immediately installing heavy ML dependencies.
"""
from typing import Optional


def create_vit(model_name: str = "vit_base_patch16_224", pretrained: bool = False, device: Optional[str] = None):
    """Create and return a Vision Transformer model from timm.

    Args:
        model_name: timm model name (default: `vit_base_patch16_224`).
        pretrained: whether to load pretrained weights.
        device: 'cpu' or 'cuda' or None to auto-select.

    Returns:
        model: a torch.nn.Module instance.

    Raises:
        RuntimeError: if torch or timm are not installed.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover - platform-specific
        raise RuntimeError("torch is required to create a ViT model. Install torch or use requirements-ml.txt.") from e

    try:
        import timm
    except Exception as e:  # pragma: no cover - platform-specific
        raise RuntimeError("timm is required to create a ViT model. Install timm or use requirements-ml.txt.") from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # num_classes=0 returns feature extractor
    model.to(device)
    model.eval()
    return model


def forward_dummy(model, image_size: int = 224):
    """Run a dummy forward pass returning feature shape.

    Args:
        model: torch.nn.Module
        image_size: input image size (square)

    Returns:
        output tensor shape tuple
    """
    import torch

    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        out = model(dummy)
    return out.shape
