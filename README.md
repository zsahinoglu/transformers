
# Transformers — Vision Transformers research & experiments

This repository is a minimal scaffold for working on Vision Transformer (ViT)
research, experiments, and utilities.

Quick layout

- `src/transformers/` — Python package source
- `tests/` — unit tests
- `examples/` — runnable example scripts (see `examples/demo_vit.py`)
- `requirements-ml.txt` — optional heavy ML dependencies (torch, timm)

Getting started (fast)

1. Create a virtualenv and install light dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

2. Run the smoke test (quick):

```bash
pytest -q
```

3. To run Vision Transformer demos and experiments, install ML deps:

```bash
pip install -r requirements-ml.txt
python examples/demo_vit.py
```

Project status

This is an initial scaffold. Next steps may include adding dataset utilities, training scripts, model evaluation, and experiment tracking.

