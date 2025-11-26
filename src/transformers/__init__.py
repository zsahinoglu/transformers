"""Transformers package - minimal scaffold."""

__all__ = ["version", "greet"]

version = "0.0.1"

def greet(name: str) -> str:
    """Return a greeting for the given name.

    This function is intentionally trivial - replace with your project logic.
    """
    return f"Hello, {name}!"
