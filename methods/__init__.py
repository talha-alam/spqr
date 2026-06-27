"""Safety-alignment method adapters and registry."""
from .base_method import BaseAlignmentMethod, MethodSpec, load_registry

__all__ = ["BaseAlignmentMethod", "MethodSpec", "load_registry"]
