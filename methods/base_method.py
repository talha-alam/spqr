"""
Adapter interface for safety-alignment methods.

SPQR is checkpoint-driven: each method is benchmarked from the *aligned model
checkpoint* it produces (converted to the diffusers format). Most methods need no
custom code beyond conversion. When a method requires special loading (e.g. an
adapter/LoRA-style safety module, or a custom scheduler), subclass
``BaseAlignmentMethod`` and register it in ``methods/__init__.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class MethodSpec:
    """Static metadata for a registered method (mirrors configs/methods.yaml)."""
    key: str
    name: str
    family: str
    venue: str
    repo: str
    description: str = ""
    extra: dict = field(default_factory=dict)


class BaseAlignmentMethod:
    """Base class for loading a safety-aligned T2I model for benchmarking.

    The default implementation loads a diffusers-format ``StableDiffusionPipeline``
    from ``model_path``. Override :meth:`load_pipeline` for methods that attach an
    adapter or otherwise diverge from a plain checkpoint.
    """

    spec: Optional[MethodSpec] = None

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device

    def load_pipeline(self, torch_dtype=None, safety_checker=None):
        """Load and return the aligned diffusers pipeline."""
        import torch
        from diffusers import StableDiffusionPipeline

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch_dtype or (torch.float16 if "cuda" in str(device) else torch.float32)
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path, torch_dtype=torch_dtype, safety_checker=safety_checker
        )
        return pipe.to(device)


def load_registry(config_path: Optional[str] = None) -> dict:
    """Load configs/methods.yaml into ``{key: MethodSpec}``."""
    if config_path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(here, "configs", "methods.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    registry = {}
    for key, entry in (cfg.get("methods") or {}).items():
        registry[key] = MethodSpec(
            key=key,
            name=entry.get("name", key),
            family=entry.get("family", ""),
            venue=entry.get("venue", ""),
            repo=entry.get("repo", ""),
            description=entry.get("description", ""),
            extra={k: v for k, v in entry.items()
                   if k not in {"name", "family", "venue", "repo", "description"}},
        )
    return registry
