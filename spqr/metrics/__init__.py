"""SPQR Metrics: Safety (S), Prompt-adherence (P), Quality (Q), Robustness (R).

The metric computation helpers require the model stack (torch, transformers,
diffusers, nudenet, ...). They are exposed lazily so that, e.g.,
``from spqr.metrics import compute_fid_score`` does not pull in the NudeNet /
LLaVA-Guard safety stack (and vice-versa). The pure scoring / normalization math
lives in :mod:`spqr.benchmark.scoring` and has no heavy dependencies.
"""
import importlib

# Public name -> submodule that defines it.
_EXPORTS = {
    # Safety (S)
    "evaluate_safety": "safety",
    "compute_safety_score": "safety",
    "compute_harmfulness": "robustness",
    # Prompt-adherence (P)
    "compute_clip_score": "prompt_adherence",
    # Quality (Q)
    "compute_fid_score": "quality",
    # Robustness (R)
    "compute_robustness": "robustness",
    "compute_safety_delta": "robustness",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """PEP 562 lazy attribute import (only loads the needed submodule)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


def __dir__():
    return sorted(list(globals().keys()) + __all__)
