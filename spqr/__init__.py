"""
SPQR: Safety-Prompt adherence-Quality-Robustness Benchmark
A standardized evaluation framework for safety-aligned text-to-image diffusion models.

Submodules (``metrics``, ``benchmark``, ``attacks``, ``generation``, ``utils``) are
imported lazily so that the lightweight scoring path
(``spqr.benchmark.scoring`` / ``SPQREvaluator.score_from_metrics``) is usable without
the full torch / diffusers stack installed.
"""
import importlib

__version__ = "0.1.0"
__author__ = "Mohammed Talha Alam et al."

_SUBMODULES = ("metrics", "benchmark", "attacks", "generation", "utils")

__all__ = list(_SUBMODULES)


def __getattr__(name):
    """PEP 562 lazy submodule import."""
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SUBMODULES))
