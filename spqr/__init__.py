"""
SPQR: Safety-Prompt adherence-Quality-Robustness Benchmark
A standardized evaluation framework for safety-aligned text-to-image diffusion models.
"""

__version__ = "0.1.0"
__author__ = "Mohammed Talha Alam et al."

from . import metrics
from . import benchmark
from . import attacks
from . import generation
from . import utils

__all__ = [
    'metrics',
    'benchmark',
    'attacks',
    'generation',
    'utils',
]
