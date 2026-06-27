"""
Standard BFT Profile: Full UNet fine-tuning.
Updates all parameters in the diffusion model's UNet.
"""
from . import run_bft


def run_standard_bft(**kwargs):
    """Run BFT with the Full UNet profile (Standard). Accepts CLI args as kwargs."""
    return run_bft("standard", **kwargs)


if __name__ == "__main__":
    import sys
    from . import bft_trainer
    sys.argv = sys.argv[:1] + sys.argv[1:] + ["--profile", "standard"]
    bft_trainer.main()
