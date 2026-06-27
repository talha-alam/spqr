"""
Lite BFT Profile: LoRA-based parameter-efficient fine-tuning.
Injects low-rank adapters into the attention projections while the UNet stays frozen.
"""
from . import run_bft


def run_lite_bft(**kwargs):
    """Run BFT with the LoRA profile (Lite). Accepts bft_trainer CLI args as kwargs."""
    kwargs.setdefault("lora_rank", 8)
    kwargs.setdefault("lora_alpha", 16)
    return run_bft("lite", **kwargs)


if __name__ == "__main__":
    # Delegate to the trainer CLI with the lite profile forced on.
    import sys
    from . import bft_trainer
    sys.argv = sys.argv[:1] + sys.argv[1:] + ["--profile", "lite"]
    bft_trainer.main()
