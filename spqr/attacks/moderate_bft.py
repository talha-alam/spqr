"""
Moderate BFT Profile: Cross-attention only fine-tuning.
Updates only cross-attention layers (attn2.to_k, attn2.to_v, attn2.to_out).
"""
from . import run_bft


def run_moderate_bft(**kwargs):
    """Run BFT with the Cross-Attention profile (Moderate). Accepts CLI args as kwargs."""
    return run_bft("moderate", **kwargs)


if __name__ == "__main__":
    import sys
    from . import bft_trainer
    sys.argv = sys.argv[:1] + sys.argv[1:] + ["--profile", "moderate"]
    bft_trainer.main()
