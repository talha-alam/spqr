"""
Moderate BFT Profile: Cross-attention only fine-tuning.
Updates only cross-attention layers (attn2.to_k, attn2.to_v, attn2.to_out).
"""
from .bft_trainer import main as bft_main

def run_moderate_bft(**kwargs):
    """Run BFT with Cross-Attention profile (Moderate)"""
    kwargs['params'] = 'xattn'
    return bft_main(**kwargs)

if __name__ == "__main__":
    run_moderate_bft()
