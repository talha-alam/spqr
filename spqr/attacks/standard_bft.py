"""
Standard BFT Profile: Full UNet fine-tuning.
Updates all parameters in the diffusion model.
"""
from .bft_trainer import main as bft_main

def run_standard_bft(**kwargs):
    """Run BFT with Full UNet profile (Standard)"""
    kwargs['params'] = 'full'
    return bft_main(**kwargs)

if __name__ == "__main__":
    run_standard_bft()
