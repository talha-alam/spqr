"""
Lite BFT Profile: LoRA-based parameter-efficient fine-tuning.
Updates only low-rank adapters while keeping UNet frozen.
"""
from .bft_trainer import main as bft_main
import sys

def run_lite_bft(**kwargs):
    """Run BFT with LoRA profile (Lite)"""
    # Set LoRA-specific parameters
    kwargs['params'] = 'lora'
    # Add other LoRA-specific configs
    return bft_main(**kwargs)

if __name__ == "__main__":
    run_lite_bft()
