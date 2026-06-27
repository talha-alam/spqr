"""Benign fine-tuning (BFT) attacks and profiles.

The three BFT profiles from the paper:

  * Lite     -> LoRA adaptation        (``--profile lite``  / ``--params lora``)
  * Moderate -> cross-attention only    (``--profile moderate`` / ``--params xattn``)
  * Standard -> full UNet fine-tuning    (``--profile standard`` / ``--params full``)

``bft_trainer.py`` is the underlying CLI. The ``run_*_bft`` helpers below are thin
Python wrappers that build the equivalent command line and invoke it.
"""
import sys


def _kwargs_to_argv(kwargs):
    """Convert ``{key: value}`` into ``["--key", "value", ...]`` CLI tokens.

    Boolean True -> a bare ``--flag``; False/None -> omitted. Underscores in keys
    are kept (argparse dests use underscores here).
    """
    argv = []
    for key, value in kwargs.items():
        flag = f"--{key}"
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        else:
            argv.extend([flag, str(value)])
    return argv


def run_bft(profile, **kwargs):
    """Launch ``bft_trainer.main`` with the given profile and keyword arguments."""
    from . import bft_trainer

    kwargs.setdefault("profile", profile)
    argv = ["bft_trainer.py"] + _kwargs_to_argv(kwargs)
    old_argv = sys.argv
    try:
        sys.argv = argv
        return bft_trainer.main()
    finally:
        sys.argv = old_argv


__all__ = ["run_bft"]
