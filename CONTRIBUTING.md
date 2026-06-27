# Contributing to SPQR

Thanks for your interest in extending the **SPQR** benchmark! This guide explains
how to add a new safety-alignment method or fine-tuning scenario and how to submit
results.

## Development setup

```bash
git clone https://github.com/talha-alam/spqr.git
cd spqr
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .            # editable install of the `spqr` package
```

## Repository layout

| Path | Purpose |
|------|---------|
| `spqr/metrics/` | Per-axis metric computation (S, P, Q, R) |
| `spqr/benchmark/scoring.py` | Pure normalization + SPQR aggregation math (Eqs. 2–7) |
| `spqr/benchmark/evaluator.py` | `SPQREvaluator` — orchestrates the four axes |
| `spqr/attacks/` | Benign fine-tuning (BFT) trainer + the 3 profiles |
| `spqr/generation/` | Image generation for the harmful / benign prompt sets |
| `spqr/utils/` | Checkpoint conversion and helpers |
| `methods/` | Method registry + the adapter interface (`methods/base_method.py`) |
| `configs/` | BFT profiles, datasets, and the method registry |
| `scripts/` | CLI entry points (`run_benchmark.py`, `run_evaluation.py`, `prepare_datasets.py`) |

## Adding a new safety-alignment method

1. **Produce an aligned checkpoint** with the method's official implementation and
   convert it to the diffusers format:
   ```bash
   python spqr/utils/checkpoint_converter.py \
       --input_dir path/to/ckpts --output_dir aligned_models/
   ```
2. **Register the method** in [`configs/methods.yaml`](configs/methods.yaml)
   (name, intervention family, venue, official repo).
3. *(Optional)* Implement a thin wrapper in `methods/<your_method>/` subclassing
   `methods.base_method.BaseAlignmentMethod` if the method needs custom loading.
4. **Run the full benchmark** and collect S, P, Q, R, SPQR:
   ```bash
   python scripts/run_benchmark.py --method <your_method> \
       --model_path aligned_models/<your_method> \
       --bft_profile standard --scenario general --output_dir results/<your_method>
   ```
5. **Open a PR** with your method registration and the produced `results/` JSON.

## Adding a new BFT scenario or dataset

- Prepare the benign dataset into an `imagefolder` layout with `metadata.jsonl`
  (see `scripts/prepare_datasets.py`).
- Register paths/sizes in [`configs/datasets.yaml`](configs/datasets.yaml).

## Code style

- Keep new metric math in `spqr/benchmark/scoring.py` as pure functions (no torch),
  so it remains unit-testable without a GPU.
- Match the surrounding style; prefer small, documented functions.
- Run a quick import smoke test before opening a PR:
  ```bash
  python -c "from spqr.benchmark import SPQREvaluator, spqr_score; print('ok')"
  ```

## Reporting issues / data access

- Bugs and feature requests: open a GitHub issue.
- Access to the gated harmful prompt sets: see [`DATA_ACCESS.md`](DATA_ACCESS.md).

By contributing you agree that your contributions are licensed under the
repository's [MIT License](LICENSE).
