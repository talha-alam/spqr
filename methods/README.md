# Safety-Alignment Methods

SPQR benchmarks **11 representative safety-alignment methods**. The benchmark is
*checkpoint-driven*: each method is evaluated from the aligned model checkpoint it
produces, converted to the diffusers format with
[`spqr/utils/checkpoint_converter.py`](../spqr/utils/checkpoint_converter.py).

- The canonical list (name, intervention family, venue, official repo) lives in
  [`configs/methods.yaml`](../configs/methods.yaml) and is loadable via
  `methods.load_registry()`.
- Most methods need **no custom code** — just a converted checkpoint.
- Methods that require special loading (e.g. an adapter/safety module) should
  subclass [`BaseAlignmentMethod`](base_method.py) in a `methods/<method>/`
  subpackage.

```python
from methods import load_registry
reg = load_registry()
print(reg["rece"].name, reg["rece"].venue, reg["rece"].repo)
```

| Family | Methods |
|--------|---------|
| Conditioning-space edits | RECE, MACE, SPM |
| Attention-path edits | ESD, SalUn, EraseDiff, Scissorhands |
| Parameter-space unlearning | UCE, STEREO, AdvUnlearn, FMN |

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for how to add a new method.
