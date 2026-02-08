# FAIRChem v1 environment recipe (fairchem-core==1.10.0 baseline)

This document is a **tested** recipe to build a clean FAIRChem v1 environment
that works with VPMDK’s `NNP=FAIRCHEM_V1` (OCPCalculator).

Key issues addressed:
- **PyG extras missing** (`torch-scatter`/`torch-sparse`) cause the
  `fairchem.core.common.relaxation.ase_utils` import to fail.
- **SciPy API change**: FAIRChem v1 uses `scipy.special.sph_harm`, which is
  removed in newer SciPy. Pin **SciPy <= 1.16.x**.

> ✅ The full sequence below was executed in this environment and validated.

Verified with `/tmp/fairchem_v1_env4` during this update. This is a venv-based
recipe; conda environments can follow the same dependency constraints.

---

## 1. Create a clean venv

```bash
python -m venv /tmp/fairchem_v1_env4
/tmp/fairchem_v1_env4/bin/pip install --upgrade pip
```

## 2. Install fairchem-core v1 baseline

```bash
/tmp/fairchem_v1_env4/bin/pip install fairchem-core==1.10.0
```

## 3. Install PyG extras (required by FAIRChem v1 imports)

> Adjust the wheel index to match your CUDA/PyTorch version. This example
> matches torch 2.4.x + CUDA 12.1.

```bash
/tmp/fairchem_v1_env4/bin/pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

## 4. Pin SciPy to a compatible version

```bash
/tmp/fairchem_v1_env4/bin/pip install "scipy==1.16.3"
```

## 5. Validate imports (OCPCalculator + sph_harm)

```bash
/tmp/fairchem_v1_env4/bin/python - <<'PY'
import importlib
from scipy.special import sph_harm
module = importlib.import_module("fairchem.core.common.relaxation.ase_utils")
print("OCPCalculator", getattr(module, "OCPCalculator", None))
print("module", module.__file__)
print("sph_harm", sph_harm)
PY
```

Expected output: `OCPCalculator` is present, and `sph_harm` imports without error.

---

## VPMDK usage notes (FAIRCHEM_V1)

When running VPMDK with `NNP=FAIRCHEM_V1`, you **must** provide:
- `MODEL=/path/to/checkpoint.pt`
- `FAIRCHEM_CONFIG=/path/to/config.yml` (required for most v1 checkpoints)

Example BCAR snippet:

```text
NNP=FAIRCHEM_V1
MODEL=/path/to/checkpoint.pt
FAIRCHEM_CONFIG=/path/to/config.yml
```

---

## Common pitfalls

1. **Missing torch-scatter/torch-sparse**
   - Symptom: `fairchem.core.common.relaxation.ase_utils` import fails.
   - Fix: install PyG extras from the correct wheel index (Step 3).

2. **SciPy too new**
   - Symptom: `ImportError: cannot import name 'sph_harm' from scipy.special`.
   - Fix: pin SciPy to `<=1.16.x` (Step 4).

3. **CUDA mismatch**
   - If you use CUDA 11.x or CPU-only builds, change the PyG wheel index
     accordingly. The `torch-scatter`/`torch-sparse` wheels must match your
     PyTorch/CUDA build.
