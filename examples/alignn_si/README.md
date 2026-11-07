# ALIGNN-FF silicon example

Install VPMDK and ALIGNN using the [environment recipe](../../docs/development/backend-environments.md#alignn-ff), then run from the repository root:

```bash
vpmdk --dir examples/alignn_si
```

The first run downloads the public `alignn_ff_mptrj` checkpoint (about 19 MB)
from the upstream Figshare registry into `~/.cache/atomgptlab/alignn_ff`.
The bundled input relaxes a displaced two-atom diamond Si cell for up to five
steps. `OUTCAR`, `OSZICAR`, `CONTCAR`, and `vasprun.xml` appear in this directory.
Five steps are a workflow smoke test and do not guarantee convergence.

For offline use, set `MODEL` in BCAR to an absolute directory containing
`best_model.pt` and `config.json`, or to a checkpoint file with a sibling
`config.json`. Keep both files from the same model download. To download ahead
of time:

```python
from alignn.ff.calculators import get_figshare_model_ff
print(get_figshare_model_ff(model_name="alignn_ff_mptrj"))
```

For a static evaluation use `IBRION=-1`, `NSW=0`. For a short MD smoke test use:

```ini
IBRION = 0
NSW = 5
POTIM = 0.5
TEBEG = 100
TEEND = 100
MDALGO = 0
```

The [validation record](../../docs/development/validation.md#2026-11-06-alignn-ff-validation)
contains the tested runtime, checkpoint checksum, and numerical checks.

GPU execution was validated on NVIDIA TITAN V. Set `DEVICE=cuda` for the
current model. For the validated legacy DGL model, use
`MODEL=v12.2.2024_dft_3d_307k` in the separate
[DGL environment](../../docs/development/backend-environments.md#legacy-dgl-checkpoints-on-cpu-and-cuda).
`v5.27.2024` failed gradient consistency checks and is rejected.
CPU/GPU numerical reports are saved in `reference/pure-cpu-gpu.json` and
`reference/dgl-cpu-gpu.json`.
