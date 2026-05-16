# API Examples With CHGNet

This directory shows how to use VPMDK as a Python library instead of the VASP-compatible CLI.

These examples use `CHGNET` because it can run without an external checkpoint file when the package is installed.

## Files

- `POSCAR`: small Si cell used by the scripts
- `single_point.py`: run one energy/force/stress evaluation
- `relax.py`: run a short ionic relaxation and write `relaxed.vasp`
- `list_backends.py`: inspect backend availability and capability metadata
- `reference/`: generated output snapshots from these scripts

## Run

From repository root:

```bash
python examples/api_chgnet/single_point.py
python examples/api_chgnet/relax.py
python examples/api_chgnet/list_backends.py
```

Or from inside this directory:

```bash
python single_point.py
python relax.py
python list_backends.py
```

## Notes

- These are library examples, so they do not write `OUTCAR`/`OSZICAR`/`vasprun.xml` by default.
- `relax.py` explicitly writes `relaxed.vasp` so the optimized structure is easy to inspect.
- The `list_backends.py` reference records the backend availability in the environment used to generate the snapshot.
- If you want VASP-style compatibility outputs from Python, attach `vpmdk.VaspCompatObserver()` with `vpmdk.compat.vasp.VaspCompatConfig(...)`.
