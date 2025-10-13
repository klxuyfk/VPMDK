---
title: 'VPMDK: VASP-protocol machine-learning dynamics kit'
tags:
  - Python
  - atomistic simulations
  - machine-learning potentials
  - molecular dynamics
  - materials science
authors:
  - name: Kagami Aso
    orcid: 0009-0005-8266-3557
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Corteo LLC, Japan
   index: 1
date: 12 October 2025
bibliography: paper.bib
---

# Summary

VPMDK (the *Vasp‑Protocol Machine‑learning Dynamics Kit*) is a lightweight command‑line driver that preserves the familiar Vienna *Ab initio* Simulation Package (VASP) workflow while executing atomistic simulations with modern machine‑learning interatomic potentials (MLPs). The tool reads VASP‑style inputs (`POSCAR`, `INCAR`, optional `POTCAR` and `BCAR`) and emits VASP‑style outputs (`CONTCAR`, `OUTCAR`, `XDATCAR`), but routes energy/force/stress evaluation to ASE‑compatible calculators, currently including CHGNet, M3GNet (via MatGL), MACE, and MatterSim [@Larsen2017ASE; @Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024]. By acting as a thin shim rather than a full re‑implementation of VASP, VPMDK makes it possible to swap DFT with an MLP surrogate in existing VASP‑centric data and workflow ecosystems without rewriting pipelines or abandoning well‑understood artefacts.

# Statement of need

High‑throughput screening, structure relaxation, and molecular dynamics in crystalline materials are frequently orchestrated around VASP’s I/O conventions. Meanwhile, MLPs have matured into practical surrogates trained on large DFT corpora, offering orders‑of‑magnitude speedups for many tasks while retaining useful accuracy across broad chemistries [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024]. In practice, however, deploying MLPs often requires ASE‑based Python scripts and data layouts that diverge from VASP‑style workflows, creating friction for users who rely on legacy post‑processing and workflow managers.

VPMDK addresses this gap with a minimal driver that speaks the VASP dialect on disk yet computes with an MLP under the hood. Researchers can therefore (i) reuse existing input repositories and provenance policies, (ii) keep downstream analysis that expects `OUTCAR`/`XDATCAR`, and (iii) rapidly switch among state‑of‑the‑art neural potentials for pre‑screening, initial relaxations, finite‑temperature MD, or dataset generation—often as a drop‑in substitute for expensive DFT calls.

![Overview of `VPMDK`](fig1.png)

# Design and implementation

**I/O compatibility.** Structures are loaded from `POSCAR` via *pymatgen* and converted to ASE `Atoms`. If present, `POTCAR` is used only to reconcile species ordering; wavefunctions or charge densities (`WAVECAR`, `CHGCAR`) are detected but intentionally ignored. Initial magnetic moments are parsed from `INCAR`’s `MAGMOM` (including VASP shorthand such as `2*1.0`) and propagated to ASE when counts match [@Larsen2017ASE; @Ong2013pymatgen].

**Configuration model.** Runtime behavior is driven primarily by a subset of VASP’s `INCAR` keys: `NSW` and `IBRION` choose single‑point (<0), MD (=0), or relaxation (>0). `TEBEG`/`TEEND` (K) and `POTIM` (fs) control MD, and `EDIFFG` follows VASP semantics: negative values set a force threshold (eV/Å) for relaxations, whereas positive values request convergence by total‑energy change between ionic steps. Crystal degrees of freedom are governed by `ISIF`; when cell updates are requested, relaxation wraps ASE’s filters (e.g., `UnitCellFilter`) and converts `PSTRESS` from kBar to eV/Å³. Unsupported tags are warned but safely ignored.

**MLP backends.** The optional `BCAR` control file (simple `key=value`) selects the calculator (`NNP=CHGNET|MATGL|MACE|MATTERSIM`) and, if needed, `MODEL=/path/to/parameters`. CHGNet and MatGL/M3GNet ship with default models; MACE and MatterSim can load user‑trained weights. Where available, GPU usage is delegated to the backend (e.g., auto‑select for MACE).

**Dynamics and thermostats.** For MD (`IBRION=0`), VPMDK uses velocity‑Verlet integration and supports common thermostats via ASE: Andersen (`MDALGO=1`), Nose–Hoover chains (`2` and `4`), Langevin (`3`), and canonical sampling velocity rescaling (Bussi; `5`). Temperatures are optionally ramped linearly from `TEBEG` to `TEEND`. Trajectories are written to `XDATCAR` incrementally, while `OUTCAR` records stepwise energies and temperature.

**Relaxation behavior.** Geometry optimizations use BFGS, printing to `OUTCAR` and writing `CONTCAR` on completion. If requested via `BCAR`, a per‑step `energy.csv` is emitted for quick inspection. Cell updates (e.g., `ISIF=3,4,6–8`) map to ASE’s filters, with temporary ionic freezing when VASP semantics require cell‑only steps.

**Outputs and provenance.** VPMDK intentionally mirrors VASP artefacts—`CONTCAR` for the final structure, `OUTCAR` for a human‑readable log, and `XDATCAR` for MD trajectories—so that downstream tools expecting VASP I/O continue to work with MLP‑generated results. The tool does **not** attempt to emulate electronic‑structure features (k‑points, smearing, or charge densities); such files are recognized only to aid mixed DFT/MLP pipelines.

# Usage

Place `POSCAR` (and optionally `INCAR`, `POTCAR`, `BCAR`) in a directory and run:

```bash
python vpmdk.py --dir calc_dir
```

`INCAR` chooses mode and control parameters; `BCAR` selects the potential and optional `MODEL`. Results appear as `CONTCAR`, `OUTCAR`, and, for MD, `XDATCAR`.

# Limitations and scope

VPMDK is not affiliated with, endorsed by, or a drop‑in replacement for VASP; it only mimics VASP I/O for convenience. Only a subset of `INCAR` keys are honored, and electronic‑structure quantities (k‑point meshes, wavefunctions, charge densities) are out of scope. Accuracy and transferability are those of the chosen MLP and its training regime; users should verify applicability for their chemistry and conditions [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024].

# Acknowledgements

We thank the developers and maintainers of ASE and pymatgen for foundational infrastructure, and the authors of CHGNet, M3GNet/MatGL, MACE, and MatterSim for making high‑quality MLPs broadly available.

# References

