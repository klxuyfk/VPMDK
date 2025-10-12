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

VPMDK (the *Vasp-Protocol Machine-learning Dynamics Kit*) is a lightweight driver enabling atomic-level simulations using machine-learning potentials while maintaining the familiar Vienna Ab initio Simulation Package (VASP) style workflow. This toolkit reads and writes VASP-format artifacts (`POSCAR`, `INCAR`, `CONTCAR`, `OUTCAR`, `XDATCAR`, and related files), but redirects computationally intensive processing to machine learning potentials provided through the Atomic Simulation Environment (ASE) interface [@Larsen2017ASE]. This offers a lightweight surrogate model for many tools that traditionally relied on VASP.
The project currently supports CHGNet [@Deng2023CHGNet], M3GNet via MatGL [@Chen2022M3GNet], MACE [@Batatia2022MACE], and MatterSim [@Mattersim2023] calculators. Users can switch between the latest neural network potentials without abandoning their VASP-based data management.

# Statement of need

Many software packages and scripts exist for reaction-path exploration, structure optimization, and stability analysis in crystalline materials, but most are designed on the premise of calling density-functional-theory (DFT) codes—VASP [@Kresse1996Efficient] in particular. Recently, however, machine-learning potentials (MLPs) have emerged as powerful surrogate models for DFT, enabling much faster execution of these tasks. In practice, using MLPs typically requires ASE-based Python programming, and researchers often face a mismatch with their established workflows. VPMDK bridges this gap by providing a minimal command-line driver that accepts VASP-format inputs, performs fast computations with an MLP, and then produces VASP-format outputs. This allows researchers to reuse their existing workflows and infrastructure while substituting DFT with an MLP, thereby enabling high-throughput materials evaluation.

# Software description

## Architecture

The main entry point (`vpmdk.py`) is a concise driver script that orchestrates
four stages: parsing run configuration, preparing atomic structures, selecting a
calculator, and executing the requested simulation mode. Configuration is
collected from VASP `INCAR` and optional `BCAR` files, the latter using a simple
`key=value` schema to select the potential, optional parameter files, and
runtime toggles. Structures are read via pymatgen’s POSCAR and POTCAR handlers
before being converted to ASE atoms. Depending on `IBRION` and `NSW`, the driver
performs a single-point evaluation, a BFGS relaxation (optionally with cell
optimisation via ASE’s `UnitCellFilter`), or a velocity Verlet
molecular-dynamics trajectory. Outputs are written in standard VASP formats,
enabling compatibility with downstream tooling for structural analysis or
provenance tracking.

## Functionality

VPMDK abstracts over several state-of-the-art machine-learning potentials while
maintaining a uniform interface for end users. Each potential is loaded on
demand when requested, and informative errors are emitted if a dependency is
missing. The driver also supports optional conveniences such as automatically
wrapping fractional coordinates on output, writing per-step energies to CSV, and
respecting a subset of VASP’s ionic relaxation parameters. The script purposely
ignores electronic-structure-only settings, warning users about unsupported tags
so that legacy `INCAR` files can be reused with minimal editing.

# Example usage

A typical workflow begins with a directory containing the familiar VASP files:

```bash
$ ls calc_dir
BCAR  INCAR  POSCAR
```

The user selects a potential in `BCAR`, e.g. `NNP=CHGNET`, and runs:

```bash
python vpmdk.py --dir calc_dir
```

If the `INCAR` sets `IBRION<0`, the script performs a single-point evaluation
without moving ions, matching VASP's behaviour even when `NSW>0`. When
`IBRION=0`, it carries out molecular dynamics for `NSW` steps at the
temperature and timestep specified by `TEBEG` and `POTIM`. Positive `IBRION`
values trigger a geometry optimisation until the requested force convergence
(`EDIFFG`) or step limit is reached, producing updated VASP outputs for further
processing or submission to downstream workflows.

# Acknowledgements

VPMDK builds upon the open-source efforts of the ASE, pymatgen, CHGNet, MatGL,
MACE, and MatterSim communities.

# References

