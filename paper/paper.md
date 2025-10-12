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

![Overview of `VPMDK`](fig1.png)

# Major Features

VPMDK centres on a streamlined Python driver that mirrors the familiar VASP
workflow while routing all expensive calculations through modern machine
learning potentials. Run configuration is derived from the standard `INCAR`
file, augmented by an optional `BCAR` control file that declares the desired
potential, auxiliary parameters, and runtime switches using an intuitive
`key=value` syntax. Atomic structures are ingested through pymatgen's robust
POSCAR/POTCAR parsers and immediately converted to ASE atoms, ensuring
consistency with existing VASP repositories.

Once configured, the driver dynamically loads the requested potential—CHGNet,
M3GNet (via MatGL), MACE, or MatterSim—and exposes them through a unified
interface. The simulation mode is automatically inferred from familiar VASP
settings: negative `IBRION` triggers single-point evaluations, zero selects
velocity-Verlet molecular dynamics with thermostatting via `TEBEG` and `POTIM`,
and positive values launch BFGS relaxations optionally wrapped in ASE's
`UnitCellFilter` for cell optimisation. Throughout a run the code preserves
VASP-style artefacts, including `CONTCAR`, `OUTCAR`, and `XDATCAR`, enabling
seamless hand-off to legacy analysis scripts. Supplementary conveniences such as
automatic coordinate wrapping, per-step CSV energy dumps, and clear warnings for
unsupported VASP flags help researchers transition existing workflows to machine
learning potentials without sacrificing usability or provenance.

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

