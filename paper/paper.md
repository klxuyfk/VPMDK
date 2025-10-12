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

VPMDK (the *Vasp-Protocol Machine-learning Dynamics Kit*) is a lightweight driver
that enables atomistic simulations with machine-learning interatomic potentials
while preserving familiar Vienna *Ab initio* Simulation Package (VASP) style
workflows. The toolkit reads and writes VASP-format artefacts—`POSCAR`, `INCAR`,
`CONTCAR`, `OUTCAR`, `XDATCAR`, and related files—yet redirects the heavy
computation to machine-learning force fields delivered through the Atomic
Simulation Environment (ASE) interface [@Larsen2017ASE]. The project currently
supports CHGNet [@Deng2023CHGNet], M3GNet via MatGL [@Chen2022M3GNet], MACE
[@Batatia2022MACE], and MatterSim [@Mattersim2023] calculators, enabling users
to swap among modern neural network potentials without abandoning VASP-based
data management.

# Statement of need

Producing training data for machine-learning interatomic potentials usually
leverages electronic structure codes such as VASP [@Kresse1996Efficient],
yielding workflows built around VASP’s file formats and job structure. When the
time comes to deploy machine-learning models for large-scale molecular dynamics
or structural relaxations, practitioners often face a disconnect between their
accustomed VASP artefacts and the Python tooling surrounding modern potentials.
VPMDK bridges this gap by providing a minimal command-line driver that accepts
VASP-formatted inputs, configures the desired potential, and writes outputs in
VASP style. Researchers can therefore prototype or replace density-functional
calculations with machine-learning surrogates while reusing existing workflows
and infrastructure for job submission, visualisation, and analysis. The project
builds on the robust file parsing of pymatgen [@Ong2013Python] and the
flexibility of ASE calculators, ensuring compatibility with a wide ecosystem of
materials informatics software.

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

# Quality control

Automated tests (via `pytest`) cover the critical control flow of the driver:
parsing of POSCAR/INCAR data, selection of potentials, propagation of
relaxation parameters, and wrapping of atomic coordinates (see
`tests/test_vpmdk.py`). The test suite uses ASE’s reference EMT potential as a
lightweight stand-in to validate the driver’s logic across all supported
calculator types, ensuring reproducibility without requiring heavyweight
machine-learning models.

# Example usage

A typical workflow begins with a directory containing the familiar VASP files:

```bash
$ ls calc_dir
BCAR  INCAR  POSCAR  POTCAR
```

The user selects a potential in `BCAR`, e.g. `NNP=CHGNET`, and runs:

```bash
python vpmdk.py --dir calc_dir
```

If the `INCAR` sets `IBRION=0`, the script carries out molecular dynamics for
`NSW` steps at the temperature and timestep specified by `TEBEG` and `POTIM`. If
`IBRION` is non-zero, it performs a geometry optimisation until the requested
force convergence (`EDIFFG`) or step limit is reached, producing updated VASP
outputs for further processing or submission to downstream workflows.

# Acknowledgements

VPMDK builds upon the open-source efforts of the ASE, pymatgen, CHGNet, MatGL,
MACE, and MatterSim communities.

# References

