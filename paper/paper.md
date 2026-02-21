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
 - name: Corteo Co., Ltd., Japan
   index: 1
date: 17 October 2025
bibliography: paper.bib
---

# Summary

VPMDK (the *Vasp-Protocol Machine-learning Dynamics Kit*) is a lightweight command-line driver that preserves the familiar Vienna *Ab initio* Simulation Package (VASP) workflow while running atomistic simulations with machine-learning interatomic potentials (MLPs). It reads VASP-style inputs (`POSCAR`, `INCAR`, optional `POTCAR` and `BCAR`) and writes VASP-style outputs (`CONTCAR`, `OUTCAR`, `XDATCAR`), while delegating energy/force/stress evaluation to ASE-compatible calculators [@Larsen2017ASE]. Supported backends include CHGNet, M3GNet (via MatGL), MACE, MatterSim, Matlantis, NequIP/Allegro, SevenNet, DeePMD-kit, ORB, FAIRChem, and GRACE [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024; @PFPMatl2024; @Batzner2022NequIP; @Musaelian2023Allegro; @Oba2024SevenNet; @Zhang2018DeePMD; @Wang2018DeePMD; @OrbModels2024; @Hegde2024FAIRChem; @Choudhary2024GRACE].

# Statement of need

Many high-throughput relaxation and molecular-dynamics pipelines in materials science are built around VASP I/O conventions. At the same time, modern MLPs provide practical speedups for many workloads [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024; @Oba2024SevenNet; @Hegde2024FAIRChem; @Choudhary2024GRACE]. In practice, adopting these models often means rewriting workflows around custom Python scripts.

VPMDK targets this migration cost. It keeps VASP-style files and control flow while enabling users to switch among multiple MLP backends for screening, pre-relaxation, and finite-temperature MD.

# State of the field

Related approaches include (i) custom ASE scripts, (ii) direct use of model-specific APIs without ASE wrappers, and (iii) engine-specific integrations such as LAMMPS with MLIAP-style interfaces. These approaches are useful, but they usually require workflow-specific glue code and often do not preserve VASP-style on-disk artifacts.

VPMDK is designed around two comparison axes: compatibility with existing VASP-centric workflows and ease of backend switching as MLP methods evolve. This allows reuse of existing scripts and post-processing pipelines that expect VASP-like files, including workflows for structure optimization and transition-state studies. The contribution is therefore an interoperability layer, not a new potential model or a full workflow platform. Electronic-structure features (k-points, wavefunctions, charge densities) are out of scope.

![Overview of `VPMDK`](fig1.png)

# Design and implementation

**I/O compatibility.** Structures are loaded from `POSCAR` via *pymatgen* and converted to ASE `Atoms` [@Larsen2017ASE; @Ong2013pymatgen]. `POTCAR` is used only for species ordering. `WAVECAR`/`CHGCAR` are detected and ignored. `MAGMOM` parsing (including shorthand like `2*1.0`) is propagated when counts match.

**Configuration model.** A subset of `INCAR` keys controls runtime. `NSW` and `IBRION` select single-point, MD, or relaxation. `TEBEG`, `TEEND`, and `POTIM` control MD. `EDIFFG` follows VASP-like semantics (force criterion when negative, energy-change criterion when positive). `ISIF` and `PSTRESS` control cell updates during relaxation.

**Backends and runtime options.** `BCAR` (`key=value`) selects backend and optional model/checkpoint paths. VPMDK supports default-model and checkpoint-driven backends and includes practical options such as `DEVICE`, DeePMD species maps, ORB precision/compilation flags, and optional per-step energy logging [@Oba2024SevenNet; @PFPMatl2024; @OrbModels2024; @Hegde2024FAIRChem; @Choudhary2024GRACE; @Zhang2018DeePMD; @Wang2018DeePMD].

**MD and relaxation.** For `IBRION=0`, VPMDK uses velocity-Verlet with optional ASE thermostats (Andersen, Nose-Hoover chains, Langevin, and CSVR/Bussi). Relaxations use BFGS and support selected variable-cell modes through ASE filters.

**Outputs.** The tool writes `CONTCAR`, `OUTCAR`, and `XDATCAR` in VASP-like formats so existing downstream scripts can continue to operate.

# Usage

Prepare a directory containing at least `POSCAR` (optional `INCAR`, `POTCAR`, `BCAR`) and run:

```bash
vpmdk [--dir calc_dir]
```

If `--dir` is omitted, the current directory (`.`) is used. `INCAR` chooses mode and control parameters; `BCAR` selects the potential and optional `MODEL`. Results appear as `CONTCAR`, `OUTCAR`, and, for MD, `XDATCAR`.

# Research impact

The author has applied VPMDK to USPEX 9.4.4 as a drop-in VASP interface and observed faster practical turnaround for crystal-structure optimization in internal use, without modifying USPEX source code. Partial applicability has also been tested for Henkelman-group scripts that rely on VASP-style artifacts.

As of February 15, 2026, no external peer-reviewed publication citing VPMDK is available. A manuscript describing the USPEX-based application is in preparation. To the author's knowledge, the software is currently used in two research laboratories. These observations indicate interoperability value, while quantitative speed and accuracy depend on the selected backend, model, and target system.

# Limitations and scope

VPMDK is not affiliated with, endorsed by, or a replacement for VASP. It mimics selected VASP I/O conventions for workflow compatibility and supports only a subset of `INCAR` keys. Electronic-structure quantities (k-point meshes, wavefunctions, charge densities) are out of scope. Accuracy and transferability are determined by the chosen MLP and its training regime; users must validate applicability for each chemistry and condition [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024; @Oba2024SevenNet; @Hegde2024FAIRChem; @Choudhary2024GRACE].

# AI use disclosure

Generative AI tools were used extensively during development. The initial prototype (POSCAR parsing, switching among a small number of MLP backends, structural relaxation, and `CONTCAR` output) was written by a human author. Subsequent additions, including `INCAR` parsing, MD functionality, support for additional backends, and substantial test expansion, were developed with AI-assisted code generation.

All AI-assisted outputs were reviewed, corrected, and validated by the author, who takes full responsibility for the software and manuscript content.

# Acknowledgements

I thank the developers and maintainers of ASE and pymatgen for foundational infrastructure, and the authors of CHGNet, M3GNet/MatGL, MACE, and MatterSim for making high-quality MLPs broadly available.

# References
