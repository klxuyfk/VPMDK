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
date: 11 April 2026
bibliography: paper.bib
---

# Summary

VPMDK (the *Vasp-Protocol Machine-learning Dynamics Kit*) is a command-line tool for running VASP-style atomistic workflows with machine-learning interatomic potentials instead of direct density-functional theory calls [@Kresse1996Efficient]. It reads familiar files such as `POSCAR`, `INCAR`, optional `POTCAR`, and `BCAR`, and writes VASP-like `CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`, and, for MD, `XDATCAR`, while delegating energy, force, and stress evaluation to ASE-compatible calculators [@Larsen2017ASE]. The backend roster now spans more than twenty calculators, including CHGNet, M3GNet (via MatGL), MACE, MatterSim, Matlantis, Eqnorm, MatRIS, AlphaNet, HIENet, Nequix, NequIP/Allegro, SevenNet/FlashTP, ORB, UPET, TACE, DeePMD-kit, FAIRChem, and GRACE [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024; @PFPMatl2024; @MatRIS2026; @AlphaNet2025; @HIENet2025; @Nequix2025; @Batzner2022NequIP; @Musaelian2023Allegro; @Oba2024SevenNet; @OrbModels2024; @PETMAD2025; @TACE2025; @Zhang2018DeePMD; @Wang2018DeePMD; @Hegde2024FAIRChem; @Choudhary2024GRACE]. The software is distributed publicly through its PyPI release archive [@VPMDKPyPI2026].

# Statement of need

Many high-throughput relaxation, molecular-dynamics, and transition-state workflows in materials science are built around VASP I/O conventions [@Kresse1996Efficient]. In practice, these workflows often depend not only on structural files but also on stepwise artifacts such as `OUTCAR`, `OSZICAR`, `vasprun.xml`, and numbered image directories. At the same time, modern MLPs provide practical speedups for many workloads [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024; @Oba2024SevenNet; @Hegde2024FAIRChem; @Choudhary2024GRACE]. Adopting them often means rewriting workflows around custom Python scripts or backend-specific wrappers.

VPMDK targets this migration cost. It keeps VASP-style files and control flow while enabling users to switch among multiple MLP backends for screening, pre-relaxation, finite-temperature MD, and VASP-oriented post-processing.

# State of the field

Two nearby design points help clarify the niche of VPMDK. First, LAMMPS provides a broad and mature simulation engine with machine-learning potential support, including interfaces such as MLIAP [@Thompson2022LAMMPS; @LAMMPSMLIAPDocs2026]. This is valuable when users already work in LAMMPS input conventions, but the surrounding workflow is centered on LAMMPS-specific pair styles, type mappings, and engine-side model integration rather than VASP-style artifacts such as `POSCAR`, `INCAR`, `OUTCAR`, and `vasprun.xml`. Second, VASP itself now includes on-the-fly machine-learned force fields [@Jinnouchi2019MLFF]. That capability is useful for accelerating VASP-centered simulations, but it is organized around force fields generated and consumed within VASP workflows rather than around switching among external pretrained ASE calculators.

The need for a separate interchange layer has grown as the ecosystem has diversified from a handful of universal MLPs to a heterogeneous set of invariant, equivariant, and hybrid foundation models such as MatRIS, AlphaNet, HIENet, Nequix, TACE, and PET-MAD [@MatRIS2026; @AlphaNet2025; @HIENet2025; @Nequix2025; @TACE2025; @PETMAD2025]. Implementing VASP-style compatibility separately inside each backend, or inside a single engine with different input conventions, would not solve the combination of requirements addressed here: VASP-like on-disk compatibility, rapid backend substitution across a heterogeneous ASE ecosystem, and lightweight command-line use inside existing workflow managers. VPMDK is therefore an interoperability layer, not a new potential model or a full workflow platform. Electronic-structure features (k-points, wavefunctions, charge densities) are out of scope.

![Overview of `VPMDK`](fig1.png)

# Software design

**I/O compatibility.** Structures are loaded from `POSCAR` via *pymatgen* and converted to ASE `Atoms` [@Larsen2017ASE; @Ong2013pymatgen]. `POTCAR` is used only for species ordering, while `WAVECAR` and `CHGCAR` are detected and ignored. `MAGMOM` parsing, including VASP shorthand such as `2*1.0`, is propagated when counts match. Beyond final structures, the compatibility layer also emits selected stepwise artifacts, notably `OSZICAR` and a minimal `vasprun.xml`.

**Configuration model.** A subset of `INCAR` keys controls runtime. `NSW` and `IBRION` choose single-point, MD, or relaxation. Temperature, time-step, thermostat, convergence, and cell-treatment options are mapped onto ASE dynamics and relaxation drivers, including canonical sampling through velocity rescaling (CSVR/Bussi) [@Bussi2007CSVR]. NEB-style tags such as `IMAGES`, `LCLIMB`, and `SPRING` activate image-directory compatibility mode when numbered image subdirectories are present.

**Backends and runtime options.** `BCAR` (`key=value`) selects the backend and resolves either named models or local checkpoints. VPMDK supports bundled/default models, automatic download helpers, and explicit checkpoint paths, with backend-specific switches for model variants, precision/compilation choices, species maps, FAIRChem v1/v2 compatibility, and optional auxiliary outputs such as per-step energy logs or LAMMPS trajectories [@Oba2024SevenNet; @PFPMatl2024; @OrbModels2024; @PETMAD2025; @Zhang2018DeePMD; @Wang2018DeePMD].

**MD, relaxation, and NEB-style workflows.** For `IBRION=0`, VPMDK uses velocity-Verlet with optional ASE thermostats (Andersen, Nose-Hoover chains, Langevin, and CSVR/Bussi). Relaxations use BFGS and support selected variable-cell modes through ASE filters. For numbered image directories, VPMDK runs independent per-image calculations and synthesizes parent-level `OUTCAR`, `OSZICAR`, and `vasprun.xml` summaries with VTST-like projection lines. This mode is intentionally compatibility-oriented and does not implement spring-coupled climbing-image NEB forces [@Henkelman2000CINEB].

**Outputs.** The tool writes `CONTCAR`, `OUTCAR`, `OSZICAR`, and `vasprun.xml` for all runs, plus `XDATCAR` for MD. Optional pseudo electronic-step blocks can be emitted for legacy parsers, and lightweight auxiliary outputs such as `energy.csv` and `lammps.lammpstrj` are available for downstream monitoring.

# Usage

Prepare a directory containing at least `POSCAR` (optional `INCAR`, `POTCAR`, `BCAR`) and run:

```bash
vpmdk
```

VPMDK operates on the current working directory. `INCAR` chooses mode and control parameters; `BCAR` selects the potential and optional `MODEL`. Results appear as `CONTCAR`, `OUTCAR`, `OSZICAR`, and `vasprun.xml`; MD adds `XDATCAR`, and optional compatibility outputs can add `lammps.lammpstrj` or `energy.csv`.

# Research impact statement

The repository includes two concrete interoperability artifacts aimed at near-term reuse. First, `examples/neb_nequip_vtst` is a runnable example showing that VPMDK can emit VASP-like NEB artifacts that remain usable by VTST post-processing scripts, including `nebresults.pl`, with example outputs committed in `reference/` [@VTSTTools2026]. Second, `examples/uspex_9_4_4_si` contains a USPEX 9.4.4 input deck configured to call `vpmdk` as the executable inside a VASP-oriented crystal-structure-search workflow [@Oganov2006USPEX; @Lyakhov2013USPEX].

These materials do not constitute a universal performance benchmark, but they provide specific and reproducible evidence that VPMDK can reduce migration cost for two important classes of VASP-dependent external tooling: transition-state analysis pipelines built around VTST and global structure-search workflows built around USPEX. Quantitative speedup and accuracy still depend strongly on the selected backend, checkpoint, and target chemistry, so the present claim is interoperability significance rather than backend-independent performance dominance.

# Limitations and scope

VPMDK is not affiliated with, endorsed by, or a replacement for VASP. It mimics selected VASP I/O conventions for workflow compatibility and supports only a subset of `INCAR` keys. Electronic-structure quantities (k-point meshes, wavefunctions, charge densities) are out of scope. NEB-style execution is a compatibility layer for image directories and summary files, not a full spring-coupled path optimizer. Some backends also depend on optional upstream packages or remote services. Accuracy and transferability are determined by the chosen MLP and its training regime; users must validate applicability for each chemistry and condition [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024; @Oba2024SevenNet; @Hegde2024FAIRChem; @Choudhary2024GRACE; @MatRIS2026; @AlphaNet2025; @HIENet2025; @Nequix2025; @PETMAD2025].

# AI usage disclosure

OpenAI Codex 5.x was used extensively during development, primarily for code generation, refactoring, test expansion, and some documentation and paper drafting. The initial prototype, covering the basic path from VASP-style structure/control inputs to a simple `CONTCAR`-producing run, was written by the author. Subsequent additions, including support for many backend adapters, richer VASP-compatibility outputs (`OUTCAR`, `OSZICAR`, `vasprun.xml`, pseudo-SCF blocks), MD functionality, NEB-style workflow support, and substantial test expansion, were developed with AI assistance.

All AI-assisted outputs were reviewed, corrected, and validated by the author, who takes full responsibility for the software and manuscript content.

# Acknowledgements

I thank the developers and maintainers of ASE and pymatgen for foundational infrastructure, and the authors and maintainers of the supported MLP packages for making high-quality models and interfaces broadly available. This work received no dedicated external funding.

# References
