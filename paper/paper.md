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

VPMDK (the *Vasp-Protocol Machine-learning Dynamics Kit*) is research software for running atomistic workflows with machine-learning interatomic potentials while reducing backend-specific integration work. It is organized around two connected layers. The first is an ASE-oriented Python API that absorbs practical differences among machine-learning potential packages, including calculator construction, model naming and checkpoint handling, runtime options, and selected capability metadata, while exposing a more uniform workflow around ASE `Atoms` [@Larsen2017ASE]. The second is a VASP-compatible command-line layer built on top of that API for workflows that still depend on `POSCAR`, `INCAR`, optional `POTCAR`, `BCAR`, and VASP-like outputs such as `CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`, and, for MD, `XDATCAR` [@Kresse1996Efficient]. VPMDK can also optionally write a VASP-like `CHGCAR` for the final structure by coupling the atomistic workflow to a separate machine-learning charge-density backend. The backend roster now spans more than twenty calculators, including CHGNet, M3GNet (via MatGL), MACE, MatterSim, Matlantis, Eqnorm, MatRIS, AlphaNet, HIENet, Nequix, NequIP/Allegro, SevenNet/FlashTP, ORB, UPET, TACE, DeePMD-kit, FAIRChem, and GRACE [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024; @PFPMatl2024; @MatRIS2026; @AlphaNet2025; @HIENet2025; @Nequix2025; @Batzner2022NequIP; @Musaelian2023Allegro; @Oba2024SevenNet; @OrbModels2024; @PETMAD2025; @TACE2025; @Zhang2018DeePMD; @Wang2018DeePMD; @Hegde2024FAIRChem; @Choudhary2024GRACE]. The software is distributed publicly through its PyPI release archive [@VPMDKPyPI2026].

# Statement of need

The immediate technical problem is no longer just how to call one machine-learning potential from Python. It is how to work across a growing collection of MLP packages that expose different calculator constructors, different model-selection conventions, different checkpoint formats, and different backend-specific option sets. Even when these models are ASE-compatible in principle, switching from one backend to another often requires backend-specific glue code and ad hoc workflow rewrites [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024; @Oba2024SevenNet; @Hegde2024FAIRChem; @Choudhary2024GRACE]. The target audience for VPMDK is therefore not only users of a single MLP, but researchers and workflow developers who need to evaluate, compare, or substitute multiple backends inside existing atomistic pipelines.

At the same time, many high-throughput relaxation, molecular-dynamics, and transition-state workflows in materials science are still built around VASP I/O conventions [@Kresse1996Efficient]. In practice, these workflows depend not only on structural files but also on stepwise artifacts such as `OUTCAR`, `OSZICAR`, `vasprun.xml`, and numbered image directories. Replacing the underlying force model without disturbing these surrounding workflows remains costly.

VPMDK addresses these two needs with separate but connected layers. Its core value is an API-level interoperability layer that normalizes backend selection and execution across a heterogeneous ASE ecosystem. On top of that, it provides a compatibility CLI that preserves VASP-style files and control flow for screening, pre-relaxation, finite-temperature MD, and VASP-oriented post-processing. In addition, some downstream visualization and analysis steps still depend on `CHGCAR`-like volumetric outputs; providing an optional path to generate these files from ML models removes another common reason to fall back to custom glue code or partial DFT reruns.

# State of the field

Two nearby design points help clarify the niche of VPMDK. First, LAMMPS provides a broad and mature simulation engine with machine-learning potential support, including interfaces such as MLIAP [@Thompson2022LAMMPS; @LAMMPSMLIAPDocs2026]. This is valuable when users already work in LAMMPS input conventions, but the surrounding workflow is centered on LAMMPS-specific pair styles, type mappings, and engine-side model integration rather than VASP-style artifacts such as `POSCAR`, `INCAR`, `OUTCAR`, and `vasprun.xml`. Second, VASP itself now includes on-the-fly machine-learned force fields [@Jinnouchi2019MLFF]. That capability is useful for accelerating VASP-centered simulations, but it is organized around force fields generated and consumed within VASP workflows rather than around switching among external pretrained ASE calculators.

The need for a separate interchange layer has grown as the ecosystem has diversified from a handful of universal MLPs to a heterogeneous set of invariant, equivariant, and hybrid foundation models such as MatRIS, AlphaNet, HIENet, Nequix, TACE, and PET-MAD [@MatRIS2026; @AlphaNet2025; @HIENet2025; @Nequix2025; @TACE2025; @PETMAD2025]. Contributing VASP-style compatibility logic separately inside each backend would duplicate effort and still leave users with backend-specific differences in configuration and runtime behavior. Relying on a single engine with different input conventions would not preserve the on-disk and workflow compatibility required by many VASP-centered pipelines. VPMDK is therefore an interoperability layer with two surfaces, not a new potential model or a full workflow platform. Its scholarly contribution is the combination of API-level backend substitution across a heterogeneous ASE ecosystem and a CLI layer that reuses that same normalization machinery for VASP-oriented workflows. Most electronic-structure features remain out of scope, but VPMDK now includes one deliberately narrow extension: optional post-run `CHGCAR` generation from dedicated ML charge-density predictors for visualization- and post-processing-oriented workflows.

![Overview of `VPMDK`](fig1.png)

# Software design

**API layer.** The core library operates on ASE `Atoms` and backend objects rather than on VASP files. It provides explicit calculator construction, single-point evaluation, relaxation, MD, backend discovery, and charge-density prediction, returning structured result objects without VASP-like side effects unless compatibility observers are explicitly enabled [@Larsen2017ASE]. The main design trade-off is that VPMDK standardizes only the common workflow surface, while still allowing backend-specific options to pass through when needed. This keeps the API useful for substitution across backends without pretending that all upstream models share identical semantics.

**CLI compatibility layer.** On top of that API, VPMDK provides a VASP-style command-line workflow. Structures are loaded from `POSCAR` via *pymatgen* and converted to ASE `Atoms` [@Larsen2017ASE; @Ong2013pymatgen]. `POTCAR` is used only for species ordering, while `WAVECAR` is detected and ignored. Existing `CHGCAR` inputs are not consumed by the force-field workflow, but VPMDK can optionally emit a new `CHGCAR` for the final structure. `MAGMOM` parsing, including VASP shorthand such as `2*1.0`, is propagated when counts match. Beyond final structures, the compatibility layer also emits selected stepwise artifacts, notably `OSZICAR` and a minimal `vasprun.xml`. The corresponding trade-off is intentional: VPMDK preserves compatibility with VASP-centered tooling, but it does not attempt to reimplement the full electronic-structure scope of VASP.

**Configuration model.** A subset of `INCAR` keys controls runtime. `NSW` and `IBRION` choose single-point, MD, or relaxation. Temperature, time-step, thermostat, convergence, and cell-treatment options are mapped onto ASE dynamics and relaxation drivers, including canonical sampling through velocity rescaling (CSVR/Bussi) [@Bussi2007CSVR]. NEB-style tags such as `IMAGES`, `LCLIMB`, and `SPRING` activate image-directory compatibility mode when numbered image subdirectories are present.

**Backends and runtime options.** `BCAR` (`key=value`) selects the force-field backend and resolves either named models or local checkpoints. VPMDK supports bundled/default models, automatic download helpers, and explicit checkpoint paths, with backend-specific switches for model variants, precision/compilation choices, species maps, FAIRChem v1/v2 compatibility, and optional auxiliary outputs such as per-step energy logs or LAMMPS trajectories [@Oba2024SevenNet; @PFPMatl2024; @OrbModels2024; @PETMAD2025; @Zhang2018DeePMD; @Wang2018DeePMD]. Keeping backend-specific options as an explicit escape hatch is a deliberate design decision: it avoids collapsing genuine backend differences into an overly restrictive abstraction while still giving users a uniform entry point. Charge-density prediction is configured independently: `WRITE_CHGCAR=1` requests output, while `CHARGE_MLP` and related `CHARGE_*` options select a dedicated predictor, checkpoint, and Python environment when needed. This separation allows one model to drive the atomistic workflow while another produces volumetric density on the final structure.

**MD, relaxation, and NEB-style workflows.** For `IBRION=0`, VPMDK uses velocity-Verlet with optional ASE thermostats (Andersen, Nose-Hoover chains, Langevin, and CSVR/Bussi). Relaxations use BFGS and support selected variable-cell modes through ASE filters. For numbered image directories, VPMDK runs independent per-image calculations and synthesizes parent-level `OUTCAR`, `OSZICAR`, and `vasprun.xml` summaries with VTST-like projection lines. This mode is intentionally compatibility-oriented and does not implement spring-coupled climbing-image NEB forces [@Henkelman2000CINEB].

**Charge-density export.** When `WRITE_CHGCAR=1` is set, VPMDK predicts charge density on a user-specified or INCAR-derived grid for the final structure and writes a VASP-like `CHGCAR`. The grid follows familiar `NGX*`/`PREC`/`ENCUT` semantics, but the predictor can be selected independently from the force-field backend and run in a separate Python environment. This capability is intended for visualization and post-processing compatibility: the volumetric density block is written, but full PAW augmentation data is not reconstructed.

**Outputs.** The tool writes `CONTCAR`, `OUTCAR`, `OSZICAR`, and `vasprun.xml` for all runs, plus `XDATCAR` for MD and optional `CHGCAR` when charge-density export is enabled. Optional pseudo electronic-step blocks can be emitted for legacy parsers, and lightweight auxiliary outputs such as `energy.csv` and `lammps.lammpstrj` are available for downstream monitoring.

# Usage

From the CLI side, prepare a directory containing at least `POSCAR` (optional `INCAR`, `POTCAR`, `BCAR`) and run:

```bash
vpmdk
```

VPMDK operates on the current working directory. `INCAR` chooses mode and control parameters; `BCAR` selects the potential and optional `MODEL`. Results appear as `CONTCAR`, `OUTCAR`, `OSZICAR`, and `vasprun.xml`; MD adds `XDATCAR`, and optional compatibility outputs can add `lammps.lammpstrj` or `energy.csv`.

Setting `WRITE_CHGCAR=1` in `BCAR` also requests a final-structure `CHGCAR`. Separately, the same backend-normalization layer is available through the importable Python API for users who need to embed VPMDK directly inside ASE-centered automation rather than invoking the compatibility CLI as a standalone executable.

# Research impact statement

The repository includes three concrete interoperability artifacts aimed at near-term reuse. First, `examples/neb_nequip_vtst` is a runnable example showing that the CLI layer can emit VASP-like NEB artifacts that remain usable by VTST post-processing scripts, including `nebresults.pl`, with example outputs committed in `reference/` [@VTSTTools2026]. Second, `examples/uspex_9_4_4_si` contains a USPEX 9.4.4 input deck configured to call `vpmdk` as the executable inside a VASP-oriented crystal-structure-search workflow [@Oganov2006USPEX; @Lyakhov2013USPEX]. Third, `examples/chgcar_charge3net` demonstrates optional `CHGCAR` generation through both the compatibility CLI and the public Python API, illustrating that the same backend-normalization layer can be reused from ASE-centered automation. Together these examples show realized integration into established research workflows rather than a purely aspirational software scope.

These materials do not constitute a universal performance benchmark, but they provide specific and reproducible evidence that VPMDK can reduce migration cost for three important classes of VASP-dependent external tooling: transition-state analysis pipelines built around VTST, global structure-search workflows built around USPEX, and visualization or density-consuming post-processing that expects a `CHGCAR`-like volumetric file. Quantitative speedup and accuracy still depend strongly on the selected backend, checkpoint, target chemistry, and, for `CHGCAR`, the chosen charge-density model, so the present claim is interoperability significance rather than backend-independent performance dominance.

# Limitations and scope

VPMDK is not affiliated with, endorsed by, or a replacement for VASP. It mimics selected VASP I/O conventions for workflow compatibility and supports only a subset of `INCAR` keys. Most electronic-structure quantities, including explicit k-point workflows and wavefunction handling, remain out of scope. Optional `CHGCAR` generation is approximate and post-processing oriented: VPMDK writes the volumetric density block, but does not reconstruct full PAW augmentation occupancies or provide restart-grade DFT fidelity. NEB-style execution is a compatibility layer for image directories and summary files, not a full spring-coupled path optimizer. Some backends also depend on optional upstream packages or remote services. Accuracy and transferability are determined by the chosen MLP and its training regime; users must validate applicability for each chemistry and condition [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Mattersim2024; @Oba2024SevenNet; @Hegde2024FAIRChem; @Choudhary2024GRACE; @MatRIS2026; @AlphaNet2025; @HIENet2025; @Nequix2025; @PETMAD2025].

# AI usage disclosure

OpenAI Codex 5.x was used during software development, documentation work, and manuscript preparation. Its use included code generation, refactoring assistance, test scaffolding and expansion, editorial revision of documentation, and drafting support for parts of this paper. The initial prototype, covering the basic path from VASP-style structure/control inputs to a simple `CONTCAR`-producing run, was written by the author. Subsequent additions, including support for many backend adapters, richer VASP-compatibility outputs (`OUTCAR`, `OSZICAR`, `vasprun.xml`, pseudo-SCF blocks), MD functionality, NEB-style workflow support, and substantial test expansion, were developed with AI assistance.

All AI-assisted outputs were reviewed, modified, and validated by the author, who made the primary architectural and design decisions and takes full responsibility for the software and manuscript content.

# Acknowledgements

I thank the developers and maintainers of ASE and pymatgen for foundational infrastructure, and the authors and maintainers of the supported MLP packages for making high-quality models and interfaces broadly available. This work received no dedicated external funding.

# References
