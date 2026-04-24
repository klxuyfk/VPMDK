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
date: 24 April 2026
bibliography: paper.bib
---

# Summary

VPMDK (the *VASP-Protocol Machine-learning Dynamics Kit*) is research software for using machine-learning interatomic potentials (MLPs) in atomistic workflows without rewriting each workflow around the conventions of one backend. It has two deliberately separate interfaces. The Python API is an ASE-oriented interoperability layer: it constructs calculators, runs single-point calculations, relaxations, molecular dynamics, and charge-density prediction around `ase.Atoms`, while absorbing differences in model names, checkpoint formats, device options, runtime flags, and selected capability metadata [@Larsen2017ASE]. The command-line interface is a VASP-style compatibility layer: it reads familiar `POSCAR`, `INCAR`, `POTCAR`, and `BCAR` inputs and writes VASP-like artifacts such as `CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`, `XDATCAR`, and optional `CHGCAR` [@Kresse1996Efficient]. The current backend roster spans more than twenty calculator integrations, including CHGNet, M3GNet/MatGL, MACE, SevenNet, NequIP/Allegro, DeePMD-kit, FAIRChem, MatterSim, Matlantis, ORB, UPET, TACE, MatRIS, AlphaNet, HIENet, Nequix, Eqnorm, FlashTP, and GRACE [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Oba2024SevenNet; @Batzner2022NequIP; @Musaelian2023Allegro; @Zhang2018DeePMD; @Wang2018DeePMD; @Hegde2024FAIRChem]. VPMDK is distributed through PyPI [@VPMDKPyPI2026].

# Statement of need

The immediate technical problem is no longer only how to call one MLP from Python. It is how to keep scripts and workflows stable while MLP packages evolve independently. ASE provides a common calculator protocol, but individual MLP projects still expose different constructor signatures, default models, checkpoint layouts, species mappings, device semantics, precision controls, and optional features. In practice, comparing or replacing backends often means adding local glue code around every new model. VPMDK targets researchers and workflow developers who want to substitute MLP backends in ASE-centered scripts without making each script track these backend-specific details.

The second need is operational compatibility. Many high-throughput relaxation, molecular-dynamics, transition-state, and structure-search workflows in materials science are still organized around VASP-style directories and downstream tools that inspect `OUTCAR`, `OSZICAR`, `vasprun.xml`, `XDATCAR`, numbered image directories, or `CHGCAR`-like volumetric files. Replacing the force model without disturbing those surrounding workflows remains costly. VPMDK therefore separates the core goal of backend normalization from the compatibility goal of preserving a VASP-like workflow surface.

This split is important. The API is intended to make new MLP releases replaceable behind a stable object-level interface. The CLI is intended to let established VASP-oriented automation adopt MLPs without replacing its file-level contract.

# State of the field

Several nearby tools address related but distinct problems. ASE defines the calculator abstraction used by many MLP projects [@Larsen2017ASE], but it does not by itself standardize each project's model registry, checkpoint handling, installation caveats, or optional runtime controls. LAMMPS provides a mature simulation engine with machine-learning potential support, including MLIAP, but its interface is centered on LAMMPS input conventions, pair styles, and type mappings rather than VASP-style artifacts [@Thompson2022LAMMPS; @LAMMPSMLIAPDocs2026]. VASP includes on-the-fly machine-learned force fields, but that route is organized around force fields generated and consumed within VASP workflows rather than around external pretrained ASE calculators [@Jinnouchi2019MLFF].

Workflow systems such as atomate2 now also provide force-field workflows through ASE calculators and predefined MLFF choices [@Atomate2ForcefieldsDocs2026]. VPMDK occupies a narrower layer: it is not a workflow manager, database system, or new potential model. Its contribution is an adapter surface that can sit inside scripts or workflow managers, plus a VASP-style CLI that reuses the same backend-normalization machinery. This makes VPMDK complementary to workflow engines while focusing on the integration burden created by a rapidly diversifying MLP ecosystem.

![Overview of `VPMDK`](fig1.png)

# Software design

VPMDK is implemented as a small public API, a backend registry, pure execution routines, optional observers, and a VASP-style CLI. The public API accepts `BackendConfig` objects, existing ASE calculators, and `ase.Atoms`, then returns structured result objects. It does not write `OUTCAR`, `OSZICAR`, `vasprun.xml`, or `CONTCAR` by default. This makes the API suitable for programmatic workflows where filesystem side effects would be surprising.

The backend registry translates a uniform configuration into backend-specific calculator construction. VPMDK deliberately standardizes only the common workflow surface. Backend-specific options remain available as explicit escape hatches, because real MLP packages differ in capabilities and semantics. This avoids presenting a false universal abstraction while still giving users one stable entry point for substitution.

VASP-style compatibility is implemented as an observer layer attached by the CLI or by explicit Python requests. The CLI parses a focused subset of `INCAR` and `BCAR`, maps common relaxation and MD controls onto ASE drivers, and emits compatibility outputs. `POTCAR`, `KPOINTS`, `WAVECAR`, and existing `CHGCAR` files are treated as compatibility metadata where appropriate, not as electronic-structure inputs. Numbered image directories are supported for VASP/VTST-style post-processing by evaluating images and writing aggregate summaries, but spring-coupled climbing-image NEB optimization remains out of scope [@Henkelman2000CINEB].

Charge-density export is isolated from the force-field calculation. Users can request final-structure `CHGCAR` output from a dedicated ML charge-density predictor while using a different MLP for forces and energies. The predictor can run in a separate Python environment, which keeps heavy optional dependencies out of the main runtime environment.

# Research impact statement

VPMDK is intended for computational materials researchers who need to evaluate or deploy MLPs inside existing atomistic workflows. The repository includes three concrete interoperability examples. `examples/neb_nequip_vtst` shows VASP-like NEB artifacts usable by VTST post-processing scripts such as `nebresults.pl` [@VTSTTools2026]. `examples/uspex_9_4_4_si` shows a USPEX input deck configured to call `vpmdk` as the executable in a VASP-oriented structure-search workflow [@Oganov2006USPEX; @Lyakhov2013USPEX]. `examples/chgcar_charge3net` shows optional `CHGCAR` generation through both the CLI and the Python API.

These examples do not constitute a universal benchmark. Instead, they demonstrate a narrower impact claim: VPMDK reduces integration effort for VASP-dependent tooling, backend comparison, and migration from DFT-style screening to MLP-assisted screening. The software is also used in internal workflows at two industrial research organizations and one national research laboratory. The scientific accuracy, speed, and transferability of any calculation remain determined by the selected MLP, checkpoint, target chemistry, and validation protocol.

# Limitations and scope

VPMDK is not affiliated with, endorsed by, or a replacement for VASP. It mimics selected VASP I/O conventions for workflow compatibility and supports only a subset of `INCAR` behavior. Most electronic-structure quantities, explicit k-point workflows, wavefunction handling, and restart-grade DFT semantics remain out of scope. Optional `CHGCAR` generation writes a volumetric density block for visualization and post-processing compatibility, but does not reconstruct full PAW augmentation data. NEB-style execution is a compatibility mode for image directories and summary files, not a full spring-coupled path optimizer. Some backends require optional packages, model files, remote services, or separate environments. Users must validate each chosen MLP for the relevant chemistry and thermodynamic conditions.

# AI usage disclosure

OpenAI Codex using GPT-5.3, GPT-5.4, and GPT-5.5 models was used during software development, documentation work, and manuscript preparation. Its use included code generation, refactoring assistance, test scaffolding and expansion, editorial revision of documentation, and drafting support for parts of this paper. The initial prototype, covering the path from VASP-style inputs to a simple `CONTCAR`-producing run, was written by the author. Subsequent additions, including backend adapters, VASP-like output writers, molecular dynamics, NEB-style workflow support, charge-density export, API restructuring, and test expansion, were developed with AI assistance.

All AI-assisted outputs were reviewed, modified, and validated by the author, who made the primary architectural and design decisions and takes full responsibility for the software and manuscript content.

# Acknowledgements

I thank the developers and maintainers of ASE and pymatgen for foundational infrastructure, and the authors and maintainers of the supported MLP packages for making high-quality models and interfaces broadly available. This work received no dedicated external funding.

# References
