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
    affiliation: 1
affiliations:
 - name: Corteo Co., Ltd., Japan
   index: 1
date: XX Month 2026
bibliography: paper.bib
---

# Summary

VPMDK (the *VASP-Protocol Machine-learning Dynamics Kit*) is Python software for using machine-learning interatomic potentials (MLPs) in atomistic workflows without rewriting each workflow around one backend package. It provides two deliberately separate interfaces. The Python API is an ASE-oriented interoperability layer that constructs calculators and runs single-point calculations, relaxations, molecular dynamics, and charge-density prediction around `ase.Atoms` objects [@Larsen2017ASE]. The command-line interface is a VASP-style compatibility layer: it reads VASP-style structure and control inputs such as `POSCAR` and `INCAR`, together with a VPMDK-specific backend-control file, `BCAR`, and writes VASP-like artifacts such as `CONTCAR`, `OUTCAR`, `OSZICAR`, `vasprun.xml`, `XDATCAR`, and optional `CHGCAR` [@Kresse1996Efficient]. Representative backend integrations include CHGNet, M3GNet/MatGL, MACE, SevenNet, NequIP/Allegro, DeePMD-kit, FAIRChem/UMA, MatterSim, ORB, MatRIS, AlphaNet, HIENet, Nequix, and GRACE, with additional documented adapters and charge-density predictors [@Deng2023CHGNet; @Chen2022M3GNet; @Batatia2022MACE; @Oba2024SevenNet; @Batzner2022NequIP; @Musaelian2023Allegro; @Wang2018DeePMD; @Wood2025UMA; @Mattersim2024; @OrbV3; @MatRIS2026; @AlphaNet2025; @HIENet2025; @Nequix2025; @Choudhary2024GRACE].

# Statement of need

Universal and semi-universal MLPs have matured to the point where researchers increasingly test them in routine materials workflows, but the surrounding software ecosystem remains heterogeneous. ASE provides a common calculator protocol, yet individual MLP packages still expose different constructor signatures, model registries, checkpoint formats, species mappings, device controls, precision options, and optional capabilities. Replacing one backend with another therefore often requires local glue code that is unrelated to the scientific workflow itself.

A second problem is file-level compatibility. Many relaxation, molecular-dynamics, transition-state, phonon, charge-analysis, and structure-search workflows in computational materials science are organized around VASP-style directories and downstream tools that inspect `OUTCAR`, `OSZICAR`, `vasprun.xml`, `XDATCAR`, numbered image directories, or `CHGCAR`-like volumetric files. Researchers may want to replace the force model with an MLP while preserving those workflow contracts. VPMDK addresses this need by combining backend normalization with a VASP-style command-line surface, so established VASP-oriented automation can be reused while the underlying calculator is selected through `BCAR` or the Python API.

# State of the field

Several nearby tools address related but distinct problems. ASE defines the calculator abstraction used by many atomistic codes, but it does not standardize backend-specific model registries, checkpoint handling, installation caveats, or VASP-style output artifacts [@Larsen2017ASE]. LAMMPS provides a mature simulation engine with machine-learning potential support, including MLIAP, but its interface is centered on LAMMPS input conventions, pair styles, and type mappings rather than VASP-style file contracts [@Thompson2022LAMMPS; @LAMMPSMLIAPDocs2026]. VASP includes on-the-fly machine-learned force fields, but that route is organized around force fields generated and consumed within VASP workflows rather than around external pretrained ASE calculators [@Jinnouchi2019MLFF].

Workflow and model-development tools are also complementary. atomate2 provides force-field workflows through predefined MLFF choices, wfl/ExPyRe supports flexible ASE-based MLIP fitting and simulation workflows, and the aMACEing Toolkit focuses on reproducible fine-tuning across several MLIP architectures [@Atomate2ForcefieldsDocs2026; @Gelzinyte2023WFL; @Hanseroth2025AMACEing]. VPMDK occupies a narrower adapter layer: it is not a workflow manager, database system, training framework, or new potential model. Its contribution is to make backend substitution and VASP-style workflow reuse less dependent on local, per-project compatibility code.

# Software design

VPMDK is organized as a small public API, a backend registry, pure execution routines, optional observers, and a VASP-style CLI. The public API accepts `BackendConfig` objects, existing ASE calculators, and `ase.Atoms`, then returns structured result objects. It does not write `OUTCAR`, `OSZICAR`, `vasprun.xml`, or `CONTCAR` by default, making it suitable for programmatic workflows where implicit filesystem side effects would be surprising. The CLI uses established atomistic Python infrastructure, including ASE for calculator execution and pymatgen for VASP-style input handling [@Larsen2017ASE; @Ong2013pymatgen].

The backend registry translates a uniform configuration into backend-specific calculator construction. VPMDK standardizes only the common workflow surface; backend-specific options remain available as explicit escape hatches because real MLP packages differ in capabilities and semantics. The documented registry currently covers `CHGNET`, `MACE`, `MATGL`, `M3GNET`, `MATTERSIM`, `MATRIS`, `ALPHANET`, `HIENET`, `NEQUIX`, `SEVENNET`, `FLASHTP`, `NEQUIP`, `ALLEGRO`, `ORB`, `UPET`, `TACE`, `EQUFLASH`, `FAIRCHEM`, `FAIRCHEM_V1`, `FAIRCHEM_V2`, `ESEN`, `GRACE`, `DEEPMD`, and `MATLANTIS`, with availability depending on optional packages, checkpoints, licenses, or services.

VASP-style compatibility is implemented as an observer layer attached by the CLI or by explicit Python requests. The CLI parses a focused subset of `INCAR` and `BCAR`, maps common relaxation and MD controls onto ASE drivers, and emits compatibility outputs. For VTST-style NEB layouts, VPMDK reads numbered image directories, constructs an ASE NEB band for `ICHAIN=0` workflows, applies spring-coupled band forces, supports climbing-image behavior through `LCLIMB`, and writes VASP-like image and aggregate outputs suitable for VTST-style post-processing [@Henkelman2000CINEB; @VTSTTools2026].

VPMDK also supports phonopy-oriented force-constant workflows. For VASP finite-difference phonon modes (`IBRION=5` and `6`), VPMDK obtains force constants from finite differences of the selected MLP backend's forces. For `IBRION=7` and `8`, VPMDK provides a VASP DFPT-style file interface for phonopy by serializing finite-difference MLP force constants into a VASP-like `dynmat` block in `vasprun.xml`, using the mass-normalized convention read by phonopy's VASP interface [@Togo2015Phonopy]. This preserves the `phonopy --fc vasprun.xml` workflow, while the generated force constants are derived from MLP force finite differences rather than electronic perturbation theory.

Charge-density export is isolated from the force-field calculation. Users can request final-structure `CHGCAR` output from a dedicated ML charge-density predictor while using a different MLP for forces and energies [@Koker2023ChargE3Net; @Jorgensen2020DeepDFT]. The predictor can run in a separate Python environment, keeping heavy optional dependencies out of the main runtime environment. The generated `CHGCAR` supports visualization and charge-analysis workflows, including Bader-style post-processing, but is not a reconstruction of a full VASP PAW electronic-structure calculation [@Henkelman2006Bader].

# Research impact statement

VPMDK is intended for computational materials researchers who need to evaluate or deploy MLPs inside existing atomistic workflows. Its primary impact is preserving existing toolchains while allowing the force backend to be changed. The repository includes concrete interoperability examples: VTST-style NEB post-processing with `nebresults.pl`, a USPEX input deck that calls `vpmdk` as a VASP-like executable, phonopy force-constant workflows using VASP-compatible `vasprun.xml`, and `CHGCAR` generation followed by Bader analysis [@Oganov2006USPEX; @Lyakhov2013USPEX; @VTSTTools2026; @Togo2015Phonopy; @Henkelman2006Bader].

These examples do not constitute universal MLP benchmarks. Instead, they demonstrate a practical integration claim: VPMDK reduces the integration effort required to switch calculators in VASP-oriented automation, compare pretrained backends, and move from DFT-style screening directories to MLP-assisted screening while preserving downstream file expectations. VPMDK has early users among researchers affiliated with the National Institute for Materials Science (NIMS), Japan, and industrial research organizations. Project validation notes record a May 2026 real-backend smoke sweep on a two-atom silicon structure for the registry entries where upstream calculators, checkpoints, or partner access were available, including package versions, CPU/CUDA status, checkpoint identifiers, and known caveats. This validation demonstrates runnable wrapper paths rather than benchmark-quality agreement with reference DFT data. Compatibility-output generation is covered separately by regression and integration tests. The regression suite covers the public API, input parsing, backend argument forwarding, VASP-compatible output formatting, NEB directory handling, MD driver selection, force-constant serialization, charge-density grid logic, and subprocess construction.

# Limitations and scope

VPMDK is not affiliated with, endorsed by, or a replacement for VASP. It mimics selected VASP I/O conventions for workflow compatibility and supports only a subset of `INCAR` behavior. Explicit k-point electronic workflows, wavefunction handling, restart-grade DFT semantics, dielectric tensors, Born effective charges, and most electronic-structure quantities are outside its scope. Optional `CHGCAR` generation writes a volumetric density block for visualization and post-processing compatibility, not full PAW augmentation data. Some backends require optional packages, model files, GPU stacks, remote services, licenses, or separate environments. Users must validate each chosen MLP for the relevant chemistry, thermodynamic conditions, and target property.

# AI usage disclosure

OpenAI Codex was used during software development, documentation work, and manuscript preparation. The author used the default coding models served by OpenAI Codex during the development period; stable public model-version identifiers were not consistently exposed to the author and are therefore not listed as fixed model identifiers here. AI assistance included code generation, refactoring assistance, backend-adapter implementation support, debugging, test scaffolding and expansion, editorial revision of documentation, and drafting support for parts of this paper.

The initial prototype, including the first VASP-style input path and a simple `CONTCAR`-producing run, was written by the author. The author specified the core I/O behavior, chose the VASP-style workflow focus, set the architectural direction, selected the feature-expansion priorities, reviewed and modified AI-assisted code and text, and performed the final checks. AI-assisted outputs were reviewed and modified by the author, and relevant changes were validated, where applicable, through code review, unit and regression tests, backend-dependent integration tests, package builds, example runs, manual CUDA backend validation, comparisons with expected VASP/ASE/phonopy/Bader formats, and source checking for manuscript claims and references. The author takes full responsibility for the software and manuscript content.

# Acknowledgements

The author thanks the developers and maintainers of ASE, pymatgen, phonopy, VTST tools, and the supported MLP and charge-density packages for foundational infrastructure and broadly available model interfaces. This work received no dedicated external funding.

# References
