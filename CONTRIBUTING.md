# Contributing to VPMDK

Thanks for your interest in improving VPMDK! We welcome bug reports, feature proposals, documentation updates, and other contributions. This guide explains how to get started.

## Code of conduct

Please be respectful and constructive when interacting with the community. Be kind, assume good intent, and keep discussions focused on the technical topic at hand.

## Getting started

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/vpmdk.git
   cd vpmdk
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```
3. Install the core dependencies listed in the README:
   ```bash
   pip install ase pymatgen
   # Install optional back-ends such as chgnet, mattersim, mace-torch, or matgl as needed
   ```

## Development workflow

1. Create a branch for your work:
   ```bash
   git checkout -b feature/my-change
   ```
2. Make your changes in small, focused commits. Include clear commit messages that describe the intent of each change.
3. Add tests whenever you fix a bug or implement a new feature.
4. Ensure the existing tests continue to pass:
   ```bash
   pytest
   ```
5. Run static analysis tools you rely on (e.g. `ruff`, `black`, `mypy`) and ensure your code follows PEP 8 style. Keep imports sorted and add type hints when they improve readability.
6. Update documentation (README, docstrings, comments) to reflect behavioural changes or new features.

## Writing good bug reports

When filing an issue, please include:

- The VPMDK version or commit hash.
- Your operating system and Python version.
- The potential and calculator you used (CHGNet, MACE, MatGL, MatterSim, etc.).
- Steps to reproduce the problem, along with sample input files if possible.
- The full error message or unexpected output.

## Submitting a pull request

Before opening a pull request (PR):

- Rebase your branch onto the latest `main` branch.
- Double-check that tests pass (`pytest`) and that you ran code formatters/linters.
- Confirm that any new dependencies are documented and optional.
- If your change affects user-facing behaviour, add or update documentation and examples.

When you open the PR:

- Provide a clear description of the change and its motivation.
- Reference any related issues (e.g. "Fixes #123").
- Include screenshots or logs when they help reviewers understand the change.
- Be ready to respond to review feedback and iterate on your branch.

## Licensing

By contributing to VPMDK you agree that your contributions will be licensed under the repository's BSD 3-Clause License.

We appreciate your help in making VPMDK better!
