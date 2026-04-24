# Repository Guidelines

## Project Structure & Module Organization
The runtime lives in `src/vpmdk_core/__init__.py` (input parsing, backend selection, MD/relaxation flows, and VASP-style outputs). `src/vpmdk.py` is a compatibility shim, and root `vpmdk.py` preserves the legacy CLI entry path.  
Tests are in `tests/`, with longer backend-dependent checks in `tests/integration/`. Reusable fixtures and stubs are in `tests/conftest.py`.  
Long-form documentation and maintainer reference material live in `docs/`; publication assets are in `paper/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -e .`: install VPMDK in editable mode for development.
- `pytest -m "not integration"`: run fast unit/regression tests.
- `pytest -m integration`: run long-running integration tests (requires optional backends/models).
- `vpmdk --dir ./calc_dir`: run a calculation from a VASP-style input directory.
- `python -m build`: build source/wheel artifacts (install `build` first if needed).

## Coding Style & Naming Conventions
Use Python 3.10+ and follow existing project style: 4-space indentation, clear docstrings where behavior is non-obvious, and type hints for new/changed interfaces.  
Naming: `snake_case` for functions/variables, `CapWords` for classes, and `UPPER_SNAKE_CASE` for constants (for example `DEFAULT_ORB_MODEL`).  
When adding parser logic, normalize and store INCAR/BCAR tags in uppercase to match current conventions.

## Testing Guidelines
Pytest is the test framework. Name files `tests/test_*.py` and test functions `test_*`.  
Prefer deterministic unit tests with mocks/monkeypatching for calculators; reserve real backend execution for `@pytest.mark.integration` tests.  
Integration tests rely on environment variables such as `VPMDK_MACE_MODEL` or `VPMDK_FAIRCHEM_MODEL`; tests should skip cleanly when prerequisites are missing.  
Add regression tests for every bug fix; no explicit coverage gate is configured.

## Commit & Pull Request Guidelines
Match the commit style used in history: prefix subjects with types like `fix:`, `feat:`, `docs:`, `packaging:`, or `release:`. Keep one logical change per commit.  
PRs should include: a short problem/solution summary, linked issue(s), and commands used to validate changes (for example `pytest -m "not integration"`).  
For output-format changes, include representative snippets or diffs of generated files (such as `OUTCAR`, `OSZICAR`, or `vasprun.xml`).

## Security & Configuration Tips
Do not commit API keys, private model checkpoints, or proprietary datasets. Keep credentials in environment variables and reference local model paths from `BCAR`.
