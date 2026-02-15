# JOSS Policy Gap Memo (2026-02-15)

This note summarizes likely review risks for the current JOSS submission draft and repository state.

## Findings (highest priority first)

1. Critical: The six-month public-development requirement is likely not yet satisfied.
- Public repository creation date appears to be 2025-10-17.
- As of 2026-02-15, this is about four months.
- If interpreted strictly, submission is safer on or after 2026-04-17.

2. Major: `Research impact` is still weak on externally verifiable evidence.
- The current text relies on internal observations and in-preparation work.
- Reviewers may ask for reproducible public evidence (benchmark protocol, input conditions, public artifacts).

3. Major: `AI use disclosure` lacks explicit tool/model/version details.
- Policy text expects concrete disclosure of what tools were used and where.

4. Major: `State of the field` is currently abstract.
- Comparative discussion exists, but named tools plus explicit citations/tables are limited.

5. Major: Build-vs-contribute rationale is still thin.
- The paper should explain why a separate interoperability layer was preferable to contributing directly to existing tools.

6. Major: Test execution can fail in some environments.
- `python -m pytest -q` produced import errors related to `tests.conftest` in this environment.
- This can weaken reviewer confidence in reproducibility.

7. Medium: Heading-name strictness risk (inference).
- JOSS examples mention `Software design`; the paper currently uses `Design and implementation`.
- Functionally similar, but format-sensitive reviewers may comment.

8. Medium: Citation coverage for named workflow tools may be insufficient.
- Mentions (e.g., USPEX, Henkelman scripts, LAMMPS/MLIAP) should ideally have citations where possible.

9. Medium: Software-archive citation (e.g., DOI) should be checked.
- Ensure the paper bibliography includes a software archive reference for this release if available.

## Open questions to resolve before final submission

1. Can AI disclosure include concrete tool/model/version names?
2. Can at least one USPEX-related benchmark be made publicly reproducible?
3. Can external usage claims be backed by citable public artifacts?
4. Is submission timing shifted to meet the six-month public-history criterion?

## Reference links

- https://joss.readthedocs.io/en/latest/paper.html
- https://joss.readthedocs.io/en/latest/review_criteria.html
- https://joss.readthedocs.io/en/latest/submitting.html
- https://joss.readthedocs.io/en/latest/policies.html
- https://blog.joss.theoj.org/2026/01/preparing-joss-for-a-generative-ai-future
- https://api.github.com/repos/klxuyfk/VPMDK
