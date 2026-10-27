"""BAM (Bayesian Atoms Modeling) backend builder."""

from __future__ import annotations

import sys
from typing import Dict


def _root():
    return sys.modules["vpmdk_core"]


def _build_bam_calculator(bcar_tags: Dict[str, str]):
    """Create a BAM RACECalculator from a local checkpoint file.

    BAM-torch publishes foundation checkpoints (e.g. the MPtrj-trained
    ``BAM-MP-core.pkl`` from Hugging Face ``myung-group/BAM_MPtrj_v1``) as
    plain ``.pkl`` files with no in-library named-model downloader, so MODEL
    is a required local path, like the NequIP/Allegro deployed-model rule.
    BAM-MP-DeNS uses this same interface but requires the upstream BAM_DeNS
    implementation; the checkpoint configuration selects the architecture.
    """

    root = _root()
    if root.BAMCalculator is None:
        raise RuntimeError(
            "BAM RACECalculator not available. Install bam-torch "
            "(https://github.com/myung-group/BAM-torch). "
            "For BAM-MP-DeNS, install the BAM_DeNS version at commit "
            "24ed0f042660a1e11ad5d91466b4275955df98c6; "
            "see docs/reference/backends.md for installation instructions."
        )

    model_reference = root._resolve_backend_model_reference(
        "BAM", bcar_tags.get("MODEL")
    )
    model_path = str(model_reference.value)
    device = root._resolve_device(bcar_tags.get("DEVICE"))
    # RACECalculator(device=None) selects cpu itself; a present-but-blank
    # DEVICE therefore lands on cpu, like the other blank-to-cpu builders.
    if device:
        return root.BAMCalculator(model=model_path, device=device)
    return root.BAMCalculator(model=model_path)
