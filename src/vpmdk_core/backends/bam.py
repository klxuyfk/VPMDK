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
    """

    root = _root()
    if root.BAMCalculator is None:
        raise RuntimeError(
            "BAM RACECalculator not available. Install bam-torch "
            "(https://github.com/myung-group/BAM-torch)."
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
