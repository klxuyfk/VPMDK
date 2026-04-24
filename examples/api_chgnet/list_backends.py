from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_local_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    for candidate in (repo_root, src_path):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


_bootstrap_local_checkout()

import vpmdk


def main() -> None:
    for spec in vpmdk.list_backends():
        caps = spec.capabilities
        print(
            f"{spec.name:12s} "
            f"available={str(spec.available):5s} "
            f"default_model={str(spec.default_model):24s} "
            f"energy={caps.energy!s:5s} "
            f"forces={str(caps.forces):5s} "
            f"stress={str(caps.stress):5s} "
            f"spin={str(caps.spin):5s}"
        )


if __name__ == "__main__":
    main()
