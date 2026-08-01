"""Import-light console entrypoint for VPMDK."""

from __future__ import annotations

import sys
from collections.abc import Sequence

# vpmdk_protocol is dependency-free (stdlib only), so reading the shared
# subcommand list here does not pull the ML runtime into the import-light entry.
from vpmdk_protocol import CLIENT_SUBCOMMANDS


def main(argv: Sequence[str] | None = None) -> int | None:
    """Dispatch client commands without importing the ML runtime."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in CLIENT_SUBCOMMANDS:
        from vpmdk_client import client_main

        return client_main(arguments)

    from vpmdk_core import main as core_main

    return core_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
