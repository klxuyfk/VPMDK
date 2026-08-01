"""Lightweight constants and path helpers shared by server-mode peers."""

from __future__ import annotations

import os


PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 1024 * 1024

# Single source of truth for the client subcommand names. The import-light entry
# (vpmdk_entry) and the full CLI (vpmdk_core.cli) both route on this, and
# add_client_subcommands registers exactly these parsers, so a new client
# subcommand is added in one place. Kept here in the dependency-free protocol
# module so the import-light entry can read it without pulling in the client.
CLIENT_SUBCOMMANDS = ("run", "status", "stop")


def default_socket_path() -> str:
    """Return the per-user default Unix socket path."""

    runtime_root = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    uid = os.getuid() if hasattr(os, "getuid") else "unsupported"
    return os.path.join(runtime_root, f"vpmdk-{uid}", "default.sock")


def resolve_socket_path(explicit: str | None = None) -> str:
    """Resolve CLI flag, environment variable, then default socket path."""

    value = explicit or os.environ.get("VPMDK_SOCKET") or default_socket_path()
    try:
        return os.path.abspath(os.path.expanduser(value))
    except OSError as exc:
        # abspath of a RELATIVE path consults os.getcwd(), which raises
        # FileNotFoundError when the shell's directory has been deleted --
        # and this runs while constructing VPMDKClient, OUTSIDE client_cli's
        # exception mapping, so run/status/stop all died with a raw traceback
        # at interpreter-default exit 1 (off-contract for status's 0/3 and
        # stop's 0/3/4). ValueError is the type client_cli maps to a clean
        # one-line diagnostic, matching the deleted-cwd handling of the
        # request builder.
        raise ValueError(
            f"cannot resolve the relative socket path {value!r}: the current "
            "working directory no longer exists; cd to an existing directory "
            "or use an absolute --socket/VPMDK_SOCKET"
        ) from exc
