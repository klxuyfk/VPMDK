"""Compatibility exports for the import-light VPMDK client implementation."""

from __future__ import annotations

# Keep the historical ``vpmdk_core.client`` path while the console entrypoint
# imports the top-level module without executing ``vpmdk_core.__init__``.
from vpmdk_client import (  # noqa: F401
    ClientTimeoutError,
    ProtocolError,
    RemoteBackendMismatch,
    RemoteCalculationError,
    RemoteInputError,
    ServerConnectionError,
    VPMDKClient,
    VPMDKClientError,
    _format_status,
    client_cli,
)
from vpmdk_client import socket as socket  # test/internal compatibility


__all__ = [
    "ClientTimeoutError",
    "ProtocolError",
    "RemoteBackendMismatch",
    "RemoteCalculationError",
    "RemoteInputError",
    "ServerConnectionError",
    "VPMDKClient",
    "VPMDKClientError",
    "client_cli",
]
