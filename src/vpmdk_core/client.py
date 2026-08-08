"""Compatibility exports for the import-light VPMDK client implementation."""

from __future__ import annotations

# Re-export the import-light client through the vpmdk_core.client compatibility path.
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
