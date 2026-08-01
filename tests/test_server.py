from __future__ import annotations

import contextlib
import errno
import io
import json
import logging
import queue
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import vpmdk
import vpmdk_client as lightweight_client_module
import vpmdk_core.client as client_module
import vpmdk_core.server as server_module
from tests.conftest import DummyCalculator
from vpmdk_core.client import (
    ClientTimeoutError,
    RemoteBackendMismatch,
    RemoteCalculationError,
    RemoteInputError,
    ServerConnectionError,
    VPMDKClient,
)
from vpmdk_core.server import (
    BackendConfigurationMismatch,
    ServerAlreadyRunning,
    VPMDKServer,
    _load_backend_for_server,
    backend_identity,
    default_socket_path,
    pidfile_path,
    prepare_socket_path,
    validate_request_backend,
)


def _wait_for(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true before timeout")


def _start_server(
    socket_path: Path,
    *,
    calculator=None,
    tags: dict[str, str] | None = None,
    idle_timeout: float = 0.0,
    heartbeat_interval: float = 0.05,
    pidfile: Path | None = None,
    executor=None,
) -> tuple[VPMDKServer, threading.Thread]:
    server = VPMDKServer(
        str(socket_path),
        calculator or DummyCalculator(),
        tags or {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(socket_path.parent),
        idle_timeout=idle_timeout,
        heartbeat_interval=heartbeat_interval,
        pidfile=str(pidfile) if pidfile is not None else None,
        executor=executor,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # serve_forever publishes self._listener only AFTER bind() creates the
    # socket file, so waiting on the file alone leaves a window in which a
    # connect gets ECONNREFUSED.
    _wait_for(lambda: socket_path.exists() and server._listener is not None)

    def responds() -> bool:
        try:
            return VPMDKClient(str(socket_path)).status(timeout=0.2)["protocol"] == 1
        except (ServerConnectionError, ClientTimeoutError):
            return False

    _wait_for(responds)
    return server, thread


def _stop_server(socket_path: Path, thread: threading.Thread) -> None:
    if socket_path.exists():
        VPMDKClient(str(socket_path)).stop(timeout=3.0)
    thread.join(timeout=3.0)
    assert not thread.is_alive()


def test_legacy_cli_dispatch_is_unchanged(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(vpmdk, "run_workdir", lambda workdir: calls.append(workdir))

    assert vpmdk.main([]) is None
    assert vpmdk.main(["--dir", "run"]) is None
    assert calls == [".", "run"]


def test_legacy_help_output_is_byte_for_byte_unchanged(capsys):
    # SERVER_MODE_SPEC 1.1 (non-negotiable): the behavior of `vpmdk` and
    # `vpmdk --dir DIR` must not change by a single byte. `--help` is not a
    # subcommand, so it dispatches to the legacy parser -- its stdout is part of
    # that contract. Locking it against an independently reconstructed pre-server
    # parser catches any description/argument/epilog drift (e.g. advertising the
    # server subcommands here, which is what this test was added to prevent).
    import argparse

    expected_parser = argparse.ArgumentParser(
        prog="vpmdk", description="Run MLP with VASP style inputs"
    )
    expected_parser.add_argument("--dir", default=".", help="Input directory")
    expected = expected_parser.format_help()

    original_argv = sys.argv
    try:
        sys.argv = ["vpmdk", "--help"]
        with pytest.raises(SystemExit) as legacy_exit:
            vpmdk.main(["--help"])
    finally:
        sys.argv = original_argv
    assert legacy_exit.value.code == 0
    assert capsys.readouterr().out == expected

    # Subcommand discovery belongs to the server parser, not the legacy one.
    assert "serve" not in expected


def test_cli_help_discovers_server_commands_and_timeout_semantics(capsys):
    with pytest.raises(SystemExit) as run_exit:
        vpmdk.main(["run", "--help"])
    assert run_exit.value.code == 0
    assert "0 waits indefinitely" in capsys.readouterr().out

    with pytest.raises(SystemExit) as stop_exit:
        vpmdk.main(["stop", "--help"])
    assert stop_exit.value.code == 0
    assert "0 does not wait" in capsys.readouterr().out


def test_server_output_matches_one_shot(
    tmp_path: Path,
    prepare_inputs,
    monkeypatch,
):
    one_shot = tmp_path / "one-shot"
    resident = tmp_path / "resident"
    one_shot.mkdir()
    resident.mkdir()
    prepare_inputs(one_shot, incar_overrides={"NSW": "0"})
    prepare_inputs(resident, incar_overrides={"NSW": "0"})
    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda *args, **kwargs: DummyCalculator(),
    )

    vpmdk.run_workdir(str(one_shot))
    socket_path = tmp_path / "server.sock"
    _, thread = _start_server(socket_path)
    lines: list[str] = []
    try:
        VPMDKClient(str(socket_path)).run(str(resident), log_callback=lines.append)
        status = VPMDKClient(str(socket_path)).status()
        assert status["jobs_completed"] == 1
        assert status["jobs_failed"] == 0
    finally:
        _stop_server(socket_path, thread)

    assert "Calculation completed." in lines
    for filename in ("OUTCAR", "OSZICAR", "CONTCAR", "vasprun.xml"):
        resident_bytes = (resident / filename).read_bytes()
        one_shot_bytes = (one_shot / filename).read_bytes()
        if filename == "OUTCAR":
            # The existing compatibility footer intentionally records live
            # process timing/resource counters, which vary between any two runs.
            marker = b" General timing and accounting informations for this job:\n"
            resident_bytes = resident_bytes.split(marker, 1)[0]
            one_shot_bytes = one_shot_bytes.split(marker, 1)[0]
        assert resident_bytes == one_shot_bytes


def test_server_lifecycle_status_and_cleanup(tmp_path: Path):
    socket_path = tmp_path / "server.sock"

    def executor(workdir: str, *, calculator) -> None:
        print("Calculation completed.")

    _, thread = _start_server(socket_path, executor=executor)
    client = VPMDKClient(str(socket_path))
    assert client.status()["state"] == "idle"
    assert client.status()["jobs_completed"] == 0
    client.run(str(tmp_path))
    status = client.status()
    assert status["state"] == "idle"
    assert status["jobs_completed"] == 1
    client.stop(timeout=3.0)
    thread.join(timeout=3.0)
    assert not socket_path.exists()
    assert not thread.is_alive()


def test_server_passes_client_cwd_and_one_bcar_snapshot(
    tmp_path: Path, monkeypatch
):
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    (request_dir / "BCAR").write_text("MLP=CHGNET\nDEVICE=cpu\n")
    parse_calls = 0
    original_parse = vpmdk.parse_key_value_file
    captured: dict[str, object] = {}

    def parse_once(path: str, **kwargs):
        # **kwargs: the server's hoisted parse passes warn_unknown_tags=False
        # (R162), which this counting stub must accept like the real function.
        nonlocal parse_calls
        parse_calls += 1
        return original_parse(path, **kwargs)

    def fake_run_workdir(
        workdir: str,
        *,
        calculator,
        bcar_tags=None,
        charge_base_dir=None,
        backend_tags=None,
    ) -> None:
        captured["workdir"] = workdir
        captured["bcar_tags"] = dict(bcar_tags or {})
        captured["charge_base_dir"] = charge_base_dir
        captured["backend_tags"] = dict(backend_tags or {})

    monkeypatch.setattr(vpmdk, "parse_key_value_file", parse_once)
    monkeypatch.setattr(vpmdk, "run_workdir", fake_run_workdir)
    _, thread = _start_server(socket_path)
    try:
        VPMDKClient(str(socket_path)).run(str(request_dir))
    finally:
        _stop_server(socket_path, thread)

    assert parse_calls == 1
    assert captured["workdir"] == str(request_dir)
    assert captured["bcar_tags"] == {"MLP": "CHGNET", "DEVICE": "cpu"}
    assert captured["charge_base_dir"] == os.getcwd()
    # R132: the RESIDENT's effective configuration travels with the request, so
    # the capability gate resolves the backend that will actually compute even
    # when the request inherits its tags (SPEC 3.4) instead of restating them.
    # It is a capability fallback only -- bcar_tags above is unchanged.
    assert captured["backend_tags"]["MLP"] == "CHGNET"
    assert captured["backend_tags"]["DEVICE"] == "cpu"


def test_server_input_error_maps_to_remote_input_and_cli_exit_one(
    tmp_path: Path, capsys
):
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "missing-poscar"
    request_dir.mkdir()
    _, thread = _start_server(socket_path)
    try:
        with pytest.raises(RemoteInputError, match="POSCAR not found"):
            VPMDKClient(str(socket_path)).run(str(request_dir))
        args = lightweight_client_module.argparse.Namespace(
            command="run",
            dir=str(request_dir),
            socket=str(socket_path),
            timeout=0.0,
        )
        assert lightweight_client_module.client_cli(args) == 1
    finally:
        _stop_server(socket_path, thread)

    captured = capsys.readouterr()
    assert "POSCAR not found." in captured.out
    assert "POSCAR not found" in captured.err


def test_unsupported_config_notimplemented_maps_to_input_error(tmp_path: Path):
    # An unsupported INCAR config/mode (e.g. VTST ICHAIN!=0) raises
    # NotImplementedError. That is a permanent fix-your-input condition -- not a
    # retryable calculation failure -- and one-shot exits 1 on it, so the server
    # must classify it as input_error (RemoteInputError / exit 1), not
    # calculation_error (exit 2), keeping the two paths and the retry-key
    # exit-code contract consistent.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    def executor(workdir: str, *, calculator) -> None:
        # The dedicated type VPMDK itself raises for this condition. Using the
        # BUILTIN NotImplementedError here would no longer be faithful: that one
        # is also what a third-party backend raises mid-calculation (torch's
        # unregistered-kernel error), which is a calculation failure, so the
        # classification keys on the type rather than on the base class.
        raise vpmdk.UnsupportedInputError(
            "VPMDK currently implements VTST-style NEB for ICHAIN=0 only."
        )

    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteInputError, match="ICHAIN=0 only"):
            VPMDKClient(str(socket_path)).run(str(request_dir))
    finally:
        _stop_server(socket_path, thread)


def test_backend_notimplemented_maps_to_calculation_error(tmp_path: Path):
    # The complement of the test above: a plain NotImplementedError raised
    # MID-CALCULATION by a third-party backend (torch's "Could not run
    # 'aten::...' with arguments from the 'CUDA' backend") is an exception DURING
    # the calculation, which SERVER_MODE_SPEC 2.5 defines as exit 2 -- and only
    # that branch prints the traceback the user needs to diagnose it.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    def executor(workdir: str, *, calculator) -> None:
        raise NotImplementedError(
            "Could not run 'aten::empty_strided' with arguments from the 'CUDA' backend"
        )

    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteCalculationError, match="aten::empty_strided") as excinfo:
            VPMDKClient(str(socket_path)).run(str(request_dir))
        assert not isinstance(excinfo.value, RemoteInputError)
        assert excinfo.value.traceback  # the diagnostic exit-1 would have dropped
    finally:
        _stop_server(socket_path, thread)


def test_ase_property_not_implemented_maps_to_calculation_error(tmp_path: Path):
    # ASE's PropertyNotImplementedError (a NotImplementedError SUBCLASS) is raised
    # mid-calculation by a calculator lacking a requested property (e.g.
    # get_stress on a forces-only model during ISIF>=3). That is a genuine
    # calculation failure (calculation_error / exit 2, with traceback), NOT the
    # unsupported-input-config case VPMDK's plain NotImplementedError signals, so
    # it must not be folded into input_error by the over-broad base-class check.
    from ase.calculators.calculator import PropertyNotImplementedError

    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    def executor(workdir: str, *, calculator) -> None:
        raise PropertyNotImplementedError("stress not implemented")

    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteCalculationError, match="stress not implemented"):
            VPMDKClient(str(socket_path)).run(str(request_dir))
    finally:
        _stop_server(socket_path, thread)


@pytest.mark.parametrize(
    "exc",
    [
        PermissionError(13, "Permission denied", "OUTCAR"),
        IsADirectoryError(21, "Is a directory", "OUTCAR"),
        NotADirectoryError(20, "Not a directory", "OUTCAR/x"),
    ],
    ids=["permission", "is-a-directory", "not-a-directory"],
)
def test_unwritable_workdir_maps_to_input_error(tmp_path: Path, exc):
    # An OUTCAR that is a directory, or a read-only workdir tree, is a
    # deterministic property of the submitted workdir: a retry reproduces it
    # byte-for-byte. Classifying it exit 2 ("retryable calculation failure")
    # sent batch drivers into a retry loop that can never succeed, and the
    # one-shot CLI dies with an uncaught OSError (exit 1) on the same tree, so
    # the server must answer input_error / exit 1 to stay in agreement.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    def executor(workdir: str, *, calculator) -> None:
        raise exc

    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteInputError):
            VPMDKClient(str(socket_path)).run(str(request_dir))
    finally:
        _stop_server(socket_path, thread)


def test_generic_oserror_stays_calculation_error(tmp_path: Path):
    # The complement: OSErrors that CAN clear up between attempts (disk full,
    # a network filesystem flapping) keep the documented retryable exit 2.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    def executor(workdir: str, *, calculator) -> None:
        raise OSError(28, "No space left on device", "OUTCAR")

    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteCalculationError, match="No space left") as excinfo:
            VPMDKClient(str(socket_path)).run(str(request_dir))
        assert not isinstance(excinfo.value, RemoteInputError)
    finally:
        _stop_server(socket_path, thread)


def test_readonly_filesystem_oserror_is_an_input_error(tmp_path: Path):
    # R150 (P2): Python has no OSError subclass for EROFS, so a read-only
    # MOUNT (read-only NFS export, remount,ro, container image layer) fell
    # past the R135 (PermissionError, ...) tuple into calculation_error /
    # exit 2 -- documented RETRYABLE for a condition a retry can never fix --
    # while permission-bit read-only-ness (EACCES -> PermissionError), the
    # docs, and the one-shot CLI all say exit 1.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    def executor(workdir: str, *, calculator) -> None:
        raise OSError(errno.EROFS, "Read-only file system", "OUTCAR")

    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteInputError, match="Read-only file system"):
            VPMDKClient(str(socket_path)).run(str(request_dir))
    finally:
        _stop_server(socket_path, thread)


def test_unwritable_foreground_log_file_is_rejected_before_the_model_load(
    tmp_path: Path, capsys, monkeypatch
):
    # R150 (P3): the daemon path opens an explicit --log-file in _daemonize
    # BEFORE any backend work, but the foreground path deferred the identical
    # open to VPMDKServer.__init__ -- after the full model load -- so an
    # unwritable path cost a complete checkpoint load per retry, and a FIFO
    # at the path blocked the post-load open forever with no socket bound and
    # no diagnostic.
    if os.geteuid() == 0:
        pytest.skip("directory permissions do not bind as root")

    def failing_loader(workdir, bcar):
        raise AssertionError("the model must not be loaded for an unusable log file")

    monkeypatch.setattr(server_module, "_load_backend_for_server", failing_loader)

    logdir = tmp_path / "logs"
    logdir.mkdir()
    logdir.chmod(0o500)
    args = SimpleNamespace(
        command="serve",
        dir=str(tmp_path),
        bcar=None,
        socket=str(tmp_path / "s.sock"),
        idle_timeout=0.0,
        daemon=False,
        log_file=str(logdir / "server.log"),
        daemon_notify_fd=None,
    )
    try:
        assert server_module.serve_cli(args) == 1
    finally:
        logdir.chmod(0o700)
    err = capsys.readouterr().err
    assert "unable to open the log file" in err
    assert "server.log" in err

    # A reader-less FIFO errors immediately (O_NONBLOCK -> ENXIO) instead of
    # hanging the foreground serve forever.
    fifo_path = tmp_path / "fifo.log"
    os.mkfifo(fifo_path)
    args.log_file = str(fifo_path)
    started = time.monotonic()
    assert server_module.serve_cli(args) == 1
    assert time.monotonic() - started < 5.0
    assert "unable to open the log file" in capsys.readouterr().err


def test_device_detection_builds_backend_identity_once(tmp_path: Path, monkeypatch):
    calls: list[dict[str, object]] = []
    original_identity = server_module.backend_identity

    def recording_identity(tags, *, base_dir):
        calls.append(dict(tags))
        return original_identity(tags, base_dir=base_dir)

    calculator = DummyCalculator()
    calculator.device = "cpu"
    monkeypatch.setattr(server_module, "backend_identity", recording_identity)
    server = VPMDKServer(
        str(tmp_path / "server.sock"),
        calculator,
        {"MLP": "CHGNET"},
        backend_base_dir=str(tmp_path),
    )

    assert calls == [{"MLP": "CHGNET", "DEVICE": "cpu"}]
    assert server.backend["device"] == "cpu"
    server._cleanup()


def test_backend_identity_canonicalizes_explicit_configuration_once(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "weights.pt").write_text("placeholder")
    calls = 0
    original_canonical = server_module._canonical_configuration

    def recording_canonical(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_canonical(*args, **kwargs)

    monkeypatch.setattr(
        server_module, "_canonical_configuration", recording_canonical
    )

    identity = backend_identity(
        {"MLP": "MACE", "MODEL": "weights.pt", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    assert calls == 1
    assert identity["configuration"]["MODEL"] == str(tmp_path / "weights.pt")


def test_server_reuses_shared_model_path_suffixes():
    # The server must not keep its own copy of the extension lists; it shares the
    # backend_common constants so the path-shape heuristics cannot drift apart.
    assert not hasattr(server_module, "_MODEL_PATH_SUFFIXES")
    assert ".pt" in vpmdk._MODEL_PATH_SUFFIXES
    # Config-file extensions shape config paths but not MODEL identities.
    assert ".yaml" not in vpmdk._MODEL_PATH_SUFFIXES
    assert ".yaml" in vpmdk._CONFIG_PATH_SUFFIXES
    assert set(vpmdk._MODEL_PATH_SUFFIXES).issubset(vpmdk._CONFIG_PATH_SUFFIXES)


def test_backend_identity_resolves_model_reference_once(
    tmp_path: Path, monkeypatch
):
    # backend_identity must not resolve MODEL twice: _canonical_configuration
    # already computes the identity, so the model field reuses it instead of
    # repeating the filesystem resolution.
    (tmp_path / "weights.pt").write_text("placeholder")
    calls = 0
    original = vpmdk._resolve_backend_model_reference

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(vpmdk, "_resolve_backend_model_reference", counting)

    identity = backend_identity(
        {"MLP": "MACE", "MODEL": "weights.pt", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    assert identity["model"] == str(tmp_path / "weights.pt")
    assert calls == 1


@pytest.mark.parametrize("backend", ["MATTERSIM", "MACE", "ORB", "CHGNET"])
def test_backend_identity_treats_empty_model_tag_as_omitted(
    backend: str, tmp_path: Path, monkeypatch
):
    # A present-but-empty ``MODEL=`` tag means "use the backend default", exactly
    # as the direct build path treats it. Backends without a default model must
    # not crash resident-server startup on such a tag.
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)

    identity = backend_identity(
        {"MLP": backend, "MODEL": "", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    assert identity["mlp"] == backend
    assert "MODEL" not in identity["configuration"]


def test_default_socket_directory_and_socket_are_private(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    socket_path = Path(default_socket_path())
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
    try:
        assert socket_path.parent.stat().st_mode & 0o777 == 0o700
        assert socket_path.stat().st_mode & 0o777 == 0o600
    finally:
        _stop_server(socket_path, thread)


def test_ensure_socket_directory_rejects_symlinked_default_parent(
    tmp_path: Path, monkeypatch
):
    # On a shared host the default parent name is predictable; a pre-planted
    # symlink there would let another user capture our socket. lstat must catch
    # it even though isdir/makedirs/chmod would follow the link.
    target = tmp_path / "attacker"
    target.mkdir(mode=0o700)
    parent = tmp_path / "vpmdk-uid"
    parent.symlink_to(target, target_is_directory=True)
    sock = parent / "default.sock"
    monkeypatch.setattr(server_module, "default_socket_path", lambda: str(sock))

    with pytest.raises(RuntimeError, match="symlink"):
        server_module.ensure_socket_directory(str(sock))


def test_ensure_socket_directory_rejects_dangling_symlinked_default_parent(
    tmp_path: Path, monkeypatch
):
    # A pre-planted *dangling* symlink makes makedirs(exist_ok=True) raise
    # FileExistsError (isdir is False for it), before the fd check runs. It must
    # still surface the clear symlink rejection, not a cryptic FileExistsError.
    parent = tmp_path / "vpmdk-uid"
    parent.symlink_to(tmp_path / "nonexistent-target", target_is_directory=True)
    sock = parent / "default.sock"
    monkeypatch.setattr(server_module, "default_socket_path", lambda: str(sock))

    with pytest.raises(RuntimeError, match="symlink"):
        server_module.ensure_socket_directory(str(sock))


def test_ensure_socket_directory_rejects_unsecurable_default_parent(
    tmp_path: Path, monkeypatch
):
    # A group/other-accessible parent that cannot be tightened to 0700 (fchmod
    # fails) is rejected rather than used for a reachable socket.
    parent = tmp_path / "vpmdk-uid"
    parent.mkdir()
    os.chmod(parent, 0o777)
    sock = parent / "default.sock"
    monkeypatch.setattr(server_module, "default_socket_path", lambda: str(sock))

    def deny(fd, mode):
        raise PermissionError("cannot chmod")

    monkeypatch.setattr(os, "fchmod", deny)
    with pytest.raises(RuntimeError, match="cannot be secured"):
        server_module.ensure_socket_directory(str(sock))


def test_ensure_socket_directory_accepts_private_default_parent(
    tmp_path: Path, monkeypatch
):
    sock = tmp_path / "vpmdk-uid" / "default.sock"
    monkeypatch.setattr(server_module, "default_socket_path", lambda: str(sock))

    server_module.ensure_socket_directory(str(sock))

    parent = tmp_path / "vpmdk-uid"
    assert parent.is_dir() and not parent.is_symlink()
    assert os.stat(parent).st_mode & 0o777 == 0o700


def test_ensure_socket_directory_private_parent_needs_no_chmod(
    tmp_path: Path, monkeypatch
):
    # An already-private (0700) parent must not trigger fchmod, so a chmod-
    # restricted-but-writable filesystem (overlay/9p/DrvFs) does not false-reject
    # a legitimate default-path startup. fchmod raising here would fail the test.
    parent = tmp_path / "vpmdk-uid"
    parent.mkdir(mode=0o700)
    sock = parent / "default.sock"
    monkeypatch.setattr(server_module, "default_socket_path", lambda: str(sock))

    def explode(fd, mode):  # pragma: no cover - must never be called
        raise AssertionError("fchmod must not run on an already-private parent")

    monkeypatch.setattr(os, "fchmod", explode)
    server_module.ensure_socket_directory(str(sock))  # no raise

    assert os.stat(parent).st_mode & 0o777 == 0o700


def test_verify_socket_parent_ownership_rejects_foreign_owner(tmp_path: Path):
    # A directory whose owner differs from the owner of a probe file we create in
    # it belongs to another user (e.g. an attacker's pre-created /tmp/vpmdk-0
    # under a root server) and must be rejected.
    dir_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        foreign_uid = os.stat(str(tmp_path)).st_uid + 1
        with pytest.raises(RuntimeError, match="another user controls"):
            server_module._verify_socket_parent_ownership(
                dir_fd, foreign_uid, str(tmp_path)
            )
    finally:
        os.close(dir_fd)


def test_verify_socket_parent_ownership_accepts_self_owner(tmp_path: Path):
    # Our own directory: the probe we create is owned by the same identity, so
    # ownership verifies even where st_uid == geteuid() would not (root,
    # uid-mapping, NFS root_squash). The probe file is cleaned up.
    dir_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        our_uid = os.stat(str(tmp_path)).st_uid
        server_module._verify_socket_parent_ownership(
            dir_fd, our_uid, str(tmp_path)
        )  # no raise
    finally:
        os.close(dir_fd)

    assert not any(p.name.startswith(".vpmdk-owner-probe") for p in tmp_path.iterdir())


def test_verify_socket_parent_ownership_tolerates_stale_probe(tmp_path: Path):
    # A stale probe from a crashed same-PID process must be removed and retried,
    # not turned into a hard "cannot verify ownership" abort.
    stale = tmp_path / f".vpmdk-owner-probe-{os.getpid()}"
    stale.write_text("stale")
    dir_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        our_uid = os.stat(str(tmp_path)).st_uid
        server_module._verify_socket_parent_ownership(
            dir_fd, our_uid, str(tmp_path)
        )  # no raise
    finally:
        os.close(dir_fd)

    assert not stale.exists()


def test_verify_socket_parent_ownership_rejects_persistent_probe_conflict(
    tmp_path: Path, monkeypatch
):
    # If the probe name stays occupied across the retry (e.g. an attacker holding
    # it in a directory they own), fail closed rather than skip the check.
    dir_fd = os.open(str(tmp_path), os.O_RDONLY | os.O_DIRECTORY)
    real_open = os.open

    def failing_open(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(".vpmdk-owner-probe"):
            raise FileExistsError(17, "File exists")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", failing_open)
    try:
        with pytest.raises(RuntimeError, match="being held"):
            server_module._verify_socket_parent_ownership(
                dir_fd, os.stat(str(tmp_path)).st_uid, str(tmp_path)
            )
    finally:
        os.close(dir_fd)


def test_ensure_socket_directory_tightens_loose_default_parent(
    tmp_path: Path, monkeypatch
):
    parent = tmp_path / "vpmdk-uid"
    parent.mkdir()
    os.chmod(parent, 0o777)
    sock = parent / "default.sock"
    monkeypatch.setattr(server_module, "default_socket_path", lambda: str(sock))

    server_module.ensure_socket_directory(str(sock))

    assert os.stat(parent).st_mode & 0o777 == 0o700


def test_package_version_is_cached(monkeypatch):
    # The installed version is constant for the process; status() is polled
    # repeatedly, so the importlib.metadata scan must run at most once.
    server_module._package_version.cache_clear()
    calls = {"n": 0}

    def counting_version(name):
        calls["n"] += 1
        return "9.9.9"

    monkeypatch.setattr(server_module.importlib.metadata, "version", counting_version)
    try:
        assert server_module._package_version() == "9.9.9"
        assert server_module._package_version() == "9.9.9"
        assert calls["n"] == 1
    finally:
        server_module._package_version.cache_clear()


def test_ensure_socket_directory_leaves_custom_symlinked_parent(
    tmp_path: Path, monkeypatch
):
    # The self-ownership/symlink check applies only to the default parent under a
    # world-writable base; a user-chosen custom path (e.g. a symlinked runtime
    # dir they own) must keep working.
    monkeypatch.setattr(
        server_module, "default_socket_path", lambda: str(tmp_path / "other.sock")
    )
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real_dir, target_is_directory=True)

    server_module.ensure_socket_directory(str(linked / "custom.sock"))  # no raise


def test_ensure_socket_directory_custom_parent_needs_no_chmod(
    tmp_path: Path, monkeypatch
):
    # A fresh custom-socket parent must not be chmod'd (makedirs already makes it
    # private), so a chmod-restricted-but-writable filesystem does not
    # false-reject a legitimate custom --socket path.
    monkeypatch.setattr(
        server_module, "default_socket_path", lambda: str(tmp_path / "other.sock")
    )

    def explode(path, mode):  # pragma: no cover - must never be called
        raise AssertionError("chmod must not run on a custom socket parent")

    monkeypatch.setattr(os, "chmod", explode)
    parent = tmp_path / "custom-run"
    server_module.ensure_socket_directory(str(parent / "s.sock"))  # no raise

    assert parent.is_dir()
    assert os.stat(parent).st_mode & 0o077 == 0


def test_client_subcommands_match_dispatch_constant():
    # The parser authority and both dispatch tables must derive from one source,
    # so adding a client subcommand cannot leave a routing table stale.
    import argparse

    import vpmdk_entry
    from vpmdk_client import add_client_subcommands
    from vpmdk_core.cli import _SERVER_SUBCOMMANDS
    from vpmdk_protocol import CLIENT_SUBCOMMANDS

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_client_subcommands(subparsers)

    assert set(subparsers.choices) == set(CLIENT_SUBCOMMANDS)
    assert vpmdk_entry.CLIENT_SUBCOMMANDS == CLIENT_SUBCOMMANDS
    assert _SERVER_SUBCOMMANDS == frozenset({"serve", *CLIENT_SUBCOMMANDS})


def test_startup_unknown_named_model_is_preserved_for_builder_warning(
    tmp_path: Path, monkeypatch
):
    # A resolver that silently substitutes an unknown named model for the
    # installed default (GRACE) must not have that substitution baked into
    # tags["MODEL"] at startup, or the builder receives a known name and skips
    # the documented "Unknown ... using default" warning.
    from vpmdk_core.backend_common import ModelReference, ModelReferenceKind

    (tmp_path / "BCAR").write_text("MLP=GRACE\nMODEL=grace-typo\nDEVICE=cpu\n")

    def fake_ref(backend, model_value, *, base_dir=None):
        return ModelReference(
            ModelReferenceKind.NAMED_MODEL, "GRACE-2L-OMAT", explicit=True
        )

    monkeypatch.setattr(vpmdk, "_resolve_backend_model_reference", fake_ref)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda tags, *, structure=None: captured.update(tags) or DummyCalculator(),
    )

    _, tags, _ = _load_backend_for_server(str(tmp_path), None)

    assert captured["MODEL"] == "grace-typo"
    assert tags["MODEL"] == "grace-typo"


def test_startup_local_model_path_is_canonicalized_for_builder(
    tmp_path: Path, monkeypatch
):
    from vpmdk_core.backend_common import ModelReference, ModelReferenceKind

    (tmp_path / "BCAR").write_text("MLP=MACE\nMODEL=weights.pt\nDEVICE=cpu\n")
    loader = str(tmp_path / "weights.pt")

    def fake_ref(backend, model_value, *, base_dir=None):
        return ModelReference(
            ModelReferenceKind.LOCAL_PATH, loader, explicit=True, identity=loader
        )

    monkeypatch.setattr(vpmdk, "_resolve_backend_model_reference", fake_ref)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda tags, *, structure=None: captured.update(tags) or DummyCalculator(),
    )

    _, tags, _ = _load_backend_for_server(str(tmp_path), None)

    # A local checkpoint path is still canonicalized (relative -> absolute).
    assert captured["MODEL"] == loader
    assert tags["MODEL"] == loader


def test_unknown_grace_request_warning(monkeypatch):
    from vpmdk_core.backend_common import ModelReference, ModelReferenceKind

    resident = {"mlp": "GRACE", "model": "GRACE-2L-OMAT"}

    def substitute_to_default(backend, model_value, *, base_dir=None):
        return ModelReference(
            ModelReferenceKind.NAMED_MODEL, "GRACE-2L-OMAT", explicit=True
        )

    monkeypatch.setattr(vpmdk, "_resolve_backend_model_reference", substitute_to_default)
    monkeypatch.setattr(vpmdk, "_resolve_grace_foundation_model", lambda value: None)

    warn = server_module._unknown_grace_request_warning(
        resident, {"MODEL": "grace-typo"}, base_dir="/req"
    )
    assert warn is not None
    assert "grace-typo" in warn and "GRACE-2L-OMAT" in warn

    # A non-GRACE resident is out of scope.
    assert (
        server_module._unknown_grace_request_warning(
            {"mlp": "CHGNET"}, {"MODEL": "x"}, base_dir="/req"
        )
        is None
    )

    # An explicit MLP override to another backend is left to the mismatch check.
    assert (
        server_module._unknown_grace_request_warning(
            resident, {"MLP": "CHGNET", "MODEL": "grace-typo"}, base_dir="/req"
        )
        is None
    )

    # A recognized name must not warn.
    monkeypatch.setattr(
        vpmdk, "_resolve_grace_foundation_model", lambda value: "GRACE-2L-OMAT"
    )
    monkeypatch.setattr(
        vpmdk,
        "_resolve_backend_model_reference",
        lambda backend, model_value, *, base_dir=None: ModelReference(
            ModelReferenceKind.NAMED_MODEL, str(model_value), explicit=True
        ),
    )
    assert (
        server_module._unknown_grace_request_warning(
            resident, {"MODEL": "GRACE-2L-OMAT"}, base_dir="/req"
        )
        is None
    )

    # A local checkpoint path is a real model, not a foundation-name typo.
    monkeypatch.setattr(vpmdk, "_resolve_grace_foundation_model", lambda value: None)
    monkeypatch.setattr(
        vpmdk,
        "_resolve_backend_model_reference",
        lambda backend, model_value, *, base_dir=None: ModelReference(
            ModelReferenceKind.LOCAL_PATH,
            "/abs/weights.pt",
            explicit=True,
            identity="/abs/weights.pt",
        ),
    )
    assert (
        server_module._unknown_grace_request_warning(
            resident, {"MODEL": "./weights.pt"}, base_dir="/req"
        )
        is None
    )


def test_status_serializes_with_enqueue_lock(tmp_path: Path):
    # status() must take the same enqueue->state lock order the worker uses, so a
    # snapshot cannot slip into the dequeue->busy window. Proven by holding the
    # enqueue lock and confirming status() blocks until it is released.
    server = VPMDKServer(
        str(tmp_path / "s.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )
    done = threading.Event()
    result: dict[str, object] = {}

    server._enqueue_lock.acquire()
    try:
        thread = threading.Thread(
            target=lambda: (result.update(status=server.status()), done.set()),
            daemon=True,
        )
        thread.start()
        assert not done.wait(0.2), "status() must block while the enqueue lock is held"
    finally:
        server._enqueue_lock.release()

    assert done.wait(2.0)
    assert result["status"]["state"] == "idle"


def test_request_backend_warning_is_streamed_to_client(tmp_path: Path, monkeypatch):
    # The resident builder never re-runs per request, so a request-time backend
    # warning must be surfaced to the client explicitly. Verify the wiring: a
    # non-empty warning is printed into the streamed log rather than dropped.
    socket_path = tmp_path / "server.sock"
    monkeypatch.setattr(
        server_module,
        "_unknown_grace_request_warning",
        lambda resident, request_tags, *, base_dir: (
            "Warning: Unknown GRACE model 'grace-typo', reusing resident "
            "default GRACE-2L-OMAT instead."
        ),
    )
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
    lines: list[str] = []
    try:
        request_dir = tmp_path / "req"
        request_dir.mkdir()
        VPMDKClient(str(socket_path)).run(str(request_dir), log_callback=lines.append)
    finally:
        _stop_server(socket_path, thread)

    assert any("Unknown GRACE model 'grace-typo'" in line for line in lines)


def test_multiple_clients_are_executed_fifo_and_never_overlap(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    active = 0
    max_active = 0
    order: list[str] = []
    lock = threading.Lock()

    def executor(workdir: str, *, calculator) -> None:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            order.append(Path(workdir).name)
        time.sleep(0.1)
        with lock:
            active -= 1
        print("Calculation completed.")

    _, server_thread = _start_server(socket_path, executor=executor)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    errors: list[BaseException] = []

    def submit(path: Path) -> None:
        try:
            VPMDKClient(str(socket_path)).run(str(path))
        except BaseException as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    first_thread = threading.Thread(target=submit, args=(first,))
    second_thread = threading.Thread(target=submit, args=(second,))
    try:
        first_thread.start()
        time.sleep(0.02)
        second_thread.start()
        first_thread.join(timeout=2.0)
        second_thread.join(timeout=2.0)
    finally:
        _stop_server(socket_path, server_thread)

    assert errors == []
    assert order == ["first", "second"]
    assert max_active == 1


def test_run_requests_follow_connection_accept_order(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    order: list[str] = []

    def executor(workdir: str, *, calculator) -> None:
        order.append(Path(workdir).name)

    server, server_thread = _start_server(socket_path, executor=executor)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    def send_run(connection: socket.socket, workdir: Path) -> None:
        request = {"op": "run", "version": 1, "workdir": str(workdir)}
        connection.sendall((json.dumps(request) + "\n").encode())

    def read_done(connection: socket.socket) -> None:
        stream = connection.makefile("rb")
        while True:
            event = json.loads(stream.readline())
            if event.get("event") == "done":
                assert event.get("ok") is True
                return

    try:
        baseline = server._next_accept_sequence
        first.connect(str(socket_path))
        _wait_for(lambda: server._next_accept_sequence >= baseline + 1)
        second.connect(str(socket_path))
        _wait_for(lambda: server._next_accept_sequence >= baseline + 2)

        # The later connection sends a complete run first. It must remain
        # behind the earlier accepted connection until that request arrives.
        send_run(second, second_dir)
        time.sleep(0.05)
        assert order == []
        send_run(first, first_dir)
        read_done(first)
        read_done(second)
    finally:
        first.close()
        second.close()
        _stop_server(socket_path, server_thread)

    assert order == ["first", "second"]


def test_long_running_request_streams_heartbeats(tmp_path: Path):
    socket_path = tmp_path / "server.sock"

    def executor(workdir: str, *, calculator) -> None:
        time.sleep(0.12)
        print("Calculation completed.")

    _, thread = _start_server(
        socket_path,
        executor=executor,
        heartbeat_interval=0.03,
    )
    events: list[dict[str, object]] = []
    try:
        VPMDKClient(str(socket_path)).run(str(tmp_path), event_callback=events.append)
    finally:
        _stop_server(socket_path, thread)

    assert events[0]["event"] == "accepted"
    assert any(event["event"] == "heartbeat" for event in events)
    assert events[-1]["event"] == "done"


def test_status_responds_while_calculation_is_busy(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    started = threading.Event()
    release = threading.Event()

    def executor(workdir: str, *, calculator) -> None:
        started.set()
        assert release.wait(2.0)
        print("Calculation completed.")

    _, server_thread = _start_server(socket_path, executor=executor)
    run_thread = threading.Thread(
        target=lambda: VPMDKClient(str(socket_path)).run(str(tmp_path))
    )
    try:
        run_thread.start()
        assert started.wait(1.0)
        before_status = time.monotonic()
        status = VPMDKClient(str(socket_path)).status(timeout=0.5)
        assert time.monotonic() - before_status < 0.5
        assert status["state"] == "busy"
        assert status["current_workdir"] == str(tmp_path)
    finally:
        release.set()
        run_thread.join(timeout=2.0)
        _stop_server(socket_path, server_thread)


def test_failed_request_is_isolated_from_following_request(tmp_path: Path):
    socket_path = tmp_path / "server.sock"

    def executor(workdir: str, *, calculator) -> None:
        if Path(workdir).name == "bad":
            raise RuntimeError("deliberate request failure")
        print("Calculation completed.")

    _, thread = _start_server(socket_path, executor=executor)
    bad = tmp_path / "bad"
    good = tmp_path / "good"
    bad.mkdir()
    good.mkdir()
    client = VPMDKClient(str(socket_path))
    try:
        with pytest.raises(RemoteCalculationError, match="deliberate request failure") as caught:
            client.run(str(bad))
        assert caught.value.traceback and "RuntimeError" in caught.value.traceback
        client.run(str(good))
        status = client.status()
        assert status["jobs_failed"] == 1
        assert status["jobs_completed"] == 1
    finally:
        _stop_server(socket_path, thread)


def test_oversized_log_line_is_split_into_protocol_safe_events(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(server_module, "MAX_REQUEST_BYTES", 2048)
    monkeypatch.setattr(lightweight_client_module, "MAX_REQUEST_BYTES", 2048)
    socket_path = tmp_path / "server.sock"
    large_line = "構造データ" * 2000

    def executor(workdir: str, *, calculator) -> None:
        print(large_line)

    _, thread = _start_server(socket_path, executor=executor)
    received: list[str] = []
    events: list[dict[str, object]] = []
    try:
        VPMDKClient(str(socket_path)).run(
            str(tmp_path),
            log_callback=received.append,
            event_callback=lambda event: events.append(dict(event)),
        )
        args = lightweight_client_module.argparse.Namespace(
            command="run",
            dir=str(tmp_path),
            socket=str(socket_path),
            timeout=0.0,
        )
        assert lightweight_client_module.client_cli(args) == 0
    finally:
        _stop_server(socket_path, thread)

    log_events = [event for event in events if event.get("event") == "log"]
    assert len(log_events) > 1
    assert all("continued" in event for event in log_events)
    assert all(event["continued"] is True for event in log_events[:-1])
    assert log_events[-1]["continued"] is False
    assert received == [large_line]
    assert capsys.readouterr().out == large_line + "\nCalculation completed.\n"


def test_oversized_exception_remains_a_calculation_failure(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(server_module, "MAX_REQUEST_BYTES", 2048)
    monkeypatch.setattr(lightweight_client_module, "MAX_REQUEST_BYTES", 2048)
    socket_path = tmp_path / "server.sock"

    def executor(workdir: str, *, calculator) -> None:
        raise RuntimeError("oversized failure " + "x" * 10000)

    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteCalculationError) as caught:
            VPMDKClient(str(socket_path)).run(str(tmp_path))
    finally:
        _stop_server(socket_path, thread)

    assert "oversized failure" in str(caught.value)
    assert "truncated to VPMDK protocol limit" in str(caught.value)
    assert caught.value.traceback is not None
    assert "truncated to VPMDK protocol limit" in caught.value.traceback


class _RequestBytesConnection:
    def __init__(self, payload: bytes):
        self.payload = payload
        self._offset = 0

    def settimeout(self, timeout: float) -> None:
        pass

    def recv(self, size: int) -> bytes:
        chunk = self.payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def makefile(self, mode: str):
        return io.BytesIO(self.payload)


def test_request_size_limit_excludes_ndjson_newline(tmp_path: Path, monkeypatch):
    limit = 256
    monkeypatch.setattr(server_module, "MAX_REQUEST_BYTES", limit)
    server = VPMDKServer(
        str(tmp_path / "server.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )
    request = {"op": "status", "version": 1, "padding": ""}
    encoded = json.dumps(request, separators=(",", ":")).encode()
    request["padding"] = "x" * (limit - len(encoded))
    encoded = json.dumps(request, separators=(",", ":")).encode()
    assert len(encoded) == limit

    parsed = server._read_request(_RequestBytesConnection(encoded + b"\n"))
    assert parsed == request
    parsed_crlf = server._read_request(_RequestBytesConnection(encoded + b"\r\n"))
    assert parsed_crlf == request

    oversized = dict(request, padding=request["padding"] + "x")
    oversized_bytes = json.dumps(oversized, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="size limit"):
        server._read_request(_RequestBytesConnection(oversized_bytes + b"\n"))
    server._cleanup()


@pytest.mark.parametrize(
    ("request_bcar", "expected_tag"),
    [
        ("MLP=MACE\n", "MLP"),
        ("MLP=CHGNET\nMODEL=other.pt\n", "MODEL"),
        ("MLP=CHGNET\nCHGNET_GRAPH_CONVERTER=fast\n", "GRAPH_CONVERTER_ALGORITHM"),
    ],
)
def test_backend_configuration_mismatch_is_rejected(
    tmp_path: Path,
    request_bcar: str,
    expected_tag: str,
):
    socket_path = tmp_path / "server.sock"
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "BCAR").write_text(request_bcar)
    ran = False

    def executor(workdir: str, *, calculator) -> None:
        nonlocal ran
        ran = True

    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteBackendMismatch, match=expected_tag):
            VPMDKClient(str(socket_path)).run(str(workdir))
        assert not ran
    finally:
        _stop_server(socket_path, thread)


def test_foreign_backend_tag_is_ignored_like_one_shot(tmp_path: Path):
    # A request carrying a leftover tag that belongs exclusively to a DIFFERENT
    # backend (e.g. ORB_PRECISION / MATTERSIM_STRESS_WEIGHT under an MLP=CHGNET
    # resident) is ignored by the one-shot builder, so the server must ignore it
    # too rather than raise a spurious exit-5 mismatch (SERVER_MODE_SPEC 3.4).
    resident = backend_identity(
        {"MLP": "CHGNET", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    for foreign in ("ORB_PRECISION", "MATTERSIM_STRESS_WEIGHT", "NEQUIX_BACKEND"):
        validate_request_backend(
            resident,
            {"MLP": "CHGNET", foreign: "float64", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )

    # SAFETY: an own-backend tag that genuinely differs is still compared and
    # rejected -- the filter must never ignore a tag the resident builder reads.
    orb_resident = backend_identity(
        {"MLP": "ORB", "ORB_PRECISION": "float32-high", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch, match="ORB_PRECISION"):
        validate_request_backend(
            orb_resident,
            {"MLP": "ORB", "ORB_PRECISION": "float64", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_foreign_family_backend_tags_are_ignored_like_one_shot(tmp_path: Path):
    # Family-shared / non-prefixed leftover tags from ANOTHER backend must also be
    # ignored, not just single-MLP-exclusive prefixes. A CHGNET builder never
    # reads SEVENNET_*/FAIRCHEM_*/the Matlantis aliases, so a CHGNET resident must
    # accept a request carrying them (as one-shot does), not raise exit 5.
    chgnet = backend_identity({"MLP": "CHGNET", "DEVICE": "cpu"}, base_dir=str(tmp_path))
    for foreign, value in (
        ("SEVENNET_ENABLE_FLASH", "0"),
        ("FAIRCHEM_V1_PREDICTOR", "0"),
        ("PRIORITY", "50"),
        ("MODEL_VERSION", "v8"),
        ("CALC_MODE", "pbe"),
    ):
        validate_request_backend(
            chgnet,
            {"MLP": "CHGNET", foreign: value, "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )

    # A HIENET resident is NOT in the SevenNet family, so SEVENNET_* is foreign to
    # it and ignored -- verifying the family boundary, not just prefix matching.
    hienet = backend_identity({"MLP": "HIENET", "DEVICE": "cpu"}, base_dir=str(tmp_path))
    validate_request_backend(
        hienet,
        {"MLP": "HIENET", "SEVENNET_ENABLE_FLASH": "1", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )

    # SAFETY: a member of the family still compares its own shared tags. A SevenNet
    # (and EquFlash, which delegates to the SevenNet builder) resident must reject
    # a genuinely different SEVENNET_ flag; GRAPH_CONVERTER (read by CHGNET) is a
    # real config, not foreign, so it is still compared too.
    sevennet = backend_identity(
        {"MLP": "SEVENNET", "SEVENNET_ENABLE_CUEQ": "1", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch, match="SEVENNET_ENABLE_CUEQ"):
        validate_request_backend(
            sevennet,
            {"MLP": "SEVENNET", "SEVENNET_ENABLE_CUEQ": "0", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )
    matlantis = backend_identity(
        {"MLP": "MATLANTIS", "PRIORITY": "50", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    with pytest.raises(BackendConfigurationMismatch, match="MATLANTIS_PRIORITY"):
        validate_request_backend(
            matlantis,
            {"MLP": "MATLANTIS", "PRIORITY": "80", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )
    with pytest.raises(BackendConfigurationMismatch, match="GRAPH_CONVERTER_ALGORITHM"):
        validate_request_backend(
            chgnet,
            {"MLP": "CHGNET", "CHGNET_GRAPH_CONVERTER": "fast", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_fairchem_family_tags_are_owned_per_sub_builder(tmp_path: Path):
    # A FairChem-family tag must be attributed to the EXACT builders that read it:
    # EQUIFORMER_V3_* only to EQUIFORMER_V3; FAIRCHEM_TASK/FAIRCHEM_INFERENCE_SETTINGS
    # only to the modern FAIRCHEM/V2/ESEN builder; FAIRCHEM_CONFIG/FAIRCHEM_V1_PREDICTOR
    # to the v1 builder AND to EQUIFORMER_V3 (which delegates to it). A tag foreign
    # to the resident's sub-builder must be ignored, not exit-5'd.
    foreign_cases = (
        ("FAIRCHEM", "EQUIFORMER_V3_MODULE", "foo"),
        ("FAIRCHEM", "FAIRCHEM_CONFIG", "x.yaml"),
        ("FAIRCHEM", "FAIRCHEM_V1_PREDICTOR", "0"),
        ("FAIRCHEM_V1", "FAIRCHEM_TASK", "omat"),
        ("FAIRCHEM_V1", "EQUIFORMER_V3_MODULE", "foo"),
        ("EQUIFORMER_V3", "FAIRCHEM_TASK", "omat"),
    )
    for mlp, tag, value in foreign_cases:
        resident = backend_identity({"MLP": mlp, "DEVICE": "cpu"}, base_dir=str(tmp_path))
        validate_request_backend(
            resident,
            {"MLP": mlp, tag: value, "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )
    # SAFETY: each sub-builder still compares ITS OWN tags.
    v2 = backend_identity(
        {"MLP": "FAIRCHEM", "FAIRCHEM_TASK": "omat", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch, match="FAIRCHEM_TASK"):
        validate_request_backend(
            v2,
            {"MLP": "FAIRCHEM", "FAIRCHEM_TASK": "omc", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )
    # SAFETY: EQUIFORMER_V3 delegates to _build_fairchem_v1_calculator, which reads
    # FAIRCHEM_V1_PREDICTOR (predictor-backed vs OCPCalculator -- numerically
    # different) and FAIRCHEM_CONFIG (config_yml). An EQUIFORMER_V3 resident must
    # therefore COMPARE both -- stripping them would silently accept a mismatched
    # predictor/config and run the wrong calculator (SERVER_MODE_SPEC 3.4).
    eqv3 = backend_identity(
        {"MLP": "EQUIFORMER_V3", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    with pytest.raises(BackendConfigurationMismatch, match="FAIRCHEM_V1_PREDICTOR"):
        validate_request_backend(
            eqv3,
            {"MLP": "EQUIFORMER_V3", "FAIRCHEM_V1_PREDICTOR": "1", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )
    with pytest.raises(BackendConfigurationMismatch, match="FAIRCHEM_CONFIG"):
        validate_request_backend(
            eqv3,
            {"MLP": "EQUIFORMER_V3", "FAIRCHEM_CONFIG": "x.yaml", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )
    # And a predictor-backed EQUIFORMER_V3 resident advertises the real value, so an
    # equivalent request naming the same predictor is accepted (not a false exit 5).
    eqv3_pred = backend_identity(
        {"MLP": "EQUIFORMER_V3", "FAIRCHEM_V1_PREDICTOR": "1", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert eqv3_pred["configuration"].get("FAIRCHEM_V1_PREDICTOR") is True
    validate_request_backend(
        eqv3_pred,
        {"MLP": "EQUIFORMER_V3", "FAIRCHEM_V1_PREDICTOR": "true", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_generic_graph_converter_tag_is_owned_by_chgnet_and_matris(tmp_path: Path):
    # Generic GRAPH_CONVERTER* is read/applied only by CHGNet and MatRIS. A MACE or
    # ORB resident ignores it, so the server must too (not exit-5); a CHGNet
    # resident still compares it (a graph-converter override is a real config).
    for mlp, extra in (
        ("MACE", {}),
        ("ORB", {"ORB_MODEL": "orb-v3-conservative-20-omat"}),
    ):
        resident = backend_identity(
            {"MLP": mlp, "DEVICE": "cpu", **extra}, base_dir=str(tmp_path)
        )
        validate_request_backend(
            resident,
            {"MLP": mlp, "GRAPH_CONVERTER": "fast", "DEVICE": "cpu", **extra},
            request_base_dir=str(tmp_path),
        )
    chgnet = backend_identity({"MLP": "CHGNET", "DEVICE": "cpu"}, base_dir=str(tmp_path))
    with pytest.raises(BackendConfigurationMismatch, match="GRAPH_CONVERTER_ALGORITHM"):
        validate_request_backend(
            chgnet,
            {"MLP": "CHGNET", "GRAPH_CONVERTER": "fast", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_blank_device_resolves_like_an_omitted_one_and_is_still_compared(tmp_path: Path):
    # A present-but-blank DEVICE= names no device, and the builders resolve it the
    # same way they resolve an omitted one -- verified for the default backend:
    # chgnet's determine_device("") == determine_device(None). Canonicalizing it
    # to "" instead made a resident started from a template BCAR with `DEVICE=`
    # advertise device="" and then PERMANENTLY reject every request naming the
    # device it actually runs on. It must resolve to a REAL device and still be
    # compared (a genuinely different device must still mismatch).
    autodetected = backend_identity(
        {"MLP": "CHGNET"}, base_dir=str(tmp_path)
    )["device"]
    assert autodetected  # never the empty string

    blank_resident = backend_identity(
        {"MLP": "CHGNET", "DEVICE": ""}, base_dir=str(tmp_path)
    )
    assert blank_resident["device"] == autodetected

    # Blank, explicit-same, and omitted all describe the resident's device.
    for request_tags in (
        {"MLP": "CHGNET", "DEVICE": ""},
        {"MLP": "CHGNET", "DEVICE": autodetected},
        {"MLP": "CHGNET"},
    ):
        validate_request_backend(
            blank_resident, request_tags, request_base_dir=str(tmp_path)
        )

    # A genuinely different device still mismatches (the comparison is not lost).
    other = "cuda" if autodetected != "cuda" else "cpu"
    with pytest.raises(BackendConfigurationMismatch, match="DEVICE"):
        validate_request_backend(
            blank_resident,
            {"MLP": "CHGNET", "DEVICE": other},
            request_base_dir=str(tmp_path),
        )


def test_blank_device_canonicalizes_to_cpu_for_cpu_defaulting_backends(tmp_path: Path):
    # EQNORM/ALPHANET/HIENET and the SevenNet family resolve DEVICE via
    # `_resolve_device(...) or "cpu"`, so a present-but-blank DEVICE actually runs
    # on CPU. The resident must ADVERTISE "cpu" (not "") and accept an equivalent
    # explicit DEVICE=cpu request, while still rejecting DEVICE=cuda -- otherwise a
    # blank-device resident spuriously exit-5's a cpu request (SERVER_MODE_SPEC 3.4).
    # MATRIS (and MATGL) belong to this family too -- matris.py's build paths
    # apply `device or "cpu"`. R139 found this test still listing MATRIS in
    # the OUTSIDE loop below, which pinned the pre-fix claim and failed on any
    # CUDA host (blank -> cpu but omitted -> cuda) while staying vacuously
    # green on CPU-only hosts.
    for mlp in ("EQNORM", "ALPHANET", "HIENET", "SEVENNET", "FLASHTP", "EQUFLASH", "MATRIS"):
        resident = backend_identity(
            {"MLP": mlp, "DEVICE": ""}, base_dir=str(tmp_path)
        )
        assert resident["device"] == "cpu", (mlp, resident["device"])
        # Explicit cpu and a blank device both match the cpu the builder runs on.
        validate_request_backend(
            resident, {"MLP": mlp, "DEVICE": "cpu"}, request_base_dir=str(tmp_path)
        )
        validate_request_backend(
            resident, {"MLP": mlp, "DEVICE": ""}, request_base_dir=str(tmp_path)
        )
        # A genuinely different device still mismatches.
        with pytest.raises(BackendConfigurationMismatch, match="DEVICE"):
            validate_request_backend(
                resident,
                {"MLP": mlp, "DEVICE": "cuda"},
                request_base_dir=str(tmp_path),
            )

    # Backends OUTSIDE that family (no `or "cpu"`) forward the blank to a
    # calculator that treats a falsy device as autodetect, so their blank resolves
    # like an omitted DEVICE -- never to the empty string, which would make the
    # resident reject every request naming its real device.
    for mlp in ("CHGNET",):
        omitted = backend_identity({"MLP": mlp}, base_dir=str(tmp_path))["device"]
        resident = backend_identity(
            {"MLP": mlp, "DEVICE": ""}, base_dir=str(tmp_path)
        )
        assert resident["device"] == omitted, (mlp, resident["device"])
        assert resident["device"] != ""
        validate_request_backend(
            resident,
            {"MLP": mlp, "DEVICE": omitted},
            request_base_dir=str(tmp_path),
        )


def test_resident_advertises_detected_graph_converter_algorithm(tmp_path: Path):
    # R140 (P3): a tagless CHGNET resident advertised
    # GRAPH_CONVERTER_ALGORITHM=None, so a request spelling out the algorithm
    # the bundled default model ALREADY uses (CHGNet v0.3.0 ships 'fast') was
    # rejected exit 5 while one-shot computed byte-identical numbers. The
    # server now READS the algorithm from the loaded calculator at startup --
    # the _detect_calculator_device pattern -- so an explicit matching request
    # matches, and a genuinely different one still rejects.
    calculator = DummyCalculator()
    calculator.model = SimpleNamespace(
        graph_converter=SimpleNamespace(algorithm="fast")
    )
    server = VPMDKServer(
        str(tmp_path / "gc.sock"),
        calculator,
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )

    validate_request_backend(
        server.backend,
        {"MLP": "CHGNET", "DEVICE": "cpu", "GRAPH_CONVERTER": "fast"},
        request_base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch, match="GRAPH_CONVERTER"):
        validate_request_backend(
            server.backend,
            {"MLP": "CHGNET", "DEVICE": "cpu", "GRAPH_CONVERTER": "legacy"},
            request_base_dir=str(tmp_path),
        )
    # A stale FOREIGN spelling (a CHGNET resident with a leftover
    # MATRIS_GRAPH_CONVERTER from an edited template) is ignored by the
    # builder AND stripped as foreign, so it must NOT suppress detection --
    # R141 found it leaving the resident advertising None and rejecting every
    # explicit converter request in both directions.
    foreign = VPMDKServer(
        str(tmp_path / "gc3.sock"),
        calculator,
        {"MLP": "CHGNET", "DEVICE": "cpu", "MATRIS_GRAPH_CONVERTER": "legacy"},
        backend_base_dir=str(tmp_path),
    )
    validate_request_backend(
        foreign.backend,
        {"MLP": "CHGNET", "DEVICE": "cpu", "GRAPH_CONVERTER": "fast"},
        request_base_dir=str(tmp_path),
    )

    # An EXPLICIT startup tag stays authoritative over detection.
    explicit = VPMDKServer(
        str(tmp_path / "gc2.sock"),
        calculator,
        {"MLP": "CHGNET", "DEVICE": "cpu", "GRAPH_CONVERTER": "legacy"},
        backend_base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch, match="GRAPH_CONVERTER"):
        validate_request_backend(
            explicit.backend,
            {"MLP": "CHGNET", "DEVICE": "cpu", "GRAPH_CONVERTER": "fast"},
            request_base_dir=str(tmp_path),
        )


def test_device_index_zero_matches_the_unindexed_spelling(tmp_path: Path):
    # R139 (P3): only the literal 'cuda:0' was normalized, so the torch-
    # equivalent cpu:0/cpu pair was rejected exit 5 in BOTH directions while
    # one-shot built the identical calculator (torch.device('cpu:0') ==
    # torch.device('cpu'); CHGNet energies byte-identical). Index 0 is the
    # default device for every type; ':1' and higher stay distinct.
    resident = backend_identity(
        {"MLP": "CHGNET", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    validate_request_backend(
        resident, {"MLP": "CHGNET", "DEVICE": "cpu:0"}, request_base_dir=str(tmp_path)
    )
    reverse = backend_identity(
        {"MLP": "CHGNET", "DEVICE": "cpu:0"}, base_dir=str(tmp_path)
    )
    assert reverse["device"] == "cpu"
    validate_request_backend(
        reverse, {"MLP": "CHGNET", "DEVICE": "cpu"}, request_base_dir=str(tmp_path)
    )
    # A genuinely different index is still a different device.
    assert server_module._resolve_backend_device("CHGNET", "cuda:1") == "cuda:1"
    assert server_module._resolve_backend_device("CHGNET", "cuda:0") == "cuda"


def test_md_rng_is_isolated_per_request(tmp_path: Path):
    # A resident MD request draws from the process-global numpy RNG. Without
    # per-request save/restore, request B advances that state so a repeated
    # request A produces different velocities. _execute_job must restore the RNG
    # so the A->B->A isolation sequence yields identical stochastic draws.
    import numpy as np

    socket_path = tmp_path / "server.sock"
    work = tmp_path / "work"
    work.mkdir()
    draws: list[float] = []

    def executor(workdir: str, *, calculator) -> None:
        draws.append(float(np.random.random()))
        print("Calculation completed.")

    _, thread = _start_server(socket_path, executor=executor)
    try:
        VPMDKClient(str(socket_path)).run(str(work), timeout=10.0)  # A
        VPMDKClient(str(socket_path)).run(str(work), timeout=10.0)  # B
        VPMDKClient(str(socket_path)).run(str(work), timeout=10.0)  # A again
        assert draws[0] == draws[1] == draws[2], draws
    finally:
        _stop_server(socket_path, thread)


def test_validation_warning_is_forwarded_to_the_client(tmp_path: Path):
    # A request value the resident's builder warns-and-ignores (e.g.
    # MATTERSIM_STRESS_WEIGHT=not-a-number) is accepted, but the warning is emitted
    # during canonicalization in validate_request_backend. That call must run
    # inside the client writer redirect so the warning reaches the client's log
    # stream, exactly as one-shot mode shows it -- not only the daemon log.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()
    (request_dir / "BCAR").write_text(
        "MLP=MATTERSIM\nMATTERSIM_STRESS_WEIGHT=not-a-number\n"
    )

    def executor(workdir: str, *, calculator) -> None:
        print("Calculation completed.")

    _, thread = _start_server(
        socket_path, tags={"MLP": "MATTERSIM", "DEVICE": "cpu"}, executor=executor
    )
    logs: list[str] = []
    try:
        result = VPMDKClient(str(socket_path)).run(
            str(request_dir), timeout=10.0, log_callback=logs.append
        )
        assert result["ok"] is True
        assert any("MATTERSIM_STRESS_WEIGHT" in line for line in logs), logs
    finally:
        _stop_server(socket_path, thread)


def test_foreign_backend_tag_does_not_crash_startup(tmp_path: Path):
    # A leftover strict-typed tag belonging to a DIFFERENT backend must not crash
    # resident startup. backend_identity strips it (the resident's own builder
    # ignores it, exactly as the one-shot builder does) instead of raising while
    # normalizing a foreign value -- which would abort `serve` AFTER a full model
    # load, with no one-shot analog (SERVER_MODE_SPEC 1-2). Mirrors the request
    # path's _strip_foreign_backend_tags so both sides ignore the foreign tag.
    for foreign, value in (
        # exclusive-prefix foreign tags -> dropped by _strip_foreign_backend_tags
        ("ALPHANET_PRECISION", "16"),
        ("NEQUIX_CAPACITY_MULTIPLIER", "auto"),
        ("EQNORM_VARIANT", "not-a-variant"),
        ("TACE_FIDELITY_IDX", "high"),
        # family-shared / non-prefixed foreign tags with a value malformed for
        # that tag's type. These are NOT stripped (family-shared exclusion), so
        # they would crash startup at _coerce_int_tag/_coerce_bool_tag unless the
        # resident canonicalization runs tolerant (dropping the uncoercible tag).
        ("SEVENNET_ENABLE_CUEQ", "maybe"),
        ("SEVENNET_ENABLE_FLASH", "perhaps"),
    ):
        resident = backend_identity(
            {"MLP": "CHGNET", foreign: value, "DEVICE": "cpu"},
            base_dir=str(tmp_path),
        )
        assert foreign not in resident["effective_configuration"]
        assert foreign not in resident["configuration"]

    # The bare Matlantis alias PRIORITY (non-prefixed, resolves to
    # MATLANTIS_PRIORITY via _coerce_int_tag) with a malformed value must also not
    # crash a non-Matlantis startup.
    chgnet_priority = backend_identity(
        {"MLP": "CHGNET", "PRIORITY": "high", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert "MATLANTIS_PRIORITY" not in chgnet_priority["effective_configuration"]

    # A FAIRCHEM v2 resident with a leftover v1-only tag carrying a bad value
    # (the finding's FAIRCHEM_V1_PREDICTOR=perhaps case) must start.
    fairchem = backend_identity(
        {"MLP": "FAIRCHEM_V2", "FAIRCHEM_V1_PREDICTOR": "perhaps", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert "FAIRCHEM_V1_PREDICTOR" not in fairchem["effective_configuration"]


def test_foreground_serve_releases_launch_directory(tmp_path: Path, monkeypatch):
    # Foreground `vpmdk serve` must chdir('/') like the --daemon path, so it does
    # not pin a launch/scratch directory that may be deleted later (which would
    # break every subsequent job's os.getcwd() in _working_directory).
    launch = tmp_path / "scratch"
    launch.mkdir()
    recorded: dict[str, str] = {}

    def fake_serve_forever(self, *, ready_callback=None):
        recorded["cwd"] = os.getcwd()

    monkeypatch.setattr(server_module.VPMDKServer, "serve_forever", fake_serve_forever)
    monkeypatch.setattr(
        server_module.VPMDKServer, "install_signal_handlers", lambda self: None
    )
    monkeypatch.setattr(
        server_module,
        "_load_backend_for_server",
        lambda workdir, bcar: (
            DummyCalculator(),
            {"MLP": "CHGNET", "DEVICE": "cpu"},
            str(launch),
        ),
    )
    args = SimpleNamespace(
        socket=str(tmp_path / "s.sock"),
        dir=str(launch),
        bcar=None,
        daemon=False,
        idle_timeout=0.0,
        log_file=None,
    )
    monkeypatch.chdir(launch)
    assert server_module.serve_cli(args) == 0
    assert recorded["cwd"] == "/", "foreground serve did not release its launch cwd"


def test_default_matgl_model_is_not_the_removed_classic_name():
    # Regression: the default MatGL/M3GNet model must be a name current matgl
    # (4.x) resolves. The classic "M3GNet-MP-2021.2.8-PES" was removed --
    # matgl.load_model() raises "Bad ... model name" -- so a default (no-MODEL)
    # run could not be built. matgl is not importable in this suite (conftest
    # mocks pymatgen), so assert against the known current M3GNet PES checkpoints
    # (matgl.get_available_pretrained_models() on matgl 4.x).
    assert vpmdk.DEFAULT_MATGL_MODEL != "M3GNet-MP-2021.2.8-PES"
    assert vpmdk.DEFAULT_MATGL_MODEL in {
        "M3GNet-PES-MatPES-PBE-2025.2",
        "M3GNet-PES-MatPES-r2SCAN-2025.2",
        "M3GNet-PES-ANI-1x-Subset",
    }


@pytest.mark.parametrize(
    ("mlp", "startup_tag", "request_tag", "value"),
    [
        ("MATLANTIS", "MATLANTIS_MODEL_VERSION", "MODEL_VERSION", "v1"),
        ("MATLANTIS", "MODEL", "MODEL_VERSION", "v1"),
        ("MATLANTIS", "MATLANTIS_PRIORITY", "PRIORITY", "50"),
        ("MATLANTIS", "MATLANTIS_CALC_MODE", "CALC_MODE", "PBE"),
        ("ALPHANET", "ALPHANET_PRECISION", "ALPHANET_DTYPE", "32"),
        ("NEQUIX", "NEQUIX_USE_KERNEL", "NEQUIX_KERNEL", "true"),
        ("NEQUIX", "NEQUIX_USE_COMPILE", "NEQUIX_COMPILE", "false"),
        ("UPET", "UPET_NEIGHBORLIST_DEVICE", "UPET_NL_DEVICE", "cpu"),
        ("TACE", "TACE_FIDELITY_IDX", "TACE_LEVEL", "1"),
        (
            "CHGNET",
            "CHGNET_GRAPH_CONVERTER_ALGORITHM",
            "GRAPH_CONVERTER",
            "fast",
        ),
        (
            "MATRIS",
            "GRAPH_CONVERTER_ALGORITHM",
            "MATRIS_GRAPH_CONVERTER",
            "legacy",
        ),
    ],
)
def test_backend_configuration_aliases_compare_by_effective_option(
    tmp_path: Path,
    mlp: str,
    startup_tag: str,
    request_tag: str,
    value: str,
):
    resident = backend_identity(
        {"MLP": mlp, startup_tag: value, "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    validate_request_backend(
        resident,
        {"MLP": mlp, request_tag: value, "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )

    alias_resident = backend_identity(
        {"MLP": mlp, request_tag: value, "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    validate_request_backend(
        alias_resident,
        {"MLP": mlp, startup_tag: value, "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_backend_configuration_alias_with_different_value_is_rejected(tmp_path: Path):
    resident = backend_identity(
        {
            "MLP": "UPET",
            "UPET_NEIGHBORLIST_DEVICE": "cpu",
            "DEVICE": "cuda",
        },
        base_dir=str(tmp_path),
    )

    with pytest.raises(BackendConfigurationMismatch, match="UPET_NEIGHBORLIST_DEVICE"):
        validate_request_backend(
            resident,
            {"UPET_NL_DEVICE": "model"},
            request_base_dir=str(tmp_path),
        )


@pytest.mark.parametrize(
    ("resident_mlp", "request_mlp"),
    [
        ("MATGL", "M3GNET"),
        ("M3GNET", "MATGL"),
        ("FAIRCHEM", "FAIRCHEM_V2"),
        ("FAIRCHEM_V2", "ESEN"),
        ("ESEN", "FAIRCHEM"),
    ],
)
def test_equivalent_backend_names_share_resident_identity(
    tmp_path: Path,
    resident_mlp: str,
    request_mlp: str,
):
    resident = backend_identity(
        {"MLP": resident_mlp, "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    validate_request_backend(
        resident,
        {"MLP": request_mlp, "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


@pytest.mark.parametrize(
    ("mlp", "canonical_model", "alias"),
    [
        ("EQNORM", vpmdk.DEFAULT_EQNORM_MODEL, "eqnorm"),
        ("ALPHANET", vpmdk.DEFAULT_ALPHANET_MODEL, "matpes"),
        ("HIENET", vpmdk.DEFAULT_HIENET_MODEL, "v3"),
        ("MATRIS", vpmdk.DEFAULT_MATRIS_MODEL, vpmdk.DEFAULT_MATRIS_MODEL.upper()),
    ],
)
def test_named_model_aliases_compare_by_canonical_model_identity(
    tmp_path: Path,
    mlp: str,
    canonical_model: str,
    alias: str,
):
    resident = backend_identity(
        {"MLP": mlp, "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    assert resident["model"] == canonical_model
    validate_request_backend(
        resident,
        {"MLP": mlp, "MODEL": alias, "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )

    alias_resident = backend_identity(
        {"MLP": mlp, "MODEL": alias, "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert alias_resident["model"] == canonical_model
    validate_request_backend(
        alias_resident,
        {"MLP": mlp, "MODEL": canonical_model, "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_nequix_named_models_compare_case_insensitively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeNequixCalculator:
        URLS = {vpmdk.DEFAULT_NEQUIX_MODEL: "https://example.invalid/model"}

    monkeypatch.setattr(vpmdk, "NequixCalculator", FakeNequixCalculator)
    resident = backend_identity(
        {"MLP": "NEQUIX", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    validate_request_backend(
        resident,
        {"MODEL": vpmdk.DEFAULT_NEQUIX_MODEL.upper()},
        request_base_dir=str(tmp_path),
    )


def test_named_model_like_missing_checkpoint_is_rejected_as_a_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="EQNORM MODEL path not found"):
        backend_identity(
            {"MLP": "EQNORM", "MODEL": "models/eqnorm", "DEVICE": "cpu"},
            base_dir=str(tmp_path),
        )


@pytest.mark.parametrize(
    ("mlp", "request_tags"),
    [
        (
            "MATLANTIS",
            {
                "MODEL_VERSION": "v8.0.0",
                "PRIORITY": "50.0",
                "CALC_MODE": "pbe",
            },
        ),
        (
            "ORB",
            {
                "ORB_MODEL": vpmdk.DEFAULT_ORB_MODEL,
                "ORB_PRECISION": "float32-high",
            },
        ),
        (
            "EQNORM",
            {"EQNORM_COMPILE": "off", "EQNORM_VARIANT": "eqnorm-mptrj"},
        ),
        ("MATRIS", {"MATRIS_TASK": "EFS"}),
        ("ALPHANET", {"ALPHANET_DTYPE": "fp32"}),
        ("HIENET", {"HIENET_FILE_TYPE": "CHECKPOINT"}),
        (
            "NEQUIX",
            {
                "NEQUIX_BACKEND": "JAX",
                "NEQUIX_KERNEL": "0",
                "NEQUIX_COMPILE": "no",
                "NEQUIX_CAPACITY_MULTIPLIER": "1.10",
            },
        ),
        ("SEVENNET", {"SEVENNET_FILE_TYPE": "CHECKPOINT"}),
        (
            "FLASHTP",
            {
                "SEVENNET_FILE_TYPE": "checkpoint",
                "SEVENNET_ENABLE_CUEQ": "false",
                "SEVENNET_ENABLE_FLASH": "1",
                "SEVENNET_ENABLE_OEQ": "off",
            },
        ),
        (
            "EQUFLASH",
            {
                "SEVENNET_FILE_TYPE": "CHECKPOINT",
                "SEVENNET_ENABLE_CUEQ": "0",
                "SEVENNET_ENABLE_FLASH": "true",
                "SEVENNET_ENABLE_OEQ": "no",
            },
        ),
        (
            "FAIRCHEM",
            {
                "FAIRCHEM_TASK": vpmdk.DEFAULT_FAIRCHEM_TASK,
                "FAIRCHEM_INFERENCE_SETTINGS": "default",
            },
        ),
        ("FAIRCHEM_V1", {"FAIRCHEM_V1_PREDICTOR": "false"}),
    ],
)
def test_explicit_backend_defaults_match_omitted_startup_values(
    tmp_path: Path,
    mlp: str,
    request_tags: dict[str, str],
):
    resident = backend_identity(
        {"MLP": mlp, "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    validate_request_backend(
        resident,
        {"MLP": mlp, "DEVICE": "cpu", **request_tags},
        request_base_dir=str(tmp_path),
    )


def test_sevennet_enabled_accelerator_records_implied_false_flags(tmp_path: Path):
    # Enabling one SevenNet accelerator forces the other two off in the builder,
    # so the resident's effective configuration must record those implied False
    # flags -- otherwise an equivalent request that spells out
    # SEVENNET_ENABLE_FLASH=false is rejected as request=False, server=None.
    resident = backend_identity(
        {"MLP": "SEVENNET", "SEVENNET_ENABLE_CUEQ": "true", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    effective = resident["effective_configuration"]
    assert effective["SEVENNET_ENABLE_CUEQ"] is True
    assert effective["SEVENNET_ENABLE_FLASH"] is False
    assert effective["SEVENNET_ENABLE_OEQ"] is False

    # The equivalent request spelling out the implied False is accepted.
    validate_request_backend(
        resident,
        {
            "MLP": "SEVENNET",
            "SEVENNET_ENABLE_CUEQ": "true",
            "SEVENNET_ENABLE_FLASH": "false",
            "DEVICE": "cpu",
        },
        request_base_dir=str(tmp_path),
    )


def test_sevennet_without_accelerators_records_resolved_false_flags(tmp_path: Path):
    # A plain SevenNet resident (no accelerator tags) records the RESOLVED flags
    # as False, not "unset". This keeps a no-op disable request equivalent to the
    # resident (both select the non-accelerated calculator) while an enabling
    # request still differs.
    resident = backend_identity(
        {"MLP": "SEVENNET", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )

    effective = resident["effective_configuration"]
    assert effective["SEVENNET_ENABLE_CUEQ"] is False
    assert effective["SEVENNET_ENABLE_FLASH"] is False
    assert effective["SEVENNET_ENABLE_OEQ"] is False

    # A request that explicitly spells out a no-op disable is accepted (it selects
    # the same plain calculator), instead of being rejected as request=False vs
    # server=None.
    validate_request_backend(
        resident,
        {"MLP": "SEVENNET", "SEVENNET_ENABLE_CUEQ": "0", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )

    # An enabling request still differs from the plain resident.
    with pytest.raises(BackendConfigurationMismatch, match="SEVENNET_ENABLE_CUEQ"):
        validate_request_backend(
            resident,
            {"MLP": "SEVENNET", "SEVENNET_ENABLE_CUEQ": "true", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


@pytest.mark.parametrize(
    ("mlp", "tag", "default_value"),
    [
        ("ORB", "ORB_PRECISION", "float32-high"),
        ("MATRIS", "MATRIS_TASK", "efs"),
        ("SEVENNET", "SEVENNET_FILE_TYPE", "checkpoint"),
    ],
)
def test_blank_optional_tag_uses_builder_default(
    tmp_path: Path, mlp: str, tag: str, default_value: str
):
    # A present-but-empty optional tag means "use the builder default" (the
    # builder defaults it via `get(tag) or DEFAULT`), so the effective config
    # must record the default rather than "", and a request naming the actual
    # default must be accepted instead of rejected as exit 5.
    resident = backend_identity(
        {"MLP": mlp, tag: "", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )

    assert resident["effective_configuration"][tag] == default_value

    validate_request_backend(
        resident,
        {"MLP": mlp, tag: default_value, "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_blank_coerce_tag_is_rejected_like_the_builder(tmp_path: Path):
    # A blank boolean tag (whether the canonical name ORB_COMPILE= or the alias
    # NEQUIX_KERNEL=) is rejected by the one-shot builder (_coerce_bool_tag('')
    # raises), so the server must reject it too rather than silently accept the
    # request -- keeping server/one-shot acceptance equivalent.
    orb_resident = backend_identity(
        {"MLP": "ORB", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(
            orb_resident,
            {"MLP": "ORB", "ORB_COMPILE": "", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )

    nequix_resident = backend_identity(
        {"MLP": "NEQUIX", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(
            nequix_resident,
            {"MLP": "NEQUIX", "NEQUIX_KERNEL": "", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_nequix_capacity_multiplier_rejects_malformed_value(tmp_path: Path):
    # The Nequix builder uses strict float(); the server must reject a malformed
    # value like "junk1.1tail" too, not leniently parse it to 1.1 and accept a
    # request one-shot mode rejects.
    resident = backend_identity(
        {"MLP": "NEQUIX", "NEQUIX_CAPACITY_MULTIPLIER": "1.1", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(
            resident,
            {
                "MLP": "NEQUIX",
                "NEQUIX_CAPACITY_MULTIPLIER": "junk1.1tail",
                "DEVICE": "cpu",
            },
            request_base_dir=str(tmp_path),
        )


def test_overflowing_integer_tag_is_a_backend_mismatch_not_calculation_error(
    tmp_path: Path,
):
    # A non-finite integer construction tag (TACE_FIDELITY_IDX=inf / 1e400) makes
    # _coerce_int_tag hit int(float("inf")), which raises OverflowError. The
    # request-validation exit-5 guard catches ValueError, so OverflowError must be
    # normalized to ValueError -- otherwise it leaks out as calculation_error
    # (exit 2) instead of BackendConfigurationMismatch (exit 5), and diverges from
    # "=nan" (which already yields exit 5). All malformed integer values must
    # classify identically.
    resident = backend_identity(
        {"MLP": "TACE", "TACE_FIDELITY_IDX": "1", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    for value in ("inf", "-inf", "1e400", "nan"):
        with pytest.raises(BackendConfigurationMismatch):
            validate_request_backend(
                resident,
                {"MLP": "TACE", "TACE_FIDELITY_IDX": value, "DEVICE": "cpu"},
                request_base_dir=str(tmp_path),
            )


def test_blank_string_tag_is_still_omitted(tmp_path: Path):
    # A blank string tag whose builder defaults it via ``or`` (SEVENNET_FILE_TYPE=)
    # is still treated as omitted, matching the builder, and accepted.
    resident = backend_identity(
        {"MLP": "SEVENNET", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    validate_request_backend(
        resident,
        {"MLP": "SEVENNET", "SEVENNET_FILE_TYPE": "", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_blank_eqnorm_variant_startup_infers_variant(tmp_path: Path):
    # A present-but-blank EQNORM_VARIANT= in the startup BCAR is treated as unset
    # by the builder (which infers the variant from the model), so the effective
    # config must infer it too -- otherwise the resident advertises no variant and
    # a request naming the identical inferred variant is wrongly rejected (exit 5).
    resident = backend_identity(
        {"MLP": "EQNORM", "EQNORM_VARIANT": "", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    inferred = resident["effective_configuration"].get("EQNORM_VARIANT")
    assert inferred, "blank EQNORM_VARIANT= suppressed variant inference"
    validate_request_backend(
        resident,
        {"MLP": "EQNORM", "EQNORM_VARIANT": inferred, "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_blank_orb_model_startup_reports_default_model(tmp_path: Path):
    # A present-but-blank ORB_MODEL= is not a model name: the ORB builder resolves
    # it to DEFAULT_ORB_MODEL (`get("ORB_MODEL") or DEFAULT_ORB_MODEL`). The
    # resident's advertised `status` model must reflect that default, not "".
    resident = backend_identity(
        {"MLP": "ORB", "ORB_MODEL": "", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    assert resident["model"] == vpmdk.DEFAULT_ORB_MODEL
    # An explicit ORB_MODEL is still advertised verbatim.
    explicit = backend_identity(
        {"MLP": "ORB", "ORB_MODEL": "orb-v3-direct-20-omat", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert explicit["model"] == "orb-v3-direct-20-omat"


def test_status_timeout_maps_to_unreachable_exit_code():
    # `status` exposes no user --timeout and the spec allows only exit 0 (alive)
    # or exit 3 (unreachable). Its internal deadline expiring is an unresponsive
    # server -> exit 3, NOT the run/stop --timeout code 4.
    import argparse

    original = lightweight_client_module.VPMDKClient.status
    try:
        lightweight_client_module.VPMDKClient.status = (
            lambda self, *, timeout=2.0: (_ for _ in ()).throw(
                lightweight_client_module.ClientTimeoutError("timed out")
            )
        )
        args = argparse.Namespace(
            command="status", socket="/tmp/nonexistent-vpmdk.sock", json=False
        )
        assert lightweight_client_module.client_cli(args) == 3
    finally:
        lightweight_client_module.VPMDKClient.status = original


def test_status_and_stop_map_all_failures_to_their_spec_exit_codes():
    # SERVER_MODE_SPEC 2.3: `status` is only exit 0/3. 2.4: `stop` is only 0/3/4.
    # A server-side internal_error (RemoteCalculationError) or protocol_error
    # (ProtocolError) reaching these commands must map to exit 3 -- NOT the
    # calculation-failure code 2 -- while a client stop timeout stays exit 4.
    import argparse

    lcm = lightweight_client_module
    orig_status, orig_stop = lcm.VPMDKClient.status, lcm.VPMDKClient.stop

    def raiser(exc):
        def _raise(self, *a, **k):
            raise exc

        return _raise

    try:
        for exc_cls in (lcm.RemoteCalculationError, lcm.ProtocolError):
            lcm.VPMDKClient.status = raiser(exc_cls("server defect"))
            status_args = argparse.Namespace(
                command="status", socket="/tmp/x.sock", json=False
            )
            assert lcm.client_cli(status_args) == 3

            lcm.VPMDKClient.stop = raiser(exc_cls("server defect"))
            stop_args = argparse.Namespace(
                command="stop", socket="/tmp/x.sock", force=False, timeout=5.0
            )
            assert lcm.client_cli(stop_args) == 3

        # A client-side stop timeout is still exit 4.
        lcm.VPMDKClient.stop = raiser(lcm.ClientTimeoutError("shutdown wait timed out"))
        stop_args = argparse.Namespace(
            command="stop", socket="/tmp/x.sock", force=False, timeout=5.0
        )
        assert lcm.client_cli(stop_args) == 4
    finally:
        lcm.VPMDKClient.status, lcm.VPMDKClient.stop = orig_status, orig_stop


def test_malformed_status_backend_is_a_protocol_error_not_a_traceback():
    # A valid JSON status whose `backend` is not an object -- e.g.
    # {"event":"status","backend":[1]} -- must be rejected as a ProtocolError
    # (-> exit 3, SERVER_MODE_SPEC 2.3), never let the formatter raise an uncaught
    # AttributeError that escapes client_cli's excepts as an off-contract traceback.
    import argparse

    lcm = lightweight_client_module

    # 1. _format_status tolerates a non-Mapping backend (defense-in-depth).
    for bad in ([1], "x", 5):
        out = lcm._format_status({"event": "status", "state": "idle", "backend": bad})
        assert "MLP=None" in out

    # 2. VPMDKClient.status() raises ProtocolError for a non-object backend, and
    #    both the formatted and --json status paths map it to exit 3.
    def fake_request(self, request, *, timeout):
        yield {"event": "status", "state": "idle", "backend": [1]}

    orig_request = lcm.VPMDKClient._request
    try:
        lcm.VPMDKClient._request = fake_request
        with pytest.raises(lcm.ProtocolError):
            client = lcm.VPMDKClient.__new__(lcm.VPMDKClient)
            client.socket_path = "/tmp/x.sock"
            client.status()
        assert issubclass(lcm.ProtocolError, lcm.VPMDKClientError)
        for use_json in (False, True):
            args = argparse.Namespace(command="status", socket="/tmp/x.sock", json=use_json)
            assert lcm.client_cli(args) == 3

        # A present-but-null or absent backend is NOT malformed: it formats fine.
        def null_backend(self, request, *, timeout):
            yield {"event": "status", "state": "idle", "backend": None,
                   "pid": 1, "uptime_s": 1.0, "protocol": 1, "vpmdk_version": "x"}

        lcm.VPMDKClient._request = null_backend
        args = argparse.Namespace(command="status", socket="/tmp/x.sock", json=False)
        assert lcm.client_cli(args) == 0
    finally:
        lcm.VPMDKClient._request = orig_request


def test_malformed_status_uptime_is_a_protocol_error_not_a_traceback():
    # Sibling of the backend guard: _format_status coerces uptime_s with float(),
    # so a present non-numeric value (null -> TypeError, "abc" -> ValueError, and a
    # JSON bool) from a non-conforming peer must map to exit 3, never an uncaught
    # exception escaping client_cli as an off-contract traceback (SERVER_MODE_SPEC
    # 2.3: status is only exit 0 or 3).
    import argparse

    lcm = lightweight_client_module

    # 1. _format_status never raises on a bad uptime_s (defense-in-depth).
    for bad in (None, "abc", [1], {}):
        out = lcm._format_status(
            {"event": "status", "state": "idle", "backend": {}, "uptime_s": bad}
        )
        assert "Uptime: 0.0 s" in out

    orig_request = lcm.VPMDKClient._request
    try:
        # 2. A present non-numeric uptime_s -> ProtocolError -> exit 3 (both paths).
        for bad in ("abc", True, [1], {"x": 1}):
            def bad_uptime(self, request, *, timeout, _v=bad):
                yield {"event": "status", "state": "idle", "backend": {}, "uptime_s": _v}

            lcm.VPMDKClient._request = bad_uptime
            with pytest.raises(lcm.ProtocolError):
                client = lcm.VPMDKClient.__new__(lcm.VPMDKClient)
                client.socket_path = "/tmp/x.sock"
                client.status()
            for use_json in (False, True):
                args = argparse.Namespace(command="status", socket="/tmp/x.sock", json=use_json)
                assert lcm.client_cli(args) == 3

        # 3. A present-but-null or absent uptime_s is NOT malformed (formats fine).
        def null_uptime(self, request, *, timeout):
            yield {"event": "status", "state": "idle",
                   "backend": {"mlp": "CHGNET", "model": "m", "device": "cpu"},
                   "uptime_s": None, "pid": 1, "protocol": 1, "vpmdk_version": "x"}

        lcm.VPMDKClient._request = null_uptime
        assert lcm.client_cli(
            argparse.Namespace(command="status", socket="/tmp/x.sock", json=False)
        ) == 0
    finally:
        lcm.VPMDKClient._request = orig_request


def test_absurd_request_timeout_is_rejected_before_settimeout_overflow():
    # A finite but absurdly large --timeout (>= ~1e10 s) would reach
    # socket.settimeout and raise OverflowError -- an ArithmeticError none of
    # client_cli's except clauses catch, giving a traceback + off-contract exit 1
    # and a leaked fd. It must be rejected up front as invalid input (ValueError
    # -> exit 1 with a clean message, before any connection is opened).
    import argparse

    too_big = lightweight_client_module._MAX_REQUEST_TIMEOUT + 1
    for bad in (1e12, 1e10, float("inf"), -1.0, too_big):
        with pytest.raises(ValueError, match="timeout must be"):
            lightweight_client_module._validate_request_timeout(bad)
    for ok in (0.0, 5.0, 3600.0, lightweight_client_module._MAX_REQUEST_TIMEOUT):
        lightweight_client_module._validate_request_timeout(ok)

    # End-to-end via client_cli: exit 1 (ValueError), not an OverflowError
    # traceback. Validation happens before connect, so the missing socket is
    # never reached.
    args = argparse.Namespace(
        command="run", socket="/tmp/nonexistent-vpmdk-timeout.sock", dir=".", timeout=1e12
    )
    assert lightweight_client_module.client_cli(args) == 1


def test_blank_lenient_float_tag_uses_builder_default(tmp_path: Path):
    # MATTERSIM_STRESS_WEIGHT / GRACE_PAD_NEIGHBORS_FRACTION / GRACE_MIN_DIST are
    # float tags, but their builders parse them with _parse_optional_float, which
    # maps a blank to None -> "use the constructor default" (a common template
    # pattern). Membership in the blank-reject set is by *builder behaviour*, not
    # value category, so the server must treat these blanks as omitted rather than
    # reject them:
    #  (a) a resident started with a blank value must not crash in backend_identity
    #      (previously the coerce-float path raised, killing server startup), and
    #  (b) a request carrying the blank must be accepted, matching one-shot mode.
    for mlp, tag in (
        ("MATTERSIM", "MATTERSIM_STRESS_WEIGHT"),
        ("GRACE", "GRACE_PAD_NEIGHBORS_FRACTION"),
        ("GRACE", "GRACE_MIN_DIST"),
    ):
        resident = backend_identity(
            {"MLP": mlp, tag: "", "DEVICE": "cpu"}, base_dir=str(tmp_path)
        )
        assert tag not in resident["effective_configuration"]
        validate_request_backend(
            resident,
            {"MLP": mlp, tag: "", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_blank_matlantis_priority_uses_builder_default(tmp_path: Path):
    # MATLANTIS_PRIORITY is an int tag, but its builder defaults a blank via
    # ``get(...) or PRIORITY``, so the server must omit a blank rather than run it
    # through _coerce_int_tag (which would raise and reject a request one-shot mode
    # accepts).
    resident = backend_identity(
        {"MLP": "MATLANTIS", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    validate_request_backend(
        resident,
        {"MLP": "MATLANTIS", "MATLANTIS_PRIORITY": "", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_blank_reject_string_tag_is_rejected_like_the_builder(tmp_path: Path):
    # NEQUIX_BACKEND / HIENET_FILE_TYPE / GRAPH_CONVERTER_ALGORITHM are string tags
    # whose builders pass the value straight into a _normalize_* that raises on any
    # unrecognized string (including ""). Unlike ``or``-defaulted string tags, a
    # blank here is NOT "use default": the one-shot builder rejects it, so the
    # server must not omit it and silently reuse the resident's real value.
    nequix_resident = backend_identity(
        {"MLP": "NEQUIX", "NEQUIX_BACKEND": "jax", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(
            nequix_resident,
            {"MLP": "NEQUIX", "NEQUIX_BACKEND": "", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )

    hienet_resident = backend_identity(
        {"MLP": "HIENET", "HIENET_FILE_TYPE": "checkpoint", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(
            hienet_resident,
            {"MLP": "HIENET", "HIENET_FILE_TYPE": "", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_non_numeric_lenient_float_uses_builder_default(tmp_path: Path):
    # A NON-blank but non-numeric value on a lenient-float tag (e.g.
    # GRACE_MIN_DIST=none / auto / default) is mapped to None by the builder's
    # _parse_optional_float, which then DROPS the kwarg and uses the constructor
    # default. The server must mirror that (treat it as omitted), not raise:
    #  (a) a resident started with such a value must not crash backend_identity
    #      AFTER loading the model, and
    #  (b) a request carrying it must be accepted, matching one-shot mode.
    for mlp, tag in (
        ("GRACE", "GRACE_MIN_DIST"),
        ("MATTERSIM", "MATTERSIM_STRESS_WEIGHT"),
        ("GRACE", "GRACE_PAD_NEIGHBORS_FRACTION"),
    ):
        for value in ("none", "auto", "default"):
            resident = backend_identity(
                {"MLP": mlp, tag: value, "DEVICE": "cpu"}, base_dir=str(tmp_path)
            )
            assert tag not in resident["effective_configuration"]
            validate_request_backend(
                resident,
                {"MLP": mlp, tag: value, "DEVICE": "cpu"},
                request_base_dir=str(tmp_path),
            )
    # A valid numeric value is still canonicalized (not dropped).
    numeric = backend_identity(
        {"MLP": "GRACE", "GRACE_MIN_DIST": "2.5", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert numeric["effective_configuration"]["GRACE_MIN_DIST"] == 2.5
    # NEQUIX_CAPACITY_MULTIPLIER stays strict (builder uses float()), so a
    # non-numeric value there is still rejected, not silently dropped.
    strict = backend_identity(
        {"MLP": "NEQUIX", "NEQUIX_CAPACITY_MULTIPLIER": "1.1", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(
            strict,
            {"MLP": "NEQUIX", "NEQUIX_CAPACITY_MULTIPLIER": "none", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_or_strict_alias_blank_secondary_is_rejected_like_the_builder(tmp_path: Path):
    # For the ``or``-strict alias tags the builder resolves
    # ``get(primary) or get(secondary)`` and feeds the result to a strict coercer.
    # A blank SECONDARY alias with the primary ABSENT resolves to "" (``None or
    # ''``) and the builder raises, so the server must keep and reject it, not
    # omit it and silently reuse the resident. A blank on the PRIMARY instead
    # falls through ``or`` to the secondary/None and defaults, so it stays omitted.
    cases = (
        ("ALPHANET", "ALPHANET_PRECISION", "ALPHANET_DTYPE", "32"),
        ("MATLANTIS", "MATLANTIS_PRIORITY", "PRIORITY", "5"),
        ("UPET", "UPET_NEIGHBORLIST_DEVICE", "UPET_NL_DEVICE", "cpu"),
    )
    for mlp, primary, secondary, resident_value in cases:
        resident = backend_identity(
            {"MLP": mlp, primary: resident_value, "DEVICE": "cpu"},
            base_dir=str(tmp_path),
        )
        # absent primary + blank secondary -> "" -> builder raises -> reject
        with pytest.raises(BackendConfigurationMismatch):
            validate_request_backend(
                resident,
                {"MLP": mlp, secondary: "", "DEVICE": "cpu"},
                request_base_dir=str(tmp_path),
            )
        # blank primary (a non-final ``or`` operand) still defaults: the builder
        # resolves ``'' or None`` -> None -> default, so the server omits it and
        # the request is accepted against a resident running that same default.
        default_resident = backend_identity(
            {"MLP": mlp, "DEVICE": "cpu"}, base_dir=str(tmp_path)
        )
        validate_request_backend(
            default_resident,
            {"MLP": mlp, primary: "", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_blank_sevennet_modal_is_rejected_like_the_builder(tmp_path: Path):
    # SevenNet passes SEVENNET_MODAL through with ``if modal is not None`` (no
    # ``or``/truthy gate), so a blank SEVENNET_MODAL= is an explicit empty-modal
    # selection, not "use default". The server must therefore keep it and compare
    # it (a mismatch against a resident modal), not omit it and silently reuse the
    # resident's modal -- which would run the request on a different modal than
    # its BCAR encodes.
    resident = backend_identity(
        {"MLP": "SEVENNET", "SEVENNET_MODAL": "mpa", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(
            resident,
            {"MLP": "SEVENNET", "SEVENNET_MODAL": "", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_malformed_request_bcar_is_reported_as_input_error(tmp_path: Path):
    # A malformed/unreadable request BCAR is invalid user input: the server must
    # report it as input_error (RemoteInputError / exit 1), not calculation_error.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "bad-bcar"
    request_dir.mkdir()
    (request_dir / "BCAR").write_bytes(b"MLP = CHGNET\n\xff\xfe not utf-8\n")
    _, thread = _start_server(socket_path)
    try:
        with pytest.raises(RemoteInputError):
            VPMDKClient(str(socket_path)).run(str(request_dir))
    finally:
        _stop_server(socket_path, thread)


def test_codeless_failed_done_maps_to_calculation_error():
    # SERVER_MODE_SPEC 3.3 documents a terminal failure as
    # {"event":"done","ok":false,"error":...} with NO code field. The client must
    # treat a failed done whose code is absent (or an unrecognized future code) as
    # a calculation failure (exit 2), not a ProtocolError/ServerConnectionError
    # (exit 3), which would misreport a real backend failure as a lost connection.
    with pytest.raises(RemoteCalculationError):
        VPMDKClient._raise_event_error(
            {"event": "done", "ok": False, "error": "OOM ...", "traceback": "tb"}
        )
    with pytest.raises(RemoteCalculationError):
        VPMDKClient._raise_event_error(
            {"event": "done", "ok": False, "error": "x", "code": "future_code"}
        )
    # Known codes still map to their specific exceptions.
    with pytest.raises(RemoteInputError):
        VPMDKClient._raise_event_error(
            {"event": "done", "ok": False, "error": "x", "code": "input_error"}
        )


def test_protocol_error_code_maps_to_connection_error_exit_three():
    # The server's `protocol_error` code (version skew, malformed/oversized frame)
    # is a protocol/connection failure, not a calculation failure: the client must
    # raise ProtocolError (a ServerConnectionError -> exit 3), not
    # RemoteCalculationError (exit 2). exit 3 is also the only non-alive result the
    # status contract (SERVER_MODE_SPEC 2.3) permits.
    with pytest.raises(lightweight_client_module.ProtocolError) as excinfo:
        VPMDKClient._raise_event_error(
            {
                "event": "error",
                "code": "protocol_error",
                "error": "Unsupported protocol version 2; expected 1",
            }
        )
    assert isinstance(excinfo.value, ServerConnectionError)


def test_non_string_remote_traceback_keeps_the_documented_exit_code():
    # `traceback` is a diagnostic-only field that client_cli prints with
    # .rstrip(). A malformed/version-skewed peer sending a TRUTHY NON-string
    # (e.g. a list of frames, a number, or true) previously stored it unchanged,
    # so .rstrip() raised an uncaught AttributeError -- a Python traceback plus an
    # off-contract exit instead of the exit code documented for the failure.
    # RemoteCalculationError.__init__ now accepts only a real string, which covers
    # every construction site, so each code still maps to its documented exit.
    import argparse

    lcm = lightweight_client_module

    # The constructor drops any non-string and preserves a genuine string.
    for bad in (["frame", "frame"], {"a": 1}, 12345, True, b"bytes"):
        assert lcm.RemoteCalculationError("m", traceback=bad).traceback is None
    assert lcm.RemoteCalculationError("m", traceback="Trace\n").traceback == "Trace\n"
    assert lcm.RemoteCalculationError("m").traceback is None

    expected_exit = {
        "calculation_error": 2,
        "input_error": 1,
        "backend_mismatch": 5,
        "internal_error": 2,
        "an_unknown_future_code": 2,
        None: 2,
    }
    orig_request = lcm.VPMDKClient._request
    try:
        for code, want in expected_exit.items():
            for bad in (["frame", "frame"], {"a": 1}, 12345, True):
                def bad_traceback(self, request, *, timeout, _c=code, _t=bad):
                    event = {"event": "done", "ok": False, "error": "boom", "traceback": _t}
                    if _c is not None:
                        event["code"] = _c
                    yield event

                lcm.VPMDKClient._request = bad_traceback
                args = argparse.Namespace(
                    command="run", dir=".", socket="/tmp/x.sock", timeout=0.0
                )
                assert lcm.client_cli(args) == want, (code, bad)
    finally:
        lcm.VPMDKClient._request = orig_request


def test_no_malformed_peer_event_can_crash_the_client_or_escape_the_exit_contract():
    # Class-level lock, not a single-field guard. Every field the client reads
    # from a peer event has produced a defect at least once (backend -> .get,
    # uptime_s -> float(), traceback -> .rstrip()), always the same shape: a
    # value of an unexpected TYPE reaching a type-specific operation, yielding an
    # uncaught exception and an off-contract exit. This asserts the invariant
    # directly: for ANY malformed event, client_cli must return one of the
    # documented exit codes and never raise. Cheaper than chasing each field.
    import argparse

    lcm = lightweight_client_module
    documented_exits = {0, 1, 2, 3, 4, 5}
    weird = [None, True, False, 0, 1, -1, 1.5, float("nan"), float("inf"), "", "x",
             [], [1, 2], {}, {"a": 1}]

    def exit_code_for(command, event):
        def _request(self, request, *, timeout):
            yield event

        if command == "run":
            args = argparse.Namespace(command="run", dir=".", socket="/s", timeout=0.0)
        elif command == "status":
            args = argparse.Namespace(command="status", socket="/s", json=False)
        else:
            args = argparse.Namespace(command="stop", socket="/s", force=False, timeout=0.0)
        original = lcm.VPMDKClient._request
        try:
            lcm.VPMDKClient._request = _request
            return lcm.client_cli(args)
        finally:
            lcm.VPMDKClient._request = original

    # Terminal `done` events: every error code x malformed traceback/error value.
    for code in (None, "calculation_error", "input_error", "backend_mismatch",
                 "internal_error", "protocol_error", "server_stopping", "future_code"):
        for value in weird:
            event = {"event": "done", "ok": False, "error": value, "traceback": value}
            if code is not None:
                event["code"] = code
            assert exit_code_for("run", event) in documented_exits, (code, value)

    # `status` events: every field the formatter touches, one malformed at a time.
    for field in ("backend", "uptime_s", "state", "pid", "jobs_completed",
                  "jobs_failed", "queue_length", "protocol", "vpmdk_version",
                  "current_workdir"):
        for value in weird:
            event = {"event": "status", "state": "idle", "backend": {}, "uptime_s": 1.0}
            event[field] = value
            assert exit_code_for("status", event) in documented_exits, (field, value)

    # `stop` acknowledgements with a malformed ok/error payload.
    for value in weird:
        for event in ({"event": "done", "ok": value},
                      {"event": "error", "code": "x", "error": value}):
            assert exit_code_for("stop", event) in documented_exits, (value, event)


def test_server_supplied_text_with_surrogates_keeps_the_exit_contract():
    # Filesystem paths decode with errors="surrogateescape", and the protocol
    # transports those code points intact. Writing them to a strict UTF-8 stdout
    # raises UnicodeEncodeError -- a ValueError -- so client_cli's trailing
    # `except ValueError` turned a SUCCESSFUL calculation into exit 1 and swallowed
    # the `Calculation completed.` marker SERVER_MODE_SPEC 1.3 requires. Output
    # encoding must never change the exit code.
    import argparse

    lcm = lightweight_client_module
    surrogate_path = "/data/caf\udce9/run"

    def run_with_streams(command, generator, **extra):
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
        stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
        args = argparse.Namespace(command=command, socket="/s", **extra)
        original_request = lcm.VPMDKClient._request
        saved = (sys.stdout, sys.stderr)
        try:
            lcm.VPMDKClient._request = generator
            sys.stdout, sys.stderr = stdout, stderr
            code = lcm.client_cli(args)
        finally:
            sys.stdout, sys.stderr = saved
            lcm.VPMDKClient._request = original_request
        stdout.flush()
        stderr.flush()
        return code, stdout.buffer.getvalue(), stderr.buffer.getvalue()

    # A successful run whose log line carries the surrogate path.
    def successful_run(self, request, *, timeout):
        yield {"event": "log", "line": f"Reading POSCAR from {surrogate_path}"}
        yield {"event": "done", "ok": True, "elapsed_s": 1.0}

    code, out, _ = run_with_streams("run", successful_run, dir=".", timeout=0.0)
    assert code == 0
    assert b"Calculation completed." in out
    # The original filesystem byte is written back out, not lost or escaped away.
    assert b"\xe9" in out

    # A failure whose message and traceback carry surrogates keeps exit 2.
    def failed_run(self, request, *, timeout):
        yield {
            "event": "done",
            "ok": False,
            "code": "calculation_error",
            "error": f"failed in {surrogate_path}",
            "traceback": f"Traceback: {surrogate_path}\n",
        }

    code, _, err = run_with_streams("run", failed_run, dir=".", timeout=0.0)
    assert code == 2
    assert b"\xe9" in err

    # Both status renderings stay exit 0 against a healthy server.
    def status_event(self, request, *, timeout):
        yield {
            "event": "status", "state": "idle", "uptime_s": 1.0, "pid": 1,
            "protocol": 1, "vpmdk_version": "x",
            "backend": {"mlp": "MATGL", "model": surrogate_path, "device": "cpu"},
        }

    # The human-readable rendering keeps byte round-tripping: the original
    # filesystem byte is written back out, not lost or escaped away.
    code, out, _ = run_with_streams("status", status_event, json=False)
    assert code == 0
    assert b"\xe9" in out

    # The machine-readable rendering must stay PARSEABLE instead (R150): the
    # raw 0xff/0xe9 byte made `status --json | jq` and json.loads fail on a
    # stream the exit code called healthy, degrading the server's conforming
    # wire frame (whose _serialize_event already falls back to
    # ensure_ascii=True for exactly this payload).
    code, out, _ = run_with_streams("status", status_event, json=True)
    assert code == 0
    parsed = json.loads(out.decode("utf-8"))
    assert parsed["backend"]["model"] == surrogate_path


def test_alphanet_inferred_config_is_advertised_so_a_matching_request_is_accepted(
    tmp_path: Path,
):
    # AlphaNet's builder INFERS the config JSON beside the checkpoint when
    # ALPHANET_CONFIG is omitted. Without recording it, the resident advertised no
    # ALPHANET_CONFIG, so a request whose BCAR spells out the very same file --
    # which `vpmdk --dir` runs with a byte-identical calculator -- was compared
    # against server=None and rejected with exit 5, contradicting
    # SERVER_MODE_SPEC 3.4 ("reject only tags that DIFFER").
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"placeholder")
    config = tmp_path / "alpha_config.json"
    config.write_text('{"a": 1}')

    resident = backend_identity(
        {"MLP": "ALPHANET", "MODEL": str(checkpoint), "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert resident["effective_configuration"]["ALPHANET_CONFIG"] == str(
        config.resolve()
    )

    # Repeating the inferred config is accepted...
    validate_request_backend(
        resident,
        {
            "MLP": "ALPHANET",
            "MODEL": str(checkpoint),
            "ALPHANET_CONFIG": str(config),
            "DEVICE": "cpu",
        },
        request_base_dir=str(tmp_path),
    )
    # ...while a genuinely different config still mismatches.
    other = tmp_path / "other.json"
    other.write_text('{"b": 2}')
    with pytest.raises(BackendConfigurationMismatch, match="ALPHANET_CONFIG"):
        validate_request_backend(
            resident,
            {
                "MLP": "ALPHANET",
                "MODEL": str(checkpoint),
                "ALPHANET_CONFIG": str(other),
                "DEVICE": "cpu",
            },
            request_base_dir=str(tmp_path),
        )


def test_ambiguous_alphanet_layout_infers_nothing(tmp_path: Path):
    # Inference is best effort and must never guess: with two JSON files beside
    # the checkpoint the builder itself refuses to infer, so the resident must
    # advertise no ALPHANET_CONFIG rather than pick one.
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"placeholder")
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")

    resident = backend_identity(
        {"MLP": "ALPHANET", "MODEL": str(checkpoint), "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert "ALPHANET_CONFIG" not in resident["effective_configuration"]


def test_foreground_log_file_is_private_like_the_daemon_log(tmp_path: Path):
    # `--daemon` opens --log-file with an explicit 0600, but logging.FileHandler
    # creates it 0666 & ~umask (typically 0644). The log records the resident
    # MLP/MODEL/DEVICE, every request's workdir and full failure tracebacks, so on
    # a shared host the two modes must not disagree. An EXISTING file keeps its
    # mode, matching os.open's create-only semantics in _daemonize.
    log_file = tmp_path / "fg.log"
    _start_server_instance = VPMDKServer(
        str(tmp_path / "a.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        log_file=str(log_file),
    )
    assert stat.S_IMODE(log_file.stat().st_mode) == 0o600

    preexisting = tmp_path / "pre.log"
    preexisting.touch()
    preexisting.chmod(0o640)
    VPMDKServer(
        str(tmp_path / "b.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        log_file=str(preexisting),
    )
    assert stat.S_IMODE(preexisting.stat().st_mode) == 0o640
    del _start_server_instance


def test_server_loggers_are_private_and_do_not_accumulate(tmp_path: Path):
    # Naming the logger with id(self) and registering it via logging.getLogger
    # kept every server's logger AND its open FileHandler in
    # Logger.manager.loggerDict forever (unbounded growth in an embedding
    # process), and CPython address reuse could hand a NEW server a DEAD one's
    # logger -- inheriting its handler and writing this server's request paths and
    # tracebacks into the previous server's log.
    import logging

    before = len(logging.Logger.manager.loggerDict)
    names = []
    for index in range(5):
        server = VPMDKServer(
            str(tmp_path / f"s{index}.sock"),
            DummyCalculator(),
            {"MLP": "CHGNET", "DEVICE": "cpu"},
            backend_base_dir=str(tmp_path),
            log_file=str(tmp_path / f"s{index}.log"),
        )
        names.append(server.logger.name)

    assert len(set(names)) == len(names)  # unique, so no inherited handlers
    assert not [name for name in names if name in logging.Logger.manager.loggerDict]
    assert len(logging.Logger.manager.loggerDict) == before


def test_status_answers_while_a_handler_resolves_a_slow_workdir(tmp_path: Path, monkeypatch):
    # os.path.realpath walks the path component by component with lstat, so on an
    # autofs/NFS mount it can block for a long time. Doing that while holding
    # _enqueue_lock -- the server's single global gate, also taken by status(),
    # the stop handler, _should_exit() on every accept iteration, and the worker's
    # next dequeue -- froze the whole control plane, violating SERVER_MODE_SPEC
    # 3.2 (status/stop must answer even while busy) and leaving an operator unable
    # to stop the server.
    socket_path = tmp_path / "slow.sock"
    workdir = tmp_path / "work"
    workdir.mkdir()
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)

    real_realpath = os.path.realpath
    inside = threading.Event()

    def slow_realpath(path, *args, **kwargs):
        if str(path).startswith(str(workdir)):
            inside.set()
            time.sleep(5)
        return real_realpath(path, *args, **kwargs)

    try:
        monkeypatch.setattr(server_module.os.path, "realpath", slow_realpath)
        def submit():
            with contextlib.suppress(Exception):
                VPMDKClient(str(socket_path)).run(str(workdir), timeout=30)

        runner = threading.Thread(target=submit, daemon=True)
        runner.start()
        assert inside.wait(15), "the handler never reached the slow realpath"

        started = time.monotonic()
        assert VPMDKClient(str(socket_path)).status(timeout=5)["state"] in {
            "idle",
            "busy",
        }
        # Must not have waited out the stalled realpath.
        assert time.monotonic() - started < 3
    finally:
        monkeypatch.undo()
        # Let the stalled request finish before shutting down, so teardown is not
        # racing the 5s sleep this test deliberately injected.
        runner.join(timeout=30)
        _stop_server(socket_path, thread)


def test_default_parent_hardening_covers_every_socket_name_in_it(tmp_path: Path, monkeypatch):
    # The squattable artifact is the predictable DIRECTORY ${XDG_RUNTIME_DIR:-/tmp}/
    # vpmdk-<uid>, but the gate compared the full socket PATH -- so any other
    # socket name in that same directory (e.g. --socket .../gpu0.sock for a second
    # GPU) skipped the symlink rejection, the ownership probe and the 0700
    # tightening, while default.sock beside it was correctly refused.
    parent = tmp_path / "vpmdk-uid"
    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir(mode=0o777)
    os.symlink(str(attacker_dir), str(parent))

    monkeypatch.setattr(
        server_module, "default_socket_path", lambda: str(parent / "default.sock")
    )
    for name in ("default.sock", "gpu0.sock"):
        with pytest.raises(RuntimeError, match="symlink"):
            server_module.ensure_socket_directory(str(parent / name))

    # A genuinely custom parent elsewhere is still left exactly as the user set it.
    monkeypatch.undo()
    custom = tmp_path / "mine"
    custom.mkdir(mode=0o755)
    server_module.ensure_socket_directory(str(custom / "s.sock"))
    assert stat.S_IMODE(custom.stat().st_mode) == 0o755


def test_daemon_start_timeout_is_capped_below_the_select_limit(monkeypatch):
    # select.select() converts its timeout to int64 nanoseconds, so a value past
    # ~9.2e9 s raises OverflowError -- an ArithmeticError, which serve_cli's
    # `except OSError` around _daemonize does NOT catch. The launcher would die
    # with a raw traceback AFTER the fork, leaving the grandchild to execv into a
    # resident server holding VRAM the user believes failed to start. The docs
    # invite raising this knob, so an "effectively infinite" value must be capped.
    import select

    for raw, expected in (
        ("99999999999", server_module._MAX_DAEMON_START_TIMEOUT),
        ("1e30", server_module._MAX_DAEMON_START_TIMEOUT),
        ("300", 300.0),
        ("not-a-number", 600.0),
        ("0", 600.0),
        ("-5", 600.0),
        ("inf", 600.0),
    ):
        monkeypatch.setenv("VPMDK_DAEMON_START_TIMEOUT", raw)
        assert server_module._daemon_start_timeout() == expected, raw

    # The cap itself is accepted by select(); an uncapped value is not.
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"x")  # readable, so select returns immediately
        select.select([read_fd], [], [], server_module._MAX_DAEMON_START_TIMEOUT)
        with pytest.raises(OverflowError):
            select.select([read_fd], [], [], 1e11)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_cleanup_survives_a_failing_log_handler(tmp_path: Path):
    # _cleanup guards every other teardown step, but the log-handler loop was
    # bare: a buffered write that only fails at flush time (ENOSPC/EDQUOT on a
    # quota-limited log filesystem, EPIPE on a foreground server whose stderr
    # consumer exited) escaped _cleanup and serve_forever's finally, turning an
    # already-complete shutdown -- pidfile removed, socket unlinked -- into
    # exit 1, and leaving the remaining handlers attached with their descriptors
    # open for an embedded caller.
    server = VPMDKServer(
        str(tmp_path / "cleanup.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )

    class _FailingHandler:
        level = 0

        def flush(self):
            raise OSError(errno.EDQUOT, "Disk quota exceeded")

        def close(self):
            raise OSError(errno.EDQUOT, "Disk quota exceeded")

        def handle(self, record):  # pragma: no cover - never emitted here
            pass

    server.logger.addHandler(_FailingHandler())
    server._cleanup()  # must not raise
    assert not server.logger.handlers


def test_deepmd_request_without_a_type_map_is_compared_not_assumed(tmp_path: Path):
    # DEEPMD_TYPE_MAP is the one tag whose ABSENCE does not mean "reuse the
    # resident": one-shot INFERS the map from the calculation's own structure, so
    # an omitted tag means "use MY species ordering", which a resident whose
    # type_map is baked into the loaded DP calculator cannot honor. Left
    # uncompared, a request whose POSCAR orders species differently was accepted
    # and silently evaluated under the RESIDENT's ordering -- wrong species
    # mapping, wrong energies, no diagnostic. Batches that share the resident's
    # ordering (the normal use case) must keep working.
    model = tmp_path / "m.pb"
    model.write_bytes(b"placeholder")
    resident = backend_identity(
        {
            "MLP": "DEEPMD",
            "MODEL": str(model),
            "DEEPMD_TYPE_MAP": "O,H",
            "DEVICE": "cpu",
        },
        base_dir=str(tmp_path),
    )

    def request_dir(species: str) -> str:
        directory = tmp_path / f"req-{species.replace(' ', '')}"
        directory.mkdir()
        (directory / "POSCAR").write_text(
            f"X\n1.0\n5 0 0\n0 5 0\n0 0 5\n{species}\n1 1\nCartesian\n0 0 0\n1 0 0\n"
        )
        return str(directory)

    base_tags = {"MLP": "DEEPMD", "MODEL": str(model), "DEVICE": "cpu"}

    # Same ordering as the resident -> accepted (the batch use case).
    validate_request_backend(
        resident, dict(base_tags), request_base_dir=request_dir("O H")
    )
    # Different ordering -> rejected instead of silently mis-mapping species.
    with pytest.raises(BackendConfigurationMismatch, match="DEEPMD_TYPE_MAP"):
        validate_request_backend(
            resident, dict(base_tags), request_base_dir=request_dir("H O")
        )
    # An explicit tag is still compared as before, in both directions.
    swapped = request_dir("H O 2")
    validate_request_backend(
        resident,
        {**base_tags, "DEEPMD_TYPE_MAP": "O,H"},
        request_base_dir=swapped,
    )
    with pytest.raises(BackendConfigurationMismatch, match="DEEPMD_TYPE_MAP"):
        validate_request_backend(
            resident,
            {**base_tags, "DEEPMD_TYPE_MAP": "H,O"},
            request_base_dir=swapped,
        )
    # An unreadable structure stays an input error from the run, not a mismatch.
    empty = tmp_path / "no-poscar"
    empty.mkdir()
    validate_request_backend(resident, dict(base_tags), request_base_dir=str(empty))


def test_repeated_stop_signal_ends_a_wait_on_a_long_job(tmp_path: Path):
    # Once the accept loop exits, _should_exit -- the ONLY reader of the signal
    # flags -- is never called again, so a further SIGINT/SIGTERM was swallowed by
    # our handler (which deliberately neither raises nor chains to the default)
    # while the unbounded worker join blocked for the rest of a long job. The
    # listener is already closed, so status/stop get ECONNREFUSED: the operator's
    # only recourse was SIGKILL, which skips _cleanup and leaves the socket file
    # and pidfile behind. A repeat signal must complete the escalation ladder.
    socket_path = tmp_path / "sig.sock"
    started = threading.Event()
    release = threading.Event()

    def long_job(workdir: str, *, calculator) -> None:
        started.set()
        release.wait(120)

    server, thread = _start_server(socket_path, executor=long_job)
    try:
        runner = threading.Thread(
            target=lambda: VPMDKClient(str(socket_path)).run(str(tmp_path), timeout=200),
            daemon=True,
        )
        runner.start()
        assert started.wait(30)

        # First signal: graceful. Second: force. (The handler only sets flags.)
        server._stop_signal = True
        server._signal_deliveries += 1
        time.sleep(0.3)
        server._stop_signal = True
        server._force_signal = True
        server._signal_deliveries += 1
        time.sleep(1.0)
        # Force alone still waits for the in-flight executor, as designed.
        assert thread.is_alive()

        # Third signal: stop waiting and finish teardown.
        server._signal_deliveries += 1
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert not socket_path.exists()
        assert not Path(server_module.pidfile_path(str(socket_path))).exists()
    finally:
        release.set()
        thread.join(timeout=30)


def test_stop_through_a_symlink_alias_reports_success(tmp_path: Path):
    # The server unlinks its OWN bound path, which leaves a symlink alias behind
    # as a DANGLING link. Polling with os.path.lexists stayed true for that, so a
    # clean shutdown was never observed and `stop` through an alias always ended
    # in a client timeout (exit 4) instead of exit 0.
    socket_path = tmp_path / "real.sock"
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
    alias = tmp_path / "alias.sock"
    os.symlink(str(socket_path), str(alias))
    try:
        VPMDKClient(str(alias)).stop(timeout=20)  # must not raise ClientTimeoutError
        thread.join(timeout=20)
        assert not thread.is_alive()
        assert not socket_path.exists()
    finally:
        if thread.is_alive():
            _stop_server(socket_path, thread)


def test_deepmd_type_map_is_compared_for_an_neb_band(tmp_path: Path):
    # An NEB band legitimately has NO top-level POSCAR: run_workdir dispatches to
    # run_neb_images on the numbered image directories, and one-shot builds a
    # calculator PER IMAGE from that image's structure. Inferring only from a
    # top-level POSCAR skipped the DEEPMD_TYPE_MAP comparison entirely for a band,
    # so every image ran under the resident's species ordering -- silently wrong
    # energies and forces reported as success.
    model = tmp_path / "m.pb"
    model.write_bytes(b"placeholder")
    resident = backend_identity(
        {
            "MLP": "DEEPMD",
            "MODEL": str(model),
            "DEEPMD_TYPE_MAP": "O,H",
            "DEVICE": "cpu",
        },
        base_dir=str(tmp_path),
    )

    def band(species: str) -> str:
        directory = tmp_path / f"band-{species.replace(' ', '')}"
        directory.mkdir()
        (directory / "INCAR").write_text(
            "IMAGES = 1\nSPRING = -5\nNSW = 0\nIBRION = -1\n"
        )
        for index in range(3):
            image_dir = directory / f"0{index}"
            image_dir.mkdir()
            (image_dir / "POSCAR").write_text(
                f"X\n1.0\n5 0 0\n0 5 0\n0 0 5\n{species}\n1 1\nCartesian\n"
                f"0 0 0\n{1 + index} 0 0\n"
            )
        assert not (directory / "POSCAR").exists()  # the point of the regression
        return str(directory)

    tags = {"MLP": "DEEPMD", "MODEL": str(model), "DEVICE": "cpu"}
    validate_request_backend(resident, dict(tags), request_base_dir=band("O H"))
    with pytest.raises(BackendConfigurationMismatch, match="DEEPMD_TYPE_MAP"):
        validate_request_backend(resident, dict(tags), request_base_dir=band("H O"))


def test_client_accepts_a_symlink_alias_to_its_own_socket(tmp_path: Path, monkeypatch):
    # A stable alias (`ln -s /run/user/<uid>/vpmdk-<uid>/gpu0.sock ~/vpmdk.sock`)
    # is a legitimate setup that connect() handles fine, but lstat'ing the final
    # component rejected it as a "non-socket path" before even trying. Follow the
    # link and validate what it points AT: an attacker's link resolves to THEIR
    # socket, whose owner check still fails.
    lcm = lightweight_client_module
    socket_path = tmp_path / "real.sock"
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
    alias = tmp_path / "alias.sock"
    os.symlink(str(socket_path), str(alias))
    try:
        assert VPMDKClient(str(alias)).status()["state"] == "idle"

        # A symlink to a NON-socket is still refused...
        regular = tmp_path / "plain"
        regular.touch()
        bad_alias = tmp_path / "bad.sock"
        os.symlink(str(regular), str(bad_alias))
        with pytest.raises(ServerConnectionError, match="non-socket"):
            VPMDKClient(str(bad_alias)).status(timeout=1.0)

        # ...and so is a link whose TARGET belongs to someone else.
        real_geteuid = os.geteuid
        monkeypatch.setattr(lcm.os, "geteuid", lambda: real_geteuid() + 1)
        with pytest.raises(ServerConnectionError, match="another user"):
            VPMDKClient(str(alias)).status(timeout=1.0)
    finally:
        monkeypatch.undo()
        _stop_server(socket_path, thread)


def test_vanishing_socket_during_the_grace_wait_still_sweeps_the_pidfile(tmp_path: Path):
    # Every exit path of prepare_socket_path sweeps the paired pidfile EXCEPT the
    # grace-loop FileNotFoundError branch, so a leftover <socket>.pid could
    # survive a concurrent unlink and make the following _write_pidfile abort
    # startup with "Refusing to overwrite pidfile not owned by this VPMDK socket".
    socket_path = tmp_path / "default.sock"
    pidfile = Path(server_module.pidfile_path(str(socket_path)))

    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_path))
    stale.close()
    pidfile.write_bytes(b"")  # zero-length residue of a SIGKILLed daemon

    def reaper():
        time.sleep(0.08)  # inside one of the grace loop's 50 ms sleeps
        with contextlib.suppress(FileNotFoundError):
            os.unlink(str(socket_path))

    threading.Thread(target=reaper, daemon=True).start()
    server_module.prepare_socket_path(str(socket_path))

    assert not socket_path.exists()
    assert not pidfile.exists(), "stale pidfile survived the vanishing-socket race"
    # ...so the next startup can claim it instead of aborting.
    server_module._write_pidfile(str(pidfile), str(socket_path))
    assert pidfile.exists()


def test_socket_is_private_at_creation_under_a_permissive_umask(tmp_path: Path):
    # bind() applies 0777 & ~umask, and on Linux connect() to an AF_UNIX socket
    # needs only write permission on that inode -- so with a permissive umask
    # (002/000, common on HPC login nodes) the endpoint was briefly connectable by
    # other users in the two syscalls before the chmod, and permissions are never
    # re-checked once a connection exists. The mode must be correct AT CREATION.
    previous_umask = os.umask(0o000)
    try:
        socket_path = tmp_path / "s.sock"
        _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
        try:
            mode = stat.S_IMODE(socket_path.stat().st_mode)
            assert not mode & (stat.S_IRWXG | stat.S_IRWXO), oct(mode)
            assert mode == 0o600, oct(mode)
        finally:
            _stop_server(socket_path, thread)
    finally:
        os.umask(previous_umask)


def test_device_is_inert_only_where_the_builder_cannot_act_on_it(tmp_path: Path, monkeypatch):
    # NEQUIX is CONDITIONALLY device-inert: nequix.py reads DEVICE only under
    # `if backend == "torch" and requested_device:` and an unset NEQUIX_BACKEND
    # resolves to "jax", so on the default backend the tag has no effect and
    # comparing it rejected a config `vpmdk --dir` builds identically -- while on
    # torch it genuinely moves the model and must still be compared. Modelling a
    # CUDA host keeps this from being vacuous on CPU-only CI.
    monkeypatch.setattr(
        vpmdk, "_resolve_device", lambda device: ("cuda" if device is None else device)
    )

    def outcome(tags, requested_device):
        resident = backend_identity(dict(tags), base_dir=str(tmp_path))
        request = dict(tags)
        request["DEVICE"] = requested_device
        try:
            validate_request_backend(
                resident, request, request_base_dir=str(tmp_path)
            )
            return resident["device"], "accept"
        except BackendConfigurationMismatch:
            return resident["device"], "reject"

    # Inert on the default (jax) backend, compared on torch.
    assert outcome({"MLP": "NEQUIX", "DEVICE": "cpu"}, "cuda") == ("cpu", "accept")
    assert outcome(
        {"MLP": "NEQUIX", "NEQUIX_BACKEND": "torch", "DEVICE": "cpu"}, "cuda"
    ) == ("cpu", "reject")

    # The unconditional set stays inert, and the resident still advertises the
    # device the user configured (deciding this at comparison time, not by
    # stripping the tag, is what keeps that faithful).
    for mlp in ("GRACE", "MATLANTIS", "DEEPMD"):
        extra = {"DEEPMD_TYPE_MAP": "O,H"} if mlp == "DEEPMD" else {}
        assert outcome({"MLP": mlp, "DEVICE": "cpu", **extra}, "cuda") == (
            "cpu",
            "accept",
        )

    # A backend that reads DEVICE still compares it.
    assert outcome({"MLP": "CHGNET", "DEVICE": "cpu"}, "cuda") == ("cpu", "reject")


def test_device_is_not_compared_for_backends_that_ignore_it(tmp_path: Path):
    # GRACE, Matlantis and DeePMD builders never read the DEVICE tag (GRACE passes
    # only pad/dtype kwargs, Matlantis only model_version/priority/calc_mode,
    # DeePMD only model/type_map/head), so DEVICE cannot make the request's
    # calculator differ. Comparing it produced a permanent exit 5 for a request
    # naming any device other than the host's autodetected one, while
    # `vpmdk --dir` on the same directory builds a byte-identical calculator.
    for mlp in ("GRACE", "MATLANTIS", "DEEPMD"):
        extra = {"DEEPMD_TYPE_MAP": "O,H"} if mlp == "DEEPMD" else {}
        resident = backend_identity(
            {"MLP": mlp, "DEVICE": "cpu", **extra}, base_dir=str(tmp_path)
        )
        for device in ("cpu", "cuda", "cuda:0", ""):
            validate_request_backend(
                resident,
                {"MLP": mlp, "DEVICE": device, **extra},
                request_base_dir=str(tmp_path),
            )

    # A backend that DOES read DEVICE still compares it.
    chgnet = backend_identity(
        {"MLP": "CHGNET", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    with pytest.raises(BackendConfigurationMismatch, match="DEVICE"):
        validate_request_backend(
            chgnet, {"MLP": "CHGNET", "DEVICE": "cuda"}, request_base_dir=str(tmp_path)
        )


def test_default_socket_parent_must_be_private(tmp_path: Path, monkeypatch):
    # Checking only the socket's own owner is not enough at the DEFAULT path: with
    # XDG_RUNTIME_DIR unset it lives under a world-writable /tmp, so before the
    # victim's first server ever runs an attacker can create the parent directory
    # and plant default.sock as a symlink to another socket the victim owns --
    # the owner check passes and the client speaks the protocol to an endpoint the
    # attacker chose. The server refuses a foreign-owned parent; mirror it here,
    # for the default path only.
    lcm = lightweight_client_module
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    default = lcm.default_socket_path()
    parent = Path(default).parent
    parent.mkdir()

    parent.chmod(0o777)
    with pytest.raises(ServerConnectionError, match="group/world-writable"):
        VPMDKClient(default).status(timeout=1.0)

    parent.chmod(0o700)
    real_geteuid = os.geteuid
    monkeypatch.setattr(lcm.os, "geteuid", lambda: real_geteuid() + 1)
    with pytest.raises(ServerConnectionError, match="owned by uid"):
        VPMDKClient(default).status(timeout=1.0)
    monkeypatch.undo()

    # A private parent falls through to the ordinary unreachable error...
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with pytest.raises(ServerConnectionError, match="Cannot connect"):
        VPMDKClient(default).status(timeout=1.0)

    # ...and an EXPLICIT socket path is the user's own choice, not gated here.
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    with pytest.raises(ServerConnectionError, match="Cannot connect"):
        VPMDKClient(str(shared / "custom.sock")).status(timeout=1.0)


def test_default_socket_parent_gate_covers_sibling_socket_names(
    tmp_path: Path, monkeypatch
):
    # R131: the client keyed this gate on the full socket FILENAME while the
    # SERVER has keyed its own on the parent DIRECTORY since round 112. Any
    # sibling name inside the predictable default directory -- the documented
    # one-server-per-GPU layout `--socket $XDG_RUNTIME_DIR/vpmdk-<uid>/gpu0.sock`
    # -- therefore skipped the symlink / foreign-owner / world-writable checks
    # entirely on the client, even though the server refuses to bind there.
    lcm = lightweight_client_module
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    parent = Path(lcm.default_socket_path()).parent
    parent.mkdir()
    sibling = str(parent / "gpu0.sock")

    parent.chmod(0o777)
    with pytest.raises(ServerConnectionError, match="group/world-writable"):
        VPMDKClient(sibling).status(timeout=1.0)

    parent.chmod(0o700)
    real_geteuid = os.geteuid
    monkeypatch.setattr(lcm.os, "geteuid", lambda: real_geteuid() + 1)
    with pytest.raises(ServerConnectionError, match="owned by uid"):
        VPMDKClient(sibling).status(timeout=1.0)
    monkeypatch.undo()

    # A private default parent still falls through to the ordinary error, and a
    # socket outside that directory stays the user's own responsibility.
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with pytest.raises(ServerConnectionError, match="Cannot connect"):
        VPMDKClient(sibling).status(timeout=1.0)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    elsewhere.chmod(0o777)
    with pytest.raises(ServerConnectionError, match="Cannot connect"):
        VPMDKClient(str(elsewhere / "gpu0.sock")).status(timeout=1.0)


def test_pidfile_owned_by_another_user_is_refused(tmp_path: Path, monkeypatch):
    # _write_pidfile validated only the recorded socket path and PID liveness,
    # never the file's OWNER -- so a pidfile another local user pre-planted at the
    # predictable <socket>.pid was treated as a candidate: at 0644 the reopen
    # failed EACCES and surfaced as the opaque "Unable to open daemon pidfile
    # safely" AFTER the full model load, and at 0666 VPMDK truncated it and wrote
    # the victim's PID into the attacker's file. (_assert_private_log_path's
    # docstring already claimed this sibling checked ownership.)
    socket_path = tmp_path / "vpmdk-job.sock"
    pidfile = Path(server_module.pidfile_path(str(socket_path)))
    # Metadata that passes both existing checks: our socket, a dead PID.
    pidfile.write_text(f"4194304\nsocket={os.path.realpath(str(socket_path))}\n")

    real_geteuid = os.geteuid
    monkeypatch.setattr(server_module.os, "geteuid", lambda: real_geteuid() + 1)
    with pytest.raises(RuntimeError, match="owned by uid"):
        server_module._write_pidfile(str(pidfile), str(socket_path))
    monkeypatch.undo()

    # Our own stale pidfile is still claimed.
    server_module._write_pidfile(str(pidfile), str(socket_path))
    assert str(os.getpid()) in pidfile.read_text()


def test_job_stdout_redirect_is_scoped_to_the_worker_thread(tmp_path: Path):
    # contextlib.redirect_stdout swaps the PROCESS-GLOBAL sys.stdout, so for the
    # duration of a job every OTHER thread's print was diverted into that job's
    # client stream -- or dropped once the client had gone -- which silently stole
    # the output of an embedded caller using the documented serve_forever
    # primitive. The redirect must follow the thread that owns the job.
    socket_path = tmp_path / "scoped.sock"
    workdir = tmp_path / "work"
    workdir.mkdir()
    started = threading.Event()
    release = threading.Event()

    def slow(job_workdir: str, *, calculator) -> None:
        started.set()
        print("JOB LINE")
        release.wait(30)

    _, thread = _start_server(socket_path, executor=slow)
    logs: list[str] = []
    try:
        runner = threading.Thread(
            target=lambda: VPMDKClient(str(socket_path)).run(
                str(workdir), timeout=60, log_callback=logs.append
            ),
            daemon=True,
        )
        runner.start()
        assert started.wait(20)

        # The main thread prints while the worker holds the redirect.
        buffer = io.StringIO()
        saved = sys.stdout
        try:
            sys.stdout = buffer
            print("MAIN LINE")
        finally:
            sys.stdout = saved
        assert "MAIN LINE" in buffer.getvalue()
    finally:
        release.set()
        runner.join(timeout=30)
        _stop_server(socket_path, thread)

    # ...and the job's own stdout still reached the client as log events.
    assert any("JOB LINE" in line for line in logs)


def test_undecodable_pidfile_reports_the_intended_diagnostic(tmp_path: Path):
    # _remove_stale_pidfile deliberately LEAVES a non-UTF-8 pidfile in place, so
    # _write_pidfile is the one that has to produce the actionable message. It read
    # the file with strict UTF-8 and never caught UnicodeDecodeError, so a corrupt
    # <socket>.pid aborted `serve --daemon` AFTER the full model load with a bare
    # decode error that never named the pidfile. Its two sibling helpers already
    # tolerate undecodable bytes.
    socket_path = tmp_path / "default.sock"
    pidfile = Path(server_module.pidfile_path(str(socket_path)))
    pidfile.write_bytes(b"\xff\xfe junk\nsocket=" + str(socket_path).encode() + b"\n")

    server_module._remove_stale_pidfile(str(socket_path))
    assert pidfile.exists()  # conservatively left in place, by design

    with pytest.raises(RuntimeError, match="not owned by this VPMDK socket"):
        server_module._write_pidfile(str(pidfile), str(socket_path))

    # A well-formed pidfile for THIS socket is still rewritten.
    pidfile.write_text(f"{os.getpid()}\nsocket={os.path.realpath(str(socket_path))}\n")
    server_module._write_pidfile(str(pidfile), str(socket_path))
    assert str(os.getpid()) in pidfile.read_text()


def test_planted_pidfile_fifo_does_not_hang_startup(tmp_path: Path):
    # Opening a FIFO read-only BLOCKS until a writer appears, so a non-regular
    # entry planted at the predictable <socket>.pid hung `serve` startup (and
    # shutdown) forever with no timeout and no diagnostic -- the same hazard the
    # sibling <socket>.log path already guards. O_NONBLOCK makes the open return
    # at once so the existing S_ISREG check can reject it.
    socket_path = tmp_path / "gpu0.sock"
    pidfile = Path(server_module.pidfile_path(str(socket_path)))
    os.mkfifo(str(pidfile))

    server_module._remove_stale_pidfile(str(socket_path))
    server_module._remove_owned_pidfile(str(pidfile), str(socket_path), os.getpid())
    server_module.prepare_socket_path(str(socket_path))

    # Not ours, so it is left alone rather than unlinked.
    assert pidfile.exists()


def test_blank_device_matches_the_cpu_family_on_a_gpu_host(tmp_path: Path, monkeypatch):
    # Regression guard that is NOT vacuous on CPU-only CI: model a CUDA-capable
    # host so an OMITTED device autodetects to cuda. MatRIS's real build paths
    # (local checkpoint and named model, including the DEFAULT one) resolve a
    # blank DEVICE through `device or "cpu"`, so a blank-DEVICE MatRIS resident
    # genuinely runs on CPU. Sending its blank down the generic
    # "blank == autodetect" branch made a request repeating the resident's own
    # BCAR canonicalize to "cuda" against a resident advertising "cpu" -- a
    # permanent exit 5 for a config one-shot builds identically.
    monkeypatch.setattr(
        vpmdk, "_resolve_device", lambda device: ("cuda" if device is None else device)
    )

    # MATGL/M3GNET belong here for a different reason: matgl.load_model() builds
    # the potential on CPU and a blank DEVICE makes the relocation helper skip
    # .to(...) entirely, so the potential STAYS on CPU. Letting the blank fall
    # through to autodetect made a CUDA host advertise "cuda" for a CPU-resident
    # potential -- accepting a request naming "cuda" (then silently running on the
    # CPU) while rejecting "cpu", the device it really uses.
    for mlp in ("MATRIS", "MATGL", "M3GNET", "HIENET", "EQNORM", "SEVENNET"):
        assert server_module._resolve_backend_device(mlp, "") == "cpu", mlp
        resident = backend_identity(
            {"MLP": mlp, "DEVICE": "cpu"}, base_dir=str(tmp_path)
        )
        # Blank and explicit-cpu both describe the CPU the builder actually uses.
        for device_tag in ("", "cpu"):
            validate_request_backend(
                resident,
                {"MLP": mlp, "DEVICE": device_tag},
                request_base_dir=str(tmp_path),
            )
        with pytest.raises(BackendConfigurationMismatch, match="DEVICE"):
            validate_request_backend(
                resident,
                {"MLP": mlp, "DEVICE": "cuda"},
                request_base_dir=str(tmp_path),
            )

    # Backends that do NOT force cpu still resolve a blank like an omitted one.
    assert server_module._resolve_backend_device("CHGNET", "") == "cuda"


def test_blank_startup_device_advertises_the_real_device(tmp_path: Path):
    # A template BCAR with `DEVICE=` (an unset ${VPMDK_DEVICE}) counts as absent:
    # it names no device and the builders resolve it exactly as an omitted one.
    # Treating the key's mere presence as "explicit" skipped calculator detection,
    # so the resident advertised device="" and then permanently rejected every
    # request naming the device it actually runs on (exit 5) -- for a directory
    # `vpmdk --dir` runs fine, breaking SERVER_MODE_SPEC 3.4.
    class _DeviceCalculator(DummyCalculator):
        device = "cpu"

    for device_tag in ("", "   "):
        server = VPMDKServer(
            str(tmp_path / f"blank{len(device_tag)}.sock"),
            _DeviceCalculator(),
            {"MLP": "CHGNET", "DEVICE": device_tag},
            backend_base_dir=str(tmp_path),
        )
        assert server.backend["device"] == "cpu"
        # The device the resident really runs on is now accepted.
        validate_request_backend(
            server.backend,
            {"MLP": "CHGNET", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )
        # And an explicit device still wins over detection.
    explicit = VPMDKServer(
        str(tmp_path / "explicit.sock"),
        _DeviceCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cuda"},
        backend_base_dir=str(tmp_path),
    )
    assert explicit.backend["device"] == "cuda"


def test_derived_log_rejects_planted_non_regular_and_foreign_files(tmp_path: Path):
    # O_NOFOLLOW only refuses a SYMLINK. At the predictable <socket>.log an
    # attacker can plant a regular file they own (the 0600 mode argument is inert
    # for an existing inode) and receive everything the daemon dup2s onto
    # stdout/stderr, or plant a FIFO whose blocking open() hangs startup -- and
    # logging.FileHandler re-opens by path right after us, so the check must
    # happen BEFORE any open.
    socket_path = tmp_path / "x.sock"
    log_path = Path(server_module.default_log_path(str(socket_path)))

    def build():
        return VPMDKServer(
            str(socket_path),
            DummyCalculator(),
            {"MLP": "CHGNET", "DEVICE": "cpu"},
            backend_base_dir=str(tmp_path),
            log_file=str(log_path),
        )

    os.mkfifo(str(log_path))
    with pytest.raises(RuntimeError, match="non-regular"):
        build()
    log_path.unlink()

    os.symlink(str(tmp_path / "victim"), str(log_path))
    with pytest.raises(RuntimeError, match="symlink"):
        build()
    log_path.unlink()

    # Our own pre-existing regular log is still appended to.
    log_path.write_text("prior\n")
    build()
    assert log_path.read_text().startswith("prior")


def test_teardown_stops_listening_before_joining_the_worker(tmp_path: Path):
    # On a force stop the accept loop exits at once but the worker join waits for
    # the whole in-flight calculation. The listener used to stay bound until
    # _cleanup (which runs AFTER that join), so a client connecting in that window
    # landed in the kernel backlog, had its request accepted by sendall, and then
    # blocked in recv with no `accepted` and no terminal event for the rest of the
    # job (`run --timeout 0` waits forever). Listening must stop BEFORE the join.
    # The socket FILE must still outlive it, so a positive-timeout `stop` client
    # cannot mistake this for completed shutdown.
    socket_path = tmp_path / "teardown.sock"
    server = VPMDKServer(
        str(socket_path),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        heartbeat_interval=0.05,
        executor=lambda *args, **kwargs: None,
    )

    observed: dict[str, object] = {}

    def serve():
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return thread

    thread = serve()
    # Wait for the WORKER, not just the socket file: serve_forever binds (creating
    # the file) before it assigns self._worker, so waiting on the file alone made
    # this test read `None.join` in about half of all runs -- a flaky failure that
    # could mask a real teardown regression.
    _wait_for(lambda: socket_path.exists() and server._worker is not None)

    real_join = server._worker.join

    def recording_join(*args, **kwargs):
        observed["listener_closed"] = server._listener is None
        observed["socket_file_present"] = socket_path.exists()
        return real_join(*args, **kwargs)

    server._worker.join = recording_join  # type: ignore[method-assign]
    server.request_stop(force=True)
    thread.join(timeout=20)

    assert not thread.is_alive()
    assert observed["listener_closed"] is True, "still accepting during the join"
    assert observed["socket_file_present"] is True, "socket file removed too early"
    assert not socket_path.exists()  # removed by _cleanup afterwards


def test_nan_backend_tag_does_not_make_a_resident_reject_itself(tmp_path: Path):
    # NEQUIX_CAPACITY_MULTIPLIER mirrors its builder's strict float(), which
    # accepts "nan", so the resident records a NaN in its own effective config.
    # Plain != makes nan != nan True, so the resident PERMANENTLY rejected a
    # request byte-identical to its own startup BCAR, reporting the
    # self-contradictory "request=nan, server=nan". Two NaNs describe the same
    # configuration -- but every genuinely different pair must still mismatch.
    def resident_for(value):
        tags = {"MLP": "NEQUIX", "DEVICE": "cpu"}
        if value is not None:
            tags["NEQUIX_CAPACITY_MULTIPLIER"] = value
        return backend_identity(dict(tags), base_dir=str(tmp_path))

    def accepts(resident, value):
        tags = {"MLP": "NEQUIX", "DEVICE": "cpu"}
        if value is not None:
            tags["NEQUIX_CAPACITY_MULTIPLIER"] = value
        try:
            validate_request_backend(resident, tags, request_base_dir=str(tmp_path))
            return True
        except BackendConfigurationMismatch:
            return False

    assert accepts(resident_for("nan"), "nan")  # was a permanent exit 5
    assert accepts(resident_for("2.0"), "2.0")
    assert accepts(resident_for("inf"), "inf")
    # Fail-closed: NaN must not become a wildcard in either direction.
    assert not accepts(resident_for("nan"), "2.0")
    assert not accepts(resident_for("2.0"), "nan")
    assert not accepts(resident_for("1.1"), "1.2")
    assert not accepts(resident_for(None), "nan")


def test_derived_default_log_refuses_a_planted_symlink(tmp_path: Path):
    # <socket>.log is DERIVED and predictable and was never named by the user, so
    # a pre-planted symlink there is an attack: the server would append its log
    # (and, under --daemon, every redirected stdout/stderr line and traceback)
    # into an attacker-chosen file with the server's privileges. The sibling
    # derived files are already hardened (<socket>.pid opens O_NOFOLLOW; the
    # socket path must be S_ISSOCK).
    socket_path = tmp_path / "a.sock"
    victim = tmp_path / "victim.txt"
    victim.write_text("original\n")
    os.symlink(str(victim), server_module.default_log_path(str(socket_path)))

    with pytest.raises(RuntimeError, match="symlink"):
        VPMDKServer(
            str(socket_path),
            DummyCalculator(),
            {"MLP": "CHGNET", "DEVICE": "cpu"},
            backend_base_dir=str(tmp_path),
            log_file=server_module.default_log_path(str(socket_path)),
        )
    assert victim.read_text() == "original\n"


def test_explicit_log_file_may_be_a_symlink(tmp_path: Path):
    # An EXPLICIT --log-file is the user's own path, where a symlink (log
    # rotation, /var/log indirection) is a legitimate setup the hardening above
    # must not break.
    target = tmp_path / "real.log"
    target.touch()
    link = tmp_path / "explicit.log"
    os.symlink(str(target), str(link))

    server = VPMDKServer(
        str(tmp_path / "b.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        log_file=str(link),
    )
    server.logger.info("hello through the symlink")
    for handler in server.logger.handlers:
        handler.flush()
    assert "hello through the symlink" in target.read_text()


def test_backend_notimplementederror_is_a_calculation_failure_not_input():
    # The blanket "NotImplementedError -> input_error" rule also captured a
    # NotImplementedError raised MID-CALCULATION by a third-party backend (torch's
    # "Could not run 'aten::...' with arguments from the 'CUDA' backend"), which
    # SERVER_MODE_SPEC 2.5 defines as exit 2 -- and the exit-1 branch additionally
    # suppresses the traceback (only the calculation-failure branch prints one).
    # VPMDK's own "mode not implemented" raises now use a dedicated subclass.
    def classify(exc):
        if isinstance(exc, BackendConfigurationMismatch):
            return "backend_mismatch"
        if isinstance(exc, (vpmdk.WorkdirInputError, vpmdk.UnsupportedInputError)):
            return "input_error"
        return "calculation_error"

    assert classify(vpmdk.UnsupportedInputError("ICHAIN=1")) == "input_error"
    assert classify(vpmdk.WorkdirInputError("bad POSCAR")) == "input_error"
    assert (
        classify(NotImplementedError("Could not run 'aten::x' with CUDA"))
        == "calculation_error"
    )
    # Still a NotImplementedError, so existing callers/tests keep working.
    assert issubclass(vpmdk.UnsupportedInputError, NotImplementedError)

    # VPMDK's own raise sites use the dedicated type.
    from vpmdk_core.runtime import single as single_module
    from vpmdk_core.settings import incar as incar_module

    with pytest.raises(vpmdk.UnsupportedInputError):
        incar_module._reject_unsupported_vtst_modes({"ICHAIN": 1})
    with pytest.raises(vpmdk.UnsupportedInputError):
        single_module._validate_nfree(3)


def test_client_usage_errors_do_not_use_the_retryable_exit_code():
    # argparse exits 2 for a usage error, but SERVER_MODE_SPEC 2.5 reserves exit 2
    # for a RETRYABLE server-side calculation failure -- so a permanently
    # malformed command line was reported to a retry driver as worth retrying.
    lcm = lightweight_client_module

    def exit_code(main, argv):
        buffer_out, buffer_err = io.StringIO(), io.StringIO()
        saved = (sys.stdout, sys.stderr)
        try:
            sys.stdout, sys.stderr = buffer_out, buffer_err
            main(argv)
        except SystemExit as exc:
            return exc.code
        finally:
            sys.stdout, sys.stderr = saved
        return None

    for argv in (
        ["run", "--timeout", "abc"],
        ["run", "--dir"],
        ["status", "--bogus"],
        ["stop", "--timeout", "x"],
    ):
        assert exit_code(lcm.client_main, argv) == 1, argv
    assert exit_code(lcm.client_main, ["run", "--help"]) == 0

    # The full CLI's subcommand parser is mapped the same way...
    assert exit_code(vpmdk.main, ["run", "--timeout", "abc"]) == 1
    assert exit_code(vpmdk.main, ["serve", "--idle-timeout", "x"]) == 1
    # ...while the LEGACY parser keeps argparse's exit 2 (SPEC 1.1 byte-for-byte).
    assert exit_code(vpmdk.main, ["--bogus"]) == 2


def test_huge_uptime_does_not_break_the_status_exit_contract():
    # OverflowError is an ArithmeticError, NOT a ValueError: a JSON integer past
    # ~1.8e308 (but under json.loads' own 4300-digit literal limit) passes
    # status()'s isinstance(int) check and then makes _format_status' float()
    # raise it, escaping every client_cli handler as a traceback plus an
    # undocumented exit code -- while `status --json` rendered the byte-identical
    # payload as exit 0. Both renderings must stay within spec 2.3 (0 or 3).
    import argparse

    lcm = lightweight_client_module
    huge = int("9" * 401)

    assert "Uptime: 0.0 s" in lcm._format_status(
        {"event": "status", "state": "idle", "backend": {}, "uptime_s": huge}
    )

    def status_event(self, request, *, timeout):
        yield {
            "event": "status", "state": "idle", "uptime_s": huge, "pid": 1,
            "protocol": 1, "vpmdk_version": "x",
            "backend": {"mlp": "CHGNET", "model": "m", "device": "cpu"},
        }

    original_request = lcm.VPMDKClient._request
    try:
        lcm.VPMDKClient._request = status_event
        for use_json in (False, True):
            args = argparse.Namespace(command="status", socket="/s", json=use_json)
            assert lcm.client_cli(args) in {0, 3}
    finally:
        lcm.VPMDKClient._request = original_request


def test_socket_vanishing_during_startup_does_not_abort_serve(tmp_path: Path, monkeypatch):
    # `vpmdk stop --timeout 0` returns before shutdown completes, so a restart
    # script can call `serve` while the dying server's _cleanup() is still
    # unlinking the socket. If the entry disappears between lexists() and the
    # lstat() that follows, FileNotFoundError reached serve_cli's generic handler
    # and aborted startup with exit 1 -- the correct outcome is to treat the path
    # as free and bind.
    socket_path = tmp_path / "racy.sock"
    real_lexists = os.path.lexists

    def racy_lexists(path):
        # Report the socket as present, then remove it before the lstat runs.
        if str(path) == str(socket_path):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(str(socket_path))
            return True
        return real_lexists(path)

    monkeypatch.setattr(server_module.os.path, "lexists", racy_lexists)
    server_module.prepare_socket_path(str(socket_path))  # must not raise

    # A genuine non-socket entry is still refused.
    monkeypatch.undo()
    regular = tmp_path / "regular"
    regular.touch()
    with pytest.raises(RuntimeError, match="non-socket"):
        server_module.prepare_socket_path(str(regular))


def test_client_refuses_a_socket_another_user_owns(tmp_path: Path, monkeypatch):
    # The default socket path is predictable and, with XDG_RUNTIME_DIR unset, sits
    # under a world-writable /tmp, so another local user can pre-bind it and
    # impersonate the server. The server already refuses a foreign-owned parent;
    # the client trusted whatever answered, and a forged
    # {"event":"done","ok":true} is indistinguishable from a real calculation
    # (the documented `until vpmdk status` readiness gate even succeeds on its
    # first iteration against the impostor). Refuse before speaking the protocol.
    lcm = lightweight_client_module

    impostor = tmp_path / "impostor.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(impostor))
    listener.listen(1)
    try:
        # Pretend the socket belongs to a different uid. Capture the real function
        # first: lcm.os IS the global os module, so patching it in terms of itself
        # would recurse.
        real_geteuid = os.geteuid
        monkeypatch.setattr(lcm.os, "geteuid", lambda: real_geteuid() + 1)
        with pytest.raises(ServerConnectionError, match="another user"):
            VPMDKClient(str(impostor)).status(timeout=1.0)
        args = lcm.argparse.Namespace(
            command="status", socket=str(impostor), json=False
        )
        assert lcm.client_cli(args) == 3  # unreachable, per spec 2.3
    finally:
        listener.close()

    # A regular file at the endpoint is not a server either.
    regular = tmp_path / "not-a-socket"
    regular.touch()
    with pytest.raises(ServerConnectionError, match="non-socket"):
        VPMDKClient(str(regular)).status(timeout=1.0)

    # A MISSING socket keeps the previous unreachable message and exit code.
    with pytest.raises(ServerConnectionError, match="Cannot connect"):
        VPMDKClient(str(tmp_path / "absent.sock")).status(timeout=1.0)


def test_client_accepts_its_own_socket(tmp_path: Path):
    # The ownership guard must not break the normal case.
    socket_path = tmp_path / "server.sock"
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
    try:
        assert VPMDKClient(str(socket_path)).status()["state"] == "idle"
    finally:
        _stop_server(socket_path, thread)


def test_cross_mlp_request_reports_the_mlp_difference_not_a_foreign_model_path(
    tmp_path: Path,
):
    # A request naming a DIFFERENT MLP was still canonicalized under the RESIDENT's
    # backend policy: a valid CHGNet named MODEL resolved under MACE's local-only
    # policy raises FileNotFoundError, and the except replaced the real difference
    # with "MACE MODEL path not found: <workdir>/CHGNet-v0.3.0" -- naming a backend
    # and a checkpoint path the user never wrote. SERVER_MODE_SPEC 3.4 requires the
    # differing tag to be enumerated.
    checkpoint = tmp_path / "mace.model"
    checkpoint.write_text("placeholder")
    resident = backend_identity(
        {"MLP": "MACE", "MODEL": str(checkpoint), "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    with pytest.raises(BackendConfigurationMismatch) as excinfo:
        validate_request_backend(
            resident,
            {"MLP": "CHGNET", "MODEL": "CHGNet-v0.3.0", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )
    assert excinfo.value.differences == ["MLP request='CHGNET', server='MACE'"]

    # Same-MLP behavior is unchanged: real differences are still enumerated, and an
    # unresolvable MODEL under the resident's own policy still reports as invalid.
    with pytest.raises(BackendConfigurationMismatch, match="DEVICE"):
        validate_request_backend(
            resident,
            {"MLP": "MACE", "MODEL": str(checkpoint), "DEVICE": "cuda"},
            request_base_dir=str(tmp_path),
        )
    with pytest.raises(BackendConfigurationMismatch, match="Request configuration is invalid"):
        validate_request_backend(
            resident,
            {"MLP": "MACE", "MODEL": "no-such-model", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_teardown_terminates_jobs_no_worker_will_ever_run(tmp_path: Path):
    # A queued _RunJob OWNS its client's accepted socket, so a job left in the queue
    # means a client blocked forever on recv (no terminal event, no EOF) plus a
    # leaked fd. serve_forever's teardown only drained the queue on the FORCE path,
    # so an abnormal exit from the accept loop stranded them -- and because
    # _stop_requested was never set, a handler still inside _read_request could
    # enqueue yet another job onto a queue whose worker had already exited.
    server = VPMDKServer(
        str(tmp_path / "drain.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        executor=lambda *args, **kwargs: None,
    )

    sent: list[dict] = []
    closed: list[bool] = []

    class RecordingSender:
        def send(self, event):
            sent.append(event)

        def close(self):
            closed.append(True)

    class BrokenSender(RecordingSender):
        def send(self, event):
            raise OSError("peer gone")

        def close(self):
            raise OSError("peer gone")

    def job(sender):
        return server_module._RunJob(
            workdir="/w", caller_cwd="/w", sender=sender, enqueued_at=0.0
        )

    for _ in range(3):
        server._queue.put(job(RecordingSender()))
    server._reject_pending_jobs("Server stopped before this calculation started.")
    assert server._queue.empty()
    assert len(sent) == 3 and len(closed) == 3
    assert all(event["code"] == "server_stopping" for event in sent)
    assert all(event["ok"] is False for event in sent)

    # A peer that already vanished must not abort the drain of the rest.
    sent.clear()
    server._queue.put(job(BrokenSender()))
    server._queue.put(job(RecordingSender()))
    server._reject_pending_jobs("Server stopped before this calculation started.")
    assert server._queue.empty()
    assert len(sent) == 1


def test_abnormal_accept_loop_exit_marks_the_server_stopping(tmp_path: Path):
    # Without _stop_requested, a handler thread still inside _read_request sails
    # past the run guard and enqueues onto a queue whose worker is exiting.
    socket_path = tmp_path / "abnormal.sock"
    server = VPMDKServer(
        str(socket_path),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        executor=lambda *args, **kwargs: None,
    )

    def serve():
        try:
            server.serve_forever()
        except BaseException:  # the abnormal exit under test
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    # Wait for the LISTENER, not just the socket file: _bind publishes
    # self._listener only after bind() has already created the file, so waiting on
    # the file alone made this call `None.close()` under load -- a flaky failure
    # that hides real regressions (the same race as
    # test_teardown_stops_listening_before_joining_the_worker).
    _wait_for(lambda: socket_path.exists() and server._listener is not None)
    server._listener.close()  # non-transient accept error -> abnormal exit
    thread.join(timeout=20)
    assert not thread.is_alive()
    assert server._stop_requested.is_set()


def test_blank_graph_converter_alias_is_rejected_like_the_builder(tmp_path: Path):
    # The graph-converter selection reaches _canonical_configuration through the
    # alias loop (CHGNET_GRAPH_CONVERTER -> canonical GRAPH_CONVERTER_ALGORITHM).
    # Its builder resolver raises on a blank, so a blank alias value must not be
    # omitted either -- it must diverge from the resident's real algorithm.
    resident = backend_identity(
        {
            "MLP": "CHGNET",
            "CHGNET_GRAPH_CONVERTER": "fast",
            "DEVICE": "cpu",
        },
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(
            resident,
            {"MLP": "CHGNET", "CHGNET_GRAPH_CONVERTER": "", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )


def test_blank_fairchem_task_uses_default(tmp_path: Path):
    # FAIRCHEM_TASK's builder now treats a blank value as unset (use default),
    # matching the blank->omit config rule, so status and the equivalence check
    # agree with the calculator.
    resident = backend_identity(
        {"MLP": "FAIRCHEM", "FAIRCHEM_TASK": "", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    assert (
        resident["effective_configuration"]["FAIRCHEM_TASK"]
        == vpmdk.DEFAULT_FAIRCHEM_TASK
    )
    validate_request_backend(
        resident,
        {"MLP": "FAIRCHEM", "FAIRCHEM_TASK": vpmdk.DEFAULT_FAIRCHEM_TASK, "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_upet_neighborlist_auto_default_matches_explicit_cpu_on_cpu_model(
    tmp_path: Path,
):
    # On a CPU model the default "auto" and an explicit "cpu" both run the
    # neighbor list on CPU, so the resident config must record "cpu" (not None)
    # and an equivalent request naming "cpu" must be accepted.
    resident = backend_identity(
        {"MLP": "UPET", "MODEL": "pet-oam-xl", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    assert resident["effective_configuration"]["UPET_NEIGHBORLIST_DEVICE"] == "cpu"
    validate_request_backend(
        resident,
        {
            "MLP": "UPET",
            "MODEL": "pet-oam-xl",
            "UPET_NL_DEVICE": "cpu",
            "DEVICE": "cpu",
        },
        request_base_dir=str(tmp_path),
    )


def test_upet_explicit_auto_matches_cuda_neighborlist_default(tmp_path: Path):
    resident = backend_identity(
        {"MLP": "UPET", "MODEL": "pet-oam-xl", "DEVICE": "cuda"},
        base_dir=str(tmp_path),
    )

    validate_request_backend(
        resident,
        {"UPET_NL_DEVICE": "auto"},
        request_base_dir=str(tmp_path),
    )


def test_equflash_request_conflicting_with_forced_flash_defaults_is_rejected(
    tmp_path: Path,
):
    (tmp_path / "equflash.ckpt").write_text("placeholder")
    resident = backend_identity(
        {"MLP": "EQUFLASH", "MODEL": "equflash.ckpt", "DEVICE": "cuda"},
        base_dir=str(tmp_path),
    )

    with pytest.raises(BackendConfigurationMismatch, match="SEVENNET_ENABLE_FLASH"):
        validate_request_backend(
            resident,
            {"SEVENNET_ENABLE_FLASH": "false"},
            request_base_dir=str(tmp_path),
        )


@pytest.mark.parametrize(
    ("mlp", "startup_tags", "request_tags"),
    [
        ("ORB", {"ORB_COMPILE": "true"}, {"ORB_COMPILE": "1"}),
        (
            "MATLANTIS",
            {"MATLANTIS_PRIORITY": "50.0", "MATLANTIS_CALC_MODE": "pbe"},
            {"PRIORITY": "50", "CALC_MODE": "PBE"},
        ),
        ("MATRIS", {"MATRIS_TASK": "EFS"}, {"MATRIS_TASK": "efs"}),
        (
            "ALPHANET",
            {"ALPHANET_PRECISION": "float32"},
            {"ALPHANET_DTYPE": "32"},
        ),
        (
            "NEQUIX",
            {
                "NEQUIX_USE_KERNEL": "yes",
                "NEQUIX_CAPACITY_MULTIPLIER": "1.10",
            },
            {"NEQUIX_KERNEL": "on", "NEQUIX_CAPACITY_MULTIPLIER": "1.1"},
        ),
        (
            "UPET",
            {"UPET_NEIGHBORLIST_DEVICE": "host"},
            {"UPET_NL_DEVICE": "cpu"},
        ),
        (
            "TACE",
            {"TACE_SPIN_ON": "true", "TACE_FIDELITY_IDX": "1.0"},
            {"TACE_SPIN_ON": "1", "TACE_LEVEL": "1"},
        ),
        (
            "GRACE",
            {
                "GRACE_PAD_NEIGHBORS_FRACTION": "0.100",
                "GRACE_PAD_ATOMS_NUMBER": "2.0",
            },
            {
                "GRACE_PAD_NEIGHBORS_FRACTION": ".1",
                "GRACE_PAD_ATOMS_NUMBER": "2",
            },
        ),
        (
            "DEEPMD",
            {"DEEPMD_TYPE_MAP": "Si, O"},
            {"DEEPMD_TYPE_MAP": "Si O"},
        ),
        (
            "EQUIFORMER_V3",
            {
                "EQUIFORMER_V3_MODULE": "package.one",
                "EQUIFORMER_V3_IMPORT_MODULE": "package.two",
            },
            {"EQUIFORMER_V3_MODULE": "package.one,package.two"},
        ),
    ],
)
def test_backend_configuration_values_compare_by_builder_semantics(
    tmp_path: Path,
    mlp: str,
    startup_tags: dict[str, str],
    request_tags: dict[str, str],
):
    resident = backend_identity(
        {"MLP": mlp, "DEVICE": "cpu", **startup_tags},
        base_dir=str(tmp_path),
    )

    validate_request_backend(
        resident,
        {"MLP": mlp, "DEVICE": "cpu", **request_tags},
        request_base_dir=str(tmp_path),
    )


def test_equiformer_module_aliases_deduplicate_before_comparison(tmp_path: Path):
    repeated_tags = {
        "MLP": "EQUIFORMER_V3",
        "DEVICE": "cpu",
        "EQUIFORMER_V3_MODULE": "package.one",
        "EQUIFORMER_V3_IMPORT_MODULE": "package.one,package.two,package.one",
    }
    single_tags = {
        "MLP": "EQUIFORMER_V3",
        "DEVICE": "cpu",
        "EQUIFORMER_V3_MODULE": "package.one,package.two",
    }

    repeated_resident = backend_identity(repeated_tags, base_dir=str(tmp_path))
    assert repeated_resident["effective_configuration"][
        "EQUIFORMER_V3_MODULES"
    ] == ("package.one", "package.two")
    validate_request_backend(
        repeated_resident,
        single_tags,
        request_base_dir=str(tmp_path),
    )

    single_resident = backend_identity(single_tags, base_dir=str(tmp_path))
    validate_request_backend(
        single_resident,
        repeated_tags,
        request_base_dir=str(tmp_path),
    )


def test_request_scoped_bcar_tags_do_not_cause_mismatch(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "BCAR").write_text(
        "WRITE_ENERGY_CSV=1\nWRITE_CHGCAR=0\nFORCE_CONSTANTS_DISPLACEMENT=0.02\n"
    )
    ran = False

    def executor(workdir: str, *, calculator) -> None:
        nonlocal ran
        ran = True
        print("Calculation completed.")

    _, thread = _start_server(socket_path, executor=executor)
    try:
        VPMDKClient(str(socket_path)).run(str(workdir))
        assert ran
    finally:
        _stop_server(socket_path, thread)


def test_explicit_bcar_uses_its_directory_for_relative_model_paths(
    tmp_path: Path,
    monkeypatch,
):
    config_dir = tmp_path / "config"
    request_dir = tmp_path / "request"
    config_dir.mkdir()
    request_dir.mkdir()
    (config_dir / "weights.pt").write_text("dummy")
    bcar_path = config_dir / "resident.bcar"
    bcar_path.write_text("MLP=MACE\nMODEL=weights.pt\nDEVICE=cpu\n")
    captured: dict[str, object] = {}
    calculator = DummyCalculator()

    def build(tags, *, structure=None):
        captured["cwd"] = os.getcwd()
        captured["tags"] = dict(tags)
        return calculator

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", build)
    loaded, tags, base_dir = _load_backend_for_server(str(request_dir), str(bcar_path))

    assert loaded is calculator
    assert captured["cwd"] == str(config_dir)
    resident = backend_identity(tags, base_dir=base_dir)
    validate_request_backend(
        resident,
        {"MLP": "MACE", "MODEL": "../config/weights.pt", "DEVICE": "cpu"},
        request_base_dir=str(request_dir),
    )


def test_server_preserves_symlink_for_builder_and_realpath_for_identity(
    tmp_path: Path,
    monkeypatch,
):
    startup_dir = tmp_path / "startup"
    target_dir = tmp_path / "target"
    startup_dir.mkdir()
    target_dir.mkdir()
    target = target_dir / "weights.pt"
    target.write_text("checkpoint")
    model_link = startup_dir / "weights.pt"
    model_link.symlink_to(target)
    (startup_dir / "BCAR").write_text(
        "MLP=MACE\nMODEL=weights.pt\nDEVICE=cpu\n"
    )
    captured: dict[str, object] = {}

    def build(tags, *, structure=None):
        captured.update(tags)
        return DummyCalculator()

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", build)

    _, tags, base_dir = _load_backend_for_server(str(startup_dir), None)
    identity = backend_identity(tags, base_dir=base_dir)

    assert captured["MODEL"] == str(model_link)
    assert tags["MODEL"] == str(model_link)
    assert identity["model"] == str(target.resolve())
    assert identity["configuration"]["MODEL"] == str(target.resolve())


def test_missing_startup_model_path_is_rejected_before_build(tmp_path: Path, monkeypatch):
    workdir = tmp_path / "startup"
    workdir.mkdir()
    (workdir / "BCAR").write_text(
        "MLP=CHGNET\nMODEL=missing/custom.pt\nDEVICE=cpu\n"
    )

    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda *args, **kwargs: pytest.fail("builder accepted a missing MODEL path"),
    )

    with pytest.raises(FileNotFoundError, match="CHGNET MODEL path not found"):
        _load_backend_for_server(str(workdir), None)


def test_extensionless_local_only_startup_model_is_rejected(
    tmp_path: Path, monkeypatch
):
    workdir = tmp_path / "startup"
    workdir.mkdir()
    (workdir / "BCAR").write_text(
        "MLP=MACE\nMODEL=weights\nDEVICE=cpu\n"
    )
    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda *args, **kwargs: pytest.fail(
            "MACE builder accepted a missing extensionless checkpoint"
        ),
    )

    with pytest.raises(FileNotFoundError, match="MACE MODEL path not found"):
        _load_backend_for_server(str(workdir), None)


def test_opaque_fairchem_model_identifier_is_not_treated_as_a_path(
    tmp_path: Path, monkeypatch
):
    workdir = tmp_path / "startup"
    workdir.mkdir()
    (workdir / "BCAR").write_text(
        "MLP=FAIRCHEM\nMODEL=provider/model-name\nDEVICE=cpu\n"
    )
    calculator = DummyCalculator()
    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda *args, **kwargs: calculator,
    )

    loaded, tags, base_dir = _load_backend_for_server(str(workdir), None)

    assert loaded is calculator
    assert tags["MODEL"] == "provider/model-name"
    resident = backend_identity(tags, base_dir=base_dir)
    assert resident["model"] == "provider/model-name"
    assert resident["effective_configuration"]["MODEL"] == "provider/model-name"
    validate_request_backend(
        resident,
        {"MLP": "FAIRCHEM", "MODEL": "provider/model-name", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path / "different-request-directory"),
    )


def test_legacy_m3gnet_default_does_not_advertise_matgl_model(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", True)
    resident = backend_identity(
        {"MLP": "M3GNET", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )

    assert resident["model"] is None
    with pytest.raises(BackendConfigurationMismatch, match="MODEL"):
        validate_request_backend(
            resident,
            {
                "MLP": "M3GNET",
                "MODEL": vpmdk.DEFAULT_MATGL_MODEL,
                "DEVICE": "cpu",
            },
            request_base_dir=str(tmp_path),
        )


def test_matgl_named_default_is_stable_across_directories(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)
    startup_dir = tmp_path / "startup"
    request_dir = tmp_path / "request"
    startup_dir.mkdir()
    request_dir.mkdir()

    resident = backend_identity(
        {"MLP": "MATGL", "DEVICE": "cpu"}, base_dir=str(startup_dir)
    )

    assert resident["model"] == vpmdk.DEFAULT_MATGL_MODEL
    validate_request_backend(
        resident,
        {
            "MLP": "M3GNET",
            "MODEL": vpmdk.DEFAULT_MATGL_MODEL,
            "DEVICE": "cpu",
        },
        request_base_dir=str(request_dir),
    )

    local_model = startup_dir / vpmdk.DEFAULT_MATGL_MODEL
    local_model.mkdir()
    explicit_local = backend_identity(
        {
            "MLP": "MATGL",
            "MODEL": vpmdk.DEFAULT_MATGL_MODEL,
            "DEVICE": "cpu",
        },
        base_dir=str(startup_dir),
    )
    assert explicit_local["model"] == str(local_model.resolve())


def test_validate_request_blank_model_inherits_resident_for_orb(
    tmp_path: Path, monkeypatch
):
    # ORB's model identity comes from ORB_MODEL, not MODEL. A blank MODEL is
    # unspecified and must inherit the resident (SERVER_MODE_SPEC §3.4), not be
    # resolved to None and mis-rejected against the resident's ORB_MODEL.
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)
    resident = backend_identity(
        {"MLP": "ORB", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    assert resident["model"] == vpmdk.DEFAULT_ORB_MODEL

    validate_request_backend(
        resident,
        {"MLP": "ORB", "MODEL": "", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_validate_request_blank_model_inherits_resident_for_matlantis(
    tmp_path: Path,
):
    # Matlantis' model identity is its version. A blank MODEL alias alongside a
    # matching MATLANTIS_MODEL_VERSION must not be resolved to the default
    # version and mis-rejected.
    resident = backend_identity(
        {"MLP": "MATLANTIS", "MATLANTIS_MODEL_VERSION": "v7.0.0", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert resident["model"] == "v7.0.0"

    validate_request_backend(
        resident,
        {
            "MLP": "MATLANTIS",
            "MATLANTIS_MODEL_VERSION": "v7.0.0",
            "MODEL": "",
            "DEVICE": "cpu",
        },
        request_base_dir=str(tmp_path),
    )


def test_validate_request_blank_model_inherits_nondefault_resident(
    tmp_path: Path, monkeypatch
):
    # An unspecified MODEL reuses the resident model even when the resident runs
    # an explicit non-default model; a blank MODEL carries no model intent.
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)
    resident = backend_identity(
        {"MLP": "MATGL", "MODEL": "M3GNet-Custom-Model", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert resident["model"] == "M3GNet-Custom-Model"

    validate_request_backend(
        resident,
        {"MLP": "MATGL", "MODEL": "", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


def test_validate_request_blank_model_inherits_resident_for_required_backend(
    tmp_path: Path, monkeypatch
):
    # Server mode reuses the resident model for an unspecified MODEL, even for a
    # required backend: per SERVER_MODE_SPEC §3.4 only backend tags that differ
    # from the resident are rejected, and a blank MODEL does not differ.
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)
    model_path = tmp_path / "deployed.pth"
    model_path.write_text("weights")
    resident = backend_identity(
        {"MLP": "NEQUIP", "MODEL": str(model_path), "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    validate_request_backend(
        resident,
        {"MLP": "NEQUIP", "MODEL": "", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )


@pytest.mark.parametrize(
    ("mlp", "default_model"),
    [
        ("EQNORM", vpmdk.DEFAULT_EQNORM_MODEL),
        ("MATRIS", vpmdk.DEFAULT_MATRIS_MODEL),
        ("SEVENNET", vpmdk.DEFAULT_SEVENNET_MODEL),
        ("FAIRCHEM", vpmdk.DEFAULT_FAIRCHEM_MODEL),
    ],
)
def test_omitted_named_default_is_not_shadowed_by_same_named_local_entry(
    tmp_path: Path,
    mlp: str,
    default_model: str,
):
    (tmp_path / default_model).mkdir()

    resident = backend_identity(
        {"MLP": mlp, "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    assert resident["model"] == default_model


def test_matgl_named_model_identifier_is_stable_across_directories(
    tmp_path: Path, monkeypatch
):
    workdir = tmp_path / "startup"
    request_dir = tmp_path / "request"
    workdir.mkdir()
    request_dir.mkdir()
    model_name = "M3GNet-MP-2018.6.1-Eform"
    (workdir / "BCAR").write_text(
        f"MLP=MATGL\nMODEL={model_name}\nDEVICE=cpu\n"
    )
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)
    calculator = DummyCalculator()
    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda *args, **kwargs: calculator,
    )

    loaded, tags, base_dir = _load_backend_for_server(str(workdir), None)

    assert loaded is calculator
    resident = backend_identity(tags, base_dir=base_dir)
    assert resident["model"] == model_name
    assert resident["effective_configuration"]["MODEL"] == model_name
    validate_request_backend(
        resident,
        {"MLP": "M3GNET", "MODEL": model_name, "DEVICE": "cpu"},
        request_base_dir=str(request_dir),
    )


def test_missing_matgl_checkpoint_with_path_suffix_is_rejected(
    tmp_path: Path, monkeypatch
):
    workdir = tmp_path / "startup"
    workdir.mkdir()
    (workdir / "BCAR").write_text("MLP=MATGL\nMODEL=weights.pt\nDEVICE=cpu\n")
    monkeypatch.setattr(vpmdk, "_USING_LEGACY_M3GNET", False)
    monkeypatch.setattr(
        vpmdk,
        "_build_calculator_from_tags",
        lambda *args, **kwargs: pytest.fail(
            "MatGL builder accepted a missing path-like checkpoint"
        ),
    )

    with pytest.raises(FileNotFoundError, match="MatGL MODEL path not found"):
        _load_backend_for_server(str(workdir), None)


def test_grace_identity_uses_installed_foundation_model_fallback(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        vpmdk, "GRACE_MODEL_NAMES", ["GRACE-INSTALLED", "GRACE-OTHER"]
    )
    resident = backend_identity(
        {"MLP": "GRACE", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )

    assert resident["model"] == "GRACE-INSTALLED"
    validate_request_backend(
        resident,
        {"MLP": "GRACE", "MODEL": "GRACE-INSTALLED", "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )
    # Unknown names follow GRACE's documented warning+fallback behavior. The
    # request identity is the actual installed fallback, not the misspelling.
    validate_request_backend(
        resident,
        {"MLP": "GRACE", "MODEL": vpmdk.DEFAULT_GRACE_MODEL, "DEVICE": "cpu"},
        request_base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch, match="MODEL"):
        validate_request_backend(
            resident,
            {"MLP": "GRACE", "MODEL": "GRACE-OTHER", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )

    fallback = backend_identity(
        {"MLP": "GRACE", "MODEL": "GRACE-UNAVAILABLE", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    assert fallback["model"] == "GRACE-INSTALLED"


def test_deepmd_server_requires_explicit_type_map(tmp_path: Path, monkeypatch):
    workdir = tmp_path / "startup"
    workdir.mkdir()
    (workdir / "BCAR").write_text("MLP=DEEPMD\nMODEL=model.pb\n")
    built = False

    def build(*args, **kwargs):
        nonlocal built
        built = True
        return DummyCalculator()

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", build)

    with pytest.raises(ValueError, match="requires an explicit DEEPMD_TYPE_MAP"):
        _load_backend_for_server(str(workdir), None)
    assert not built

    with pytest.raises(ValueError, match="startup POSCAR is unsafe"):
        VPMDKServer(
            str(tmp_path / "embedded.sock"),
            DummyCalculator(),
            {"MLP": "DEEPMD", "MODEL": "model.pb"},
            backend_base_dir=str(tmp_path),
        )


def test_deepmd_server_uses_explicit_type_map_without_startup_structure(
    tmp_path: Path,
    monkeypatch,
):
    workdir = tmp_path / "startup"
    workdir.mkdir()
    (workdir / "BCAR").write_text(
        "MLP=DEEPMD\nMODEL=model.pb\nDEEPMD_TYPE_MAP=Si,O\n"
    )
    (workdir / "model.pb").write_text("dummy model")
    (workdir / "POSCAR").write_text("startup structure placeholder\n")
    calculator = DummyCalculator()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        vpmdk,
        "read_structure",
        lambda *args, **kwargs: pytest.fail("DeepMD server read its startup POSCAR"),
    )

    def build(tags, *, structure=None):
        captured["tags"] = dict(tags)
        captured["structure"] = structure
        return calculator

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", build)

    loaded, tags, _ = _load_backend_for_server(str(workdir), None)

    assert loaded is calculator
    assert tags["DEEPMD_TYPE_MAP"] == "Si,O"
    assert captured["structure"] is None


def test_stale_socket_is_replaced_but_live_server_is_rejected(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_path))
    stale.close()
    stale_pidfile = Path(pidfile_path(str(socket_path)))
    stale_pidfile.write_text("unrelated metadata\n")
    assert socket_path.exists()

    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
    try:
        assert stale_pidfile.read_text() == "unrelated metadata\n"
        with pytest.raises(ServerAlreadyRunning, match="already running"):
            prepare_socket_path(str(socket_path))
    finally:
        _stop_server(socket_path, thread)


def test_pidfile_path_appends_suffix_without_socket_name_collisions(tmp_path: Path):
    cpu_socket = tmp_path / "model.cpu"
    gpu_socket = tmp_path / "model.gpu"

    assert pidfile_path(str(cpu_socket)) == f"{cpu_socket}.pid"
    assert pidfile_path(str(gpu_socket)) == f"{gpu_socket}.pid"
    assert pidfile_path(str(cpu_socket)) != pidfile_path(str(gpu_socket))


def test_foreground_start_preserves_pidfile_when_socket_is_absent(tmp_path: Path):
    socket_path = tmp_path / "service.sock"
    unrelated_pidfile = Path(pidfile_path(str(socket_path)))
    unrelated_pidfile.write_text("external supervisor metadata\n")

    prepare_socket_path(str(socket_path))

    assert unrelated_pidfile.read_text() == "external supervisor metadata\n"


def test_unresponsive_socket_with_live_server_pidfile_is_not_stolen(
    tmp_path: Path, monkeypatch
):
    # R136 (P3): a FOREGROUND server draining an uninterruptible job after
    # `stop --force` keeps its socket file but stops answering (listener
    # closed, worker computing). A second `vpmdk serve` classified that as a
    # stale socket, unlinked the LIVE server's socket, and loaded a second
    # resident model beside it -- the daemon path refused via its pidfile, but
    # serve_cli passed pidfile=None for foreground, so the protection was
    # one-sided. Foreground serves now write the pidfile too, and
    # prepare_socket_path consults it BEFORE unlinking: an unresponsive socket
    # whose pidfile names a live vpmdk serve for this socket must be refused,
    # not stolen.
    socket_path = tmp_path / "service.sock"
    drained = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    drained.bind(str(socket_path))
    drained.close()  # bound then closed: connect() now refuses, like a drain
    pidfile = Path(pidfile_path(str(socket_path)))
    pidfile.write_text(
        f"{os.getpid()}\nsocket={os.path.realpath(str(socket_path))}\n"
    )
    # The pytest process is not a `vpmdk serve`, so positively identify it as
    # one for this check only (the real scenario's cmdline carries
    # 'serve ... --socket <path>').
    monkeypatch.setattr(
        server_module,
        "_pid_is_live_server_for_socket",
        lambda pid, sock, **kwargs: pid == os.getpid(),
    )

    with pytest.raises(server_module.ServerAlreadyRunning, match="still running"):
        prepare_socket_path(str(socket_path))

    # Neither the live server's socket nor its pidfile was touched.
    assert socket_path.exists()
    assert pidfile.exists()


def test_serve_cli_writes_pidfile_for_foreground_servers(
    tmp_path: Path, prepare_inputs, monkeypatch
):
    # Companion to the drain test above: the pidfile is the only liveness
    # evidence that survives a force-stop drain, so the FOREGROUND path must
    # write one too (it used to be daemon-only, which made the drain
    # protection one-sided).
    workdir = tmp_path / "work"
    workdir.mkdir()
    prepare_inputs(workdir, incar_overrides={"NSW": "0"})
    socket_path = tmp_path / "cli.sock"
    monkeypatch.setattr(
        vpmdk, "_build_calculator_from_tags", lambda *a, **k: DummyCalculator()
    )

    results: list[int] = []
    serve_thread = threading.Thread(
        target=lambda: results.append(
            vpmdk.main(
                ["serve", "--dir", str(workdir), "--socket", str(socket_path)]
            )
        ),
        daemon=True,
    )
    serve_thread.start()
    try:
        _wait_for(socket_path.exists)
        pidfile = Path(pidfile_path(str(socket_path)))
        _wait_for(pidfile.exists)
        content = pidfile.read_text()
        assert content.startswith(f"{os.getpid()}\n")
        assert f"socket={os.path.realpath(str(socket_path))}" in content
    finally:
        VPMDKClient(str(socket_path)).stop(timeout=10.0)
        serve_thread.join(timeout=10.0)

    assert results == [0]
    # A clean stop removes the pidfile with the socket.
    assert not Path(pidfile_path(str(socket_path))).exists()


def test_pidfile_records_start_time_and_identifies_default_socket_serves(
    tmp_path: Path,
):
    # R137 (P2): the R136 drain protection identified a live server only by
    # finding the socket path inside /proc/<pid>/cmdline -- but a DEFAULT-
    # socket `vpmdk serve` (no --socket argument) carries no socket path in
    # its cmdline at all, so the protection was silently inapplicable to the
    # plain invocation: a second serve during a force-stop drain unlinked the
    # LIVE server's socket and pidfile and double-loaded the model. The
    # pidfile now records the writer's kernel start time; (pid, starttime) is
    # a recycle-proof process identity that needs no cmdline at all.
    socket_path = tmp_path / "service.sock"
    my_start = server_module._process_start_time(os.getpid())
    assert my_start is not None  # Linux test host

    # This pytest process's cmdline contains no 'serve' and no socket path,
    # exactly like a default-socket serve -- yet the starttime identifies it.
    assert server_module._pid_is_live_server_for_socket(
        os.getpid(), str(socket_path), start_time=my_start
    )
    # A recycled pid (same number, different process) has a different
    # starttime and must NOT block a restart.
    assert not server_module._pid_is_live_server_for_socket(
        os.getpid(), str(socket_path), start_time="1"
    )

    # End-to-end through prepare_socket_path: an unresponsive socket whose
    # 3-line pidfile names this live process is refused, not stolen.
    drained = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    drained.bind(str(socket_path))
    drained.close()
    pidfile = Path(pidfile_path(str(socket_path)))
    pidfile.write_text(
        f"{os.getpid()}\n"
        f"socket={os.path.realpath(str(socket_path))}\n"
        f"starttime={my_start}\n"
    )
    with pytest.raises(server_module.ServerAlreadyRunning, match="still running"):
        prepare_socket_path(str(socket_path))
    assert socket_path.exists()
    assert pidfile.exists()


def test_daemon_startup_failure_strips_the_error_marker(tmp_path: Path, capsys, monkeypatch):
    # R141 (P3): the readiness-pipe children prepend 'ERROR:' but the parent
    # stripped only READY:/TIMEOUT:, so every failed daemon start printed the
    # internal marker twice: 'Error: daemon failed to start: ERROR:...'.
    monkeypatch.setattr(
        server_module,
        "_daemonize",
        lambda *a, **k: (True, None, "ERROR:RuntimeError: boom"),
    )
    args = SimpleNamespace(
        command="serve",
        dir=str(tmp_path),
        bcar=None,
        socket=str(tmp_path / "d.sock"),
        idle_timeout=0.0,
        daemon=True,
        log_file=None,
        daemon_notify_fd=None,
    )
    assert server_module.serve_cli(args) == 1
    err = capsys.readouterr().err
    assert "daemon failed to start: RuntimeError: boom" in err
    assert "ERROR:RuntimeError" not in err


def test_externally_deleted_socket_with_live_server_refuses_before_load(
    tmp_path: Path,
):
    # R143 (P3): "no socket => no live server" is false when a /tmp ager or
    # an accidental rm deleted the socket under a LIVING server -- the
    # restart passed prepare_socket_path silently, paid for the FULL model
    # load, and only then failed in _write_pidfile with a message naming no
    # pid. The pidfile still names the live process, so the refusal is fully
    # decidable up front.
    # The live owner must be a FOREIGN process: a pidfile naming THIS process
    # is serve_cli's own pre-load endpoint reservation and passes through
    # (see test_serve_cli_reserves_the_endpoint_before_the_model_load).
    socket_path = tmp_path / "gone.sock"  # never created: deleted externally
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        child_start = server_module._process_start_time(child.pid)
        assert child_start is not None
        pidfile = Path(pidfile_path(str(socket_path)))
        pidfile.write_text(
            f"{child.pid}\n"
            f"socket={os.path.realpath(str(socket_path))}\n"
            f"starttime={child_start}\n"
        )
        with pytest.raises(
            server_module.ServerAlreadyRunning, match=f"pid {child.pid}"
        ):
            prepare_socket_path(str(socket_path))
        assert pidfile.exists()  # the live owner's pidfile is untouched
    finally:
        child.kill()
        child.wait()

    # THIS process's own record is the pre-load reservation, not a foreign
    # deaf resident: startup proceeds.
    my_start = server_module._process_start_time(os.getpid())
    pidfile.write_text(
        f"{os.getpid()}\n"
        f"socket={os.path.realpath(str(socket_path))}\n"
        f"starttime={my_start}\n"
    )
    prepare_socket_path(str(socket_path))


def test_unusable_pidfile_is_refused_before_the_model_load(tmp_path: Path):
    # R145 (P3): a pre-existing unusable pidfile (foreign content, FIFO)
    # aborted `serve` only AFTER the full model load, inside _write_pidfile.
    # The refusal is decidable up front; prepare_socket_path now performs it
    # for pidfile-using launches (serve_cli), while library servers running
    # WITHOUT a pidfile leave a foreign file alone as before.
    socket_path = tmp_path / "pre.sock"
    pidfile = Path(pidfile_path(str(socket_path)))
    pidfile.write_text("external supervisor metadata\n")
    with pytest.raises(RuntimeError, match="not owned by this VPMDK socket"):
        prepare_socket_path(str(socket_path), pidfile_expected=True)
    assert pidfile.read_text() == "external supervisor metadata\n"  # preserved
    # Default (no pidfile planned): unchanged tolerant behavior.
    prepare_socket_path(str(socket_path))

    pidfile.unlink()
    os.mkfifo(pidfile)
    with pytest.raises(RuntimeError, match="non-regular pidfile"):
        prepare_socket_path(str(socket_path), pidfile_expected=True)

    # R146 (P3): a WELL-FORMED pidfile recording a DIFFERENT socket (a moved
    # runtime directory) passed the first version of this gate and paid the
    # full model load before _write_pidfile refused -- on every retry, since
    # the stale sweep deliberately preserves a different-socket file. Both
    # halves of _write_pidfile's refusal are now mirrored.
    pidfile.unlink()
    pidfile.write_text("999999\nsocket=/somewhere/else.sock\n")
    with pytest.raises(RuntimeError, match="not owned by this VPMDK socket"):
        prepare_socket_path(str(socket_path), pidfile_expected=True)


def test_named_log_file_may_be_a_symlink_even_at_the_default_name(tmp_path: Path):
    # R148 (P3): the foreground symlink refusal keyed on the log path STRING
    # equalling <socket>.log, so `vpmdk serve --log-file X.log` (the default
    # name spelled out) died after the full model load while the identical
    # --daemon line accepted it. serve_cli now threads log_file_named=True
    # for any explicit --log-file; library callers passing the derived path
    # WITHOUT the flag keep the hardened refusal (planted-symlink test).
    socket_path = tmp_path / "n.sock"
    real_log = tmp_path / "rotated" / "real.log"
    real_log.parent.mkdir()
    real_log.write_text("")
    derived = server_module.default_log_path(str(socket_path))
    os.symlink(str(real_log), derived)

    server = VPMDKServer(
        str(socket_path),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        log_file=derived,
        log_file_named=True,
    )
    server.logger.info("through the symlink")
    assert "through the symlink" in real_log.read_text()


def test_foreign_owned_pidfile_is_refused_before_the_model_load(
    tmp_path: Path, monkeypatch
):
    # R148 (P3, third half-mirror of this pair): _write_pidfile refuses a
    # foreign-owned pidfile, but the pre-load gate omitted that condition, so
    # a pidfile planted by another user in a shared sticky directory passed
    # the gate and aborted only after the full model load -- on every retry.
    shared = tmp_path / "shared"
    shared.mkdir()
    socket_path = shared / "own.sock"
    pidfile = Path(pidfile_path(str(socket_path)))
    pidfile.write_text(f"4242\nsocket={os.path.realpath(str(socket_path))}\n")
    real_uid = os.geteuid()
    monkeypatch.setattr(server_module.os, "geteuid", lambda: real_uid + 1)
    # In the real scenario the directory is a WRITABLE shared sticky /tmp and
    # only the foreign file's unlink hits EPERM. Simulate exactly that at the
    # unlink call (a read-only directory would instead trip the R149
    # unwritable-parent gate first, which rejects for a different reason).
    real_unlink = os.unlink

    def sticky_unlink(path, *args, **kwargs):
        if os.path.realpath(str(path)) == os.path.realpath(str(pidfile)):
            raise PermissionError(1, "Operation not permitted", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(server_module.os, "unlink", sticky_unlink)
    with pytest.raises(RuntimeError, match="owned by uid"):
        prepare_socket_path(str(socket_path), pidfile_expected=True)


def test_zombie_server_process_does_not_block_restart(tmp_path: Path):
    # R138 (P2): a SIGKILLed serve whose supervisor never called wait() stays
    # in /proc as a ZOMBIE with its original starttime, so the R137
    # (pid, starttime) identity matched and prepare_socket_path refused every
    # restart with "refusing to replace it while its process holds the model"
    # -- indefinitely, for a process holding nothing. State 'Z' now reads as
    # not-live.
    child = os.fork()
    if child == 0:
        os._exit(0)
    try:
        # Do NOT wait: the child stays a zombie for the duration of this test.
        _wait_for(
            lambda: (server_module._process_stat_fields(child) or ("", ""))[0] == "Z"
        )
        state, start_time = server_module._process_stat_fields(child)
        assert state == "Z"

        socket_path = tmp_path / "service.sock"
        assert not server_module._pid_is_live_server_for_socket(
            child, str(socket_path), start_time=start_time
        )

        # End-to-end: stale socket + 3-line pidfile naming the zombie must be
        # cleaned up, not refused.
        drained = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        drained.bind(str(socket_path))
        drained.close()
        pidfile = Path(pidfile_path(str(socket_path)))
        pidfile.write_text(
            f"{child}\n"
            f"socket={os.path.realpath(str(socket_path))}\n"
            f"starttime={start_time}\n"
        )
        prepare_socket_path(str(socket_path))
        assert not socket_path.exists()
        assert not pidfile.exists()
    finally:
        os.waitpid(child, 0)


def test_client_run_from_deleted_cwd_reports_clean_error(monkeypatch):
    # R137 (P3): `vpmdk run` from a deleted working directory died with a raw
    # FileNotFoundError traceback from os.getcwd() while building the
    # request's caller_cwd field -- exit 1 only via the interpreter's
    # uncaught-exception default -- while status/stop from the same state
    # printed the clean one-line 'Error:' diagnostic. The guarded helper maps
    # it to ValueError, which client_cli reports cleanly as exit 1.
    def deleted_cwd():
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(lightweight_client_module.os, "getcwd", deleted_cwd)
    with pytest.raises(ValueError, match="no longer exists"):
        lightweight_client_module._current_directory_for_request()


def test_stale_vpmdk_pidfile_with_reused_pid_is_removed(tmp_path: Path):
    # After an ungraceful death, a leftover VPMDK pidfile may name a PID the OS
    # has since recycled to a live process. Since staleness is keyed on the socket
    # (SERVER_MODE_SPEC 2.1), once the socket is gone/stale that pidfile is stale
    # too and must be removed, so a legitimate restart is not blocked by a false
    # ServerAlreadyRunning at _write_pidfile. A LIVE PID (this process) is used to
    # prove removal does not depend on the recorded PID being dead.
    socket_path = tmp_path / "service.sock"
    stale_pidfile = Path(pidfile_path(str(socket_path)))
    stale_pidfile.write_text(
        f"{os.getpid()}\nsocket={os.path.realpath(str(socket_path))}\n"
    )

    # Socket absent -> no live server -> the well-formed pidfile is stale.
    prepare_socket_path(str(socket_path))
    assert not stale_pidfile.exists()

    # A stale *socket* file (bound then closed) with a matching pidfile: both are
    # removed so the restart proceeds.
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_path))
    stale.close()
    stale_pidfile.write_text(
        f"{os.getpid()}\nsocket={os.path.realpath(str(socket_path))}\n"
    )
    prepare_socket_path(str(socket_path))
    assert not socket_path.exists()
    assert not stale_pidfile.exists()


def test_zero_length_pidfile_is_removed_so_daemon_restart_is_not_blocked(tmp_path: Path):
    # _write_pidfile's O_CREAT|O_EXCL creates an empty file and flushes the bytes
    # only on close, so a SIGKILL/power loss in that window (or ext4 delayed
    # allocation after a crash) leaves a ZERO-LENGTH <socket>.pid. If it survived,
    # the next `serve --daemon` would hit _write_pidfile's FileExistsError branch,
    # read empty (metadata None), and raise "not owned by this VPMDK socket" --
    # permanently blocking restart. Since the socket is absent, the empty pidfile
    # is stale and must be removed; _write_pidfile must then succeed.
    from vpmdk_core.server import _write_pidfile

    socket_path = tmp_path / "service.sock"
    pidfile = Path(pidfile_path(str(socket_path)))
    pidfile.write_bytes(b"")  # zero-length crash residue, socket absent

    prepare_socket_path(str(socket_path))
    assert not pidfile.exists(), "empty pidfile left in place -> daemon restart blocked"

    # The restart's pidfile write now proceeds instead of raising.
    _write_pidfile(str(pidfile), str(socket_path))
    assert pidfile.exists()
    text = pidfile.read_text()
    assert text.splitlines()[0] == str(os.getpid())


def test_nonempty_unparseable_pidfile_is_preserved_not_clobbered(tmp_path: Path):
    # A ZERO-LENGTH file is treated as VPMDK's own crash residue, but a NON-empty
    # file that simply does not parse as a VPMDK pidfile (an external supervisor's
    # own metadata, or non-UTF-8 bytes) carries data and must NOT be clobbered --
    # only removed once _write_pidfile can prove ownership. Socket absent so the
    # stale-pidfile path runs.
    socket_path = tmp_path / "service.sock"
    pidfile = Path(pidfile_path(str(socket_path)))

    for payload in (b"external supervisor metadata\n", b"12345\n", b"\xff\xfe\x00garbage"):
        pidfile.write_bytes(payload)
        prepare_socket_path(str(socket_path))
        assert pidfile.exists(), f"non-empty pidfile wrongly removed: {payload!r}"
        assert pidfile.read_bytes() == payload
    pidfile.unlink()


def test_unparseable_request_does_not_wedge_the_accept_sequence(tmp_path: Path):
    # json.loads raises beyond the protocol error types (RecursionError on deeply
    # nested input). If such an escape skipped _finish_handler_turn, every later
    # connection would block forever in _wait_for_handler_turn.
    socket_path = tmp_path / "wedge.sock"
    _, thread = _start_server(socket_path)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(10.0)
            connection.connect(str(socket_path))
            connection.sendall(("[" * 20000).encode("utf-8") + b"\n")
            try:
                connection.recv(65536)
            except OSError:
                pass

        # The server must still serve subsequent clients.
        status = VPMDKClient(str(socket_path)).status(timeout=10.0)
        assert status["state"] in {"idle", "busy"}
    finally:
        _stop_server(socket_path, thread)


def test_event_sender_construction_failure_does_not_wedge_the_server(tmp_path: Path):
    # _EventSender allocates a Lock in _handle_connection; if that raises (e.g.
    # MemoryError) and the construction sat OUTSIDE the try, the finally would not
    # run: the connection's accept-order turn would never finish (blocking every
    # later run) and _active_connections would never decrement (blocking graceful
    # stop / idle-timeout). Simulate the failure and assert the turn advances and
    # the in-flight count clears anyway.
    server = VPMDKServer(
        str(tmp_path / "s.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )
    sequence = server._next_accept_sequence
    with server._state_lock:
        server._active_connections += 1
    server._next_accept_sequence += 1  # serve_forever advances after thread start

    original = server_module._EventSender

    def _boom(_connection):
        raise MemoryError("simulated allocation failure")

    server_module._EventSender = _boom
    left, right = socket.socketpair()
    try:
        before = server._next_handler_sequence
        server._handle_connection(left, sequence)
        assert server._next_handler_sequence == before + 1  # turn finished
        assert server._active_connections == 0  # in-flight mark cleared
    finally:
        server_module._EventSender = original
        left.close()
        right.close()
        server._cleanup()


def test_execute_job_setup_failure_does_not_wedge_the_worker(tmp_path: Path):
    # _execute_job allocates threading.Event()/Thread()/_LineEventWriter before
    # running the job. If one raises (e.g. MemoryError) OUTSIDE the guarded try,
    # the sole worker dies with _busy still True: the client hangs (no terminal
    # event) and every later run + graceful stop + idle-timeout wedges forever.
    # The construction must sit inside the try so the finally always resets _busy
    # and delivers a terminal `done` event.
    server = VPMDKServer(
        str(tmp_path / "s.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )
    left, right = socket.socketpair()
    sender = server_module._EventSender(left)
    with server._state_lock:
        server._busy = True
        server._current_sender = sender
    job = server_module._RunJob(
        workdir=str(tmp_path), caller_cwd=str(tmp_path), sender=sender, enqueued_at=0.0
    )

    original = server_module._LineEventWriter

    def _boom(_sender):
        raise MemoryError("simulated allocation failure")

    server_module._LineEventWriter = _boom
    try:
        server._execute_job(job)  # must not raise out (worker survives)
        assert server._busy is False  # in-flight state reset
        assert server._current_sender is None
        right.settimeout(1.0)
        delivered = right.recv(65536).decode("utf-8")
        assert '"event": "done"' in delivered or '"event":"done"' in delivered
    finally:
        server_module._LineEventWriter = original
        left.close()
        right.close()
        server._cleanup()


def test_execute_job_rng_snapshot_failure_does_not_wedge_the_worker(tmp_path: Path):
    # The per-request RNG snapshot (np.random.get_state()) and the default
    # terminal-event dict are real allocations; they must sit INSIDE the guarded
    # try so a MemoryError there still runs the finally (resets _busy, sends a
    # terminal event, closes the sender) rather than wedging the sole worker with
    # _busy=True and a hung client.
    server = VPMDKServer(
        str(tmp_path / "s.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )
    left, right = socket.socketpair()
    sender = server_module._EventSender(left)
    with server._state_lock:
        server._busy = True
        server._current_sender = sender
    job = server_module._RunJob(
        workdir=str(tmp_path), caller_cwd=str(tmp_path), sender=sender, enqueued_at=0.0
    )

    class _BoomRandom:
        def get_state(self):
            raise MemoryError("simulated RNG snapshot allocation failure")

        def set_state(self, _state):
            pass

    class _BoomNumpy:
        random = _BoomRandom()

    original_np = server_module._np
    server_module._np = _BoomNumpy()
    try:
        server._execute_job(job)  # must not raise out
        assert server._busy is False
        assert server._current_sender is None
        right.settimeout(1.0)
        assert right.recv(65536)  # sender closed / terminal event delivered
    finally:
        server_module._np = original_np
        left.close()
        right.close()
        server._cleanup()


def test_worker_survives_exception_handler_allocation_failure(tmp_path: Path):
    # Even if _execute_job's OWN except handler raises (e.g. MemoryError while
    # traceback_module.format_exc() reports a job failure), that exception must
    # not escape _worker_loop and terminate the sole worker (which would wedge the
    # server: later runs queue unprocessed, graceful stop / idle-timeout hang).
    # _worker_loop's catch-all recovers; _execute_job's finally already delivered
    # a terminal event so the failed job's client is not left hanging.
    socket_path = tmp_path / "server.sock"
    failing = tmp_path / "failing"
    ok_dir = tmp_path / "ok"
    failing.mkdir()
    ok_dir.mkdir()
    state = {"boom": False}
    original_format_exc = server_module.traceback_module.format_exc

    def flaky_format_exc(*args, **kwargs):
        if state["boom"]:
            state["boom"] = False
            raise MemoryError("simulated OOM while reporting the error")
        return original_format_exc(*args, **kwargs)

    def executor(workdir: str, *, calculator) -> None:
        if os.path.realpath(workdir) == os.path.realpath(str(failing)):
            state["boom"] = True  # the except handler's format_exc will now raise
            raise RuntimeError("job failure that reaches the except handler")
        print("Calculation completed.")

    server_module.traceback_module.format_exc = flaky_format_exc
    _, thread = _start_server(socket_path, executor=executor)
    try:
        # First job fails and its except handler hits the injected MemoryError.
        with pytest.raises(RemoteCalculationError):
            VPMDKClient(str(socket_path)).run(str(failing))
        # The worker must have survived: a subsequent job still completes.
        result = VPMDKClient(str(socket_path)).run(str(ok_dir), timeout=10.0)
        assert result["ok"] is True
        assert VPMDKClient(str(socket_path)).status(timeout=2.0)["state"] == "idle"
    finally:
        server_module.traceback_module.format_exc = original_format_exc
        _stop_server(socket_path, thread)


def test_serialize_event_survives_surrogate_escaped_text():
    # Paths decoded with errors="surrogateescape" have no UTF-8 encoding. If the
    # encode raised, it escaped _EventSender.send and _execute_job's finally and
    # killed the worker thread, wedging the server permanently.
    event = {"event": "error", "error": "Is a directory: '/data/caf\udce9/BCAR'"}

    payload = server_module._serialize_event(event)

    assert json.loads(payload.decode("utf-8"))["error"] == event["error"]


def _protocol_payloads_ok(payloads: list[bytes]) -> bool:
    # The client checks the payload excluding the trailing newline.
    for payload in payloads:
        body = payload[:-1] if payload.endswith(b"\n") else payload
        assert len(body) <= server_module.MAX_REQUEST_BYTES
        json.loads(body)  # must remain valid JSON
    return True


def test_oversized_log_event_with_surrogates_splits_without_crashing():
    # The oversized-event path (_fit_event_text) must apply the same
    # ensure_ascii fallback as _serialize_event: a surrogate-escaped byte in a
    # line longer than the limit used to raise UnicodeEncodeError, be swallowed
    # by send(), and drop the terminal event.
    line = "a" * server_module.MAX_REQUEST_BYTES + "\udce9" + "b" * 128
    chunks = server_module._split_log_event({"event": "log", "line": line})

    assert len(chunks) > 1
    assert _protocol_payloads_ok(server_module._event_payloads({"event": "log", "line": line}))
    assert "".join(chunk["line"] for chunk in chunks if chunk.get("line")) == line


def test_oversized_terminal_event_with_surrogates_does_not_crash():
    # _truncate_event handles the oversized non-log "done" event; its byte
    # truncation of the error/traceback must tolerate surrogate-escaped bytes,
    # otherwise the terminal event is dropped and the client hangs with no error.
    marker = server_module.MAX_REQUEST_BYTES
    event = {
        "event": "done",
        "ok": False,
        "code": "calculation_error",
        "error": "FileNotFoundError: /data/caf\udce9/BCAR" + "x" * marker,
        "traceback": "Traceback\udce9" + "y" * marker,
    }

    assert _protocol_payloads_ok(server_module._event_payloads(event))


def test_oversized_event_sizes_ensure_ascii_from_all_fields():
    # A surrogate in one field forces _serialize_event to escape the WHOLE event
    # (ensure_ascii=True), lengthening other non-ASCII fields. _fit_event_text
    # must size with the same whole-event decision, or a large CJK sibling field
    # would push the real send over MAX_REQUEST_BYTES.
    event = {
        "event": "done",
        "ok": False,
        "code": "calculation_error",
        "error": "エラー詳細" * 200000,  # non-ASCII, non-surrogate, huge
        "traceback": "Traceback\udce9" + "z" * 200000,  # surrogate + huge
    }

    assert _protocol_payloads_ok(server_module._event_payloads(event))


def test_very_large_log_line_splits_in_bounded_time():
    # _split_log_event calls _fit_event_text once per chunk with the whole
    # remaining tail; an unbounded per-chunk scan of that tail is O(N^2) and
    # CPU-pins the job thread. A multi-megabyte line must split quickly and
    # losslessly.
    line = "x" * (24 * server_module.MAX_REQUEST_BYTES)
    started = time.monotonic()
    chunks = server_module._split_log_event({"event": "log", "line": line})
    elapsed = time.monotonic() - started

    assert _protocol_payloads_ok(
        server_module._event_payloads({"event": "log", "line": line})
    )
    assert "".join(c["line"] for c in chunks if c.get("line")) == line
    assert elapsed < 20.0


def test_small_surrogate_terminal_event_is_delivered():
    payloads = server_module._event_payloads(
        {"event": "done", "ok": False, "error": "bad path /x\udce9/model"}
    )
    assert _protocol_payloads_ok(payloads)


def test_event_sender_never_propagates_a_serialization_failure():
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sender = server_module._EventSender(left)
    try:
        # Unserializable payload must be swallowed, not raised at the caller.
        assert sender.send({"event": "log", "line": object()}) is False
    finally:
        left.close()
        right.close()


def test_deeply_nested_request_is_a_protocol_error_not_a_server_defect(
    tmp_path: Path,
):
    # Nesting depth is client-supplied, so it must not be logged as an internal
    # defect: a peer could otherwise flood the daemon log with tracebacks.
    socket_path = tmp_path / "nested.sock"
    server, thread = _start_server(socket_path)
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    server.logger.addHandler(_Capture())
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(10.0)
            connection.connect(str(socket_path))
            connection.sendall(("[" * 20000).encode("utf-8") + b"\n")
            payload = connection.recv(65536).decode()
    finally:
        _stop_server(socket_path, thread)

    assert '"code":"protocol_error"' in payload.replace(" ", "")
    assert not any("Unexpected failure" in message for message in messages)


def test_stray_orb_model_tag_does_not_leak_into_another_backend(tmp_path: Path):
    # ORB_MODEL names the resident model for ORB only; a leftover tag from a
    # switched backend must not make CHGNET advertise ORB's model.
    identity = backend_identity(
        {"MLP": "CHGNET", "ORB_MODEL": "orb-v3-conservative-20-omat", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )

    assert identity["mlp"] == "CHGNET"
    assert identity["model"] != "orb-v3-conservative-20-omat"


def test_blank_request_mlp_is_a_backend_mismatch_not_a_calculation_error(
    tmp_path: Path,
):
    # A blank MLP= in a request BCAR is a request-side selector problem (exit 5).
    resident = backend_identity(
        {"MLP": "CHGNET", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )

    with pytest.raises(BackendConfigurationMismatch):
        validate_request_backend(
            resident, {"MLP": ""}, request_base_dir=str(tmp_path)
        )


def test_request_read_uses_a_total_deadline(tmp_path: Path, monkeypatch):
    # A socket timeout resets on every recv, so a peer that dribbles bytes could
    # hold its accept-order turn indefinitely and wedge every later status/stop.
    # The whole read must be bounded by one deadline.
    monkeypatch.setattr(server_module, "REQUEST_READ_TIMEOUT", 0.3)
    server = VPMDKServer(
        str(tmp_path / "unused.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )

    class DribblingConnection:
        def settimeout(self, timeout):
            pass

        def recv(self, size):
            time.sleep(0.05)
            return b"x"  # never terminates the line

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        server._read_request(DribblingConnection())
    assert time.monotonic() - started < 5.0


def test_event_sender_close_preempts_a_blocked_send():
    # close() is force-shutdown's only preemption path. If it waited for the
    # lock held across a blocked sendall, a client that stopped draining would
    # stall shutdown for the whole (generous) send timeout.
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    left.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    right.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    left.settimeout(30.0)
    sender = server_module._EventSender(left)
    entered = threading.Event()

    def flood():
        entered.set()
        sender.send({"event": "log", "line": "x" * 4_000_000})

    worker = threading.Thread(target=flood, daemon=True)
    worker.start()
    assert entered.wait(5.0)
    time.sleep(0.3)  # let the send block on a full buffer

    started = time.monotonic()
    sender.close()
    elapsed = time.monotonic() - started

    worker.join(timeout=5.0)
    left.close()
    right.close()
    assert elapsed < 5.0, "close() waited for the in-flight send to time out"


def test_request_deadline_does_not_govern_event_delivery(tmp_path: Path):
    # The short request-read deadline must be replaced once the request is in,
    # otherwise a client that pauses briefly kills a successful job's stream.
    assert server_module.EVENT_SEND_TIMEOUT > server_module.REQUEST_READ_TIMEOUT
    server = VPMDKServer(
        str(tmp_path / "unused.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        right.sendall(json.dumps({"version": 1, "op": "status"}).encode() + b"\n")
        assert server._read_request(left)["op"] == "status"
        assert left.gettimeout() == server_module.EVENT_SEND_TIMEOUT

        # A malformed request must restore the send deadline too.
        right.sendall(b"not json\n")
        with pytest.raises(Exception):
            server._read_request(left)
        assert left.gettimeout() == server_module.EVENT_SEND_TIMEOUT
    finally:
        left.close()
        right.close()


def test_unexpected_handler_failure_is_logged_and_not_called_protocol_error(
    tmp_path: Path,
):
    # A server-side defect must be recorded and distinguishable from a client
    # protocol violation, otherwise it looks like a connectivity problem.
    socket_path = tmp_path / "internal.sock"
    server, thread = _start_server(socket_path)
    messages: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    # server.logger does not propagate, so attach directly to it.
    server.logger.addHandler(_Capture())

    def boom():
        raise RuntimeError("internal defect")

    server.status = boom
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(10.0)
            connection.connect(str(socket_path))
            connection.sendall(
                json.dumps({"version": 1, "op": "status"}).encode() + b"\n"
            )
            payload = connection.recv(65536).decode()
    finally:
        _stop_server(socket_path, thread)

    assert '"code":"internal_error"' in payload.replace(" ", "")
    assert "internal defect" in payload
    assert any("Unexpected failure" in message for message in messages)


def test_move_fd_above_stdio_relocates_low_descriptors():
    # os.pipe() hands out the lowest free fds, so a caller with closed standard
    # streams can get fd 1 or 2 — which the daemon's dup2 would then clobber.
    read_fd, write_fd = os.pipe()
    moved = None
    try:
        moved = server_module._move_fd_above_stdio(write_fd)
        assert moved > 2
        # Already-high descriptors are returned untouched.
        assert server_module._move_fd_above_stdio(moved) == moved
        os.write(moved, b"ok\n")
        assert os.read(read_fd, 3) == b"ok\n"
    finally:
        os.close(read_fd)
        if moved is not None:
            try:
                os.close(moved)
            except OSError:
                pass


def test_daemon_exec_argv_rebuilds_a_fresh_serve_command(tmp_path: Path):
    # The daemon child must exec a brand-new interpreter: importing vpmdk_core
    # starts the ML runtimes' native threads, and fork() clones only the calling
    # thread, so continuing in the forked process could deadlock while loading a
    # model. The rebuilt command must therefore be runnable standalone.
    args = SimpleNamespace(
        dir=str(tmp_path / "cfg"),
        bcar=None,
        idle_timeout=120.0,
    )

    argv = server_module._daemon_exec_argv(
        args, str(tmp_path / "s.sock"), str(tmp_path / "s.log")
    )

    import vpmdk_entry

    assert argv[0] == sys.executable
    # -u is required, not cosmetic: the daemon's stdout is a dup2'd regular file
    # (the log), which CPython block-buffers, so every print()-based startup
    # diagnostic -- including the SPEC 2.1 "BCAR not found" warning -- would sit
    # in that buffer for the life of the daemon and be lost entirely on SIGKILL.
    assert "-u" in argv[1:argv.index("serve")]
    # The entrypoint *script* is executed rather than `-m vpmdk_entry`, so
    # sys.path[0] becomes its directory and a source checkout keeps working
    # (the runtime sys.path insertion is not inherited across execv).
    script_index = argv.index("serve") - 1
    assert argv[script_index] == os.path.abspath(vpmdk_entry.__file__)
    assert os.path.isfile(argv[script_index])
    # Only interpreter flags may precede the script.
    assert all(item.startswith("-") for item in argv[1:script_index])
    assert argv[argv.index("--socket") + 1] == str(tmp_path / "s.sock")
    assert argv[argv.index("--log-file") + 1] == str(tmp_path / "s.log")
    assert argv[argv.index("--dir") + 1] == str(tmp_path / "cfg")
    assert float(argv[argv.index("--idle-timeout") + 1]) == 120.0
    assert "--daemon" in argv
    # _daemonize appends the inherited pipe fd; it is not part of the base argv.
    assert "--daemon-notify-fd" not in argv


def test_daemon_exec_argv_absolutizes_paths(tmp_path: Path, monkeypatch):
    # The daemon chdirs to "/", so every path it receives must be absolute.
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(dir="cfg", bcar="custom.bcar", idle_timeout=0.0)

    argv = server_module._daemon_exec_argv(
        args, str(tmp_path / "s.sock"), str(tmp_path / "s.log")
    )

    assert os.path.isabs(argv[argv.index("--dir") + 1])
    assert os.path.isabs(argv[argv.index("--bcar") + 1])


def test_daemon_exec_argv_expands_tilde_before_absolutizing(monkeypatch):
    # `~` must be expanded before abspath: abspath alone would produce
    # $PWD/~/... which the re-executed daemon's expanduser cannot recover, so a
    # BCAR/dir path that works in the foreground would be "not found" only under
    # --daemon.
    monkeypatch.chdir("/tmp")
    args = SimpleNamespace(
        dir="~/models/cfg", bcar="~/models/BCAR", idle_timeout=0.0
    )

    argv = server_module._daemon_exec_argv(args, "/tmp/s.sock", "/tmp/s.log")

    home = os.path.expanduser("~")
    resolved_dir = argv[argv.index("--dir") + 1]
    resolved_bcar = argv[argv.index("--bcar") + 1]
    assert resolved_dir == os.path.join(home, "models", "cfg")
    assert resolved_bcar == os.path.join(home, "models", "BCAR")
    assert "~" not in resolved_dir and "~" not in resolved_bcar


def test_load_backend_for_server_expands_tilde_in_dir_and_bcar(
    tmp_path: Path, monkeypatch
):
    # The daemon re-exec now expands `~`; the foreground path (which calls
    # _load_backend_for_server directly with the raw --dir/--bcar) must match,
    # otherwise `serve --dir '~/calc'` silently loads $PWD/~/calc/BCAR (not
    # found) and falls back to the default backend.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(tmp_path), 1))
    calc_dir = tmp_path / "calc"
    calc_dir.mkdir()
    (calc_dir / "BCAR").write_text("MLP=CHGNET\nDEVICE=cpu\n")
    captured: dict[str, object] = {}

    def build(tags, *, structure=None):
        captured["cwd"] = os.getcwd()
        captured["tags"] = dict(tags)
        return DummyCalculator()

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", build)

    # Quoted `--dir '~/calc'` reaches us as a literal '~/calc'.
    _, tags, base_dir = _load_backend_for_server("~/calc", None)

    assert captured["cwd"] == str(calc_dir)
    assert base_dir == str(calc_dir)
    assert "~" not in base_dir

    # And an explicit `--bcar '~/calc/BCAR'` resolves the same way.
    _, _, base_dir2 = _load_backend_for_server(str(tmp_path), "~/calc/BCAR")
    assert base_dir2 == str(calc_dir)
    assert "~" not in base_dir2


def test_server_log_file_expands_tilde(tmp_path: Path, monkeypatch):
    # A foreground `--log-file '~/logs/s.log'` must land in $HOME, not a literal
    # '~' directory under the current working directory.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home), 1))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    server = VPMDKServer(
        str(tmp_path / "s.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        log_file="~/logs/server.log",
    )
    try:
        handler = server.logger.handlers[0]
        assert isinstance(handler, logging.FileHandler)
        assert handler.baseFilename == str(home / "logs" / "server.log")
        assert (home / "logs").is_dir()
        # No stray literal '~' directory under CWD.
        assert not (cwd / "~").exists()
    finally:
        for h in list(server.logger.handlers):
            h.close()
            server.logger.removeHandler(h)


def test_daemon_exec_argv_forwards_explicit_bcar(tmp_path: Path):
    args = SimpleNamespace(
        dir=str(tmp_path), bcar=str(tmp_path / "custom.bcar"), idle_timeout=0.0
    )

    argv = server_module._daemon_exec_argv(
        args, str(tmp_path / "s.sock"), str(tmp_path / "s.log")
    )

    assert argv[argv.index("--bcar") + 1] == str(tmp_path / "custom.bcar")


def test_serve_parser_hides_internal_daemon_notify_fd(capsys):
    from vpmdk_core.cli import _server_parser

    parsed = _server_parser().parse_args(["serve", "--daemon-notify-fd", "7"])
    assert parsed.daemon_notify_fd == 7
    assert _server_parser().parse_args(["serve"]).daemon_notify_fd is None

    # Internal plumbing must not appear in user-facing help.
    with pytest.raises(SystemExit):
        _server_parser().parse_args(["serve", "--help"])
    assert "--daemon-notify-fd" not in capsys.readouterr().out


def test_serve_cli_reports_early_failures_over_the_readiness_pipe(
    tmp_path: Path, monkeypatch
):
    # The re-executed child's stderr goes to the log file, so the parent only
    # ever sees the pipe. Validation failures must therefore reach it instead of
    # surfacing as a bare "readiness pipe closed".
    monkeypatch.setattr(
        server_module,
        "prepare_socket_path",
        lambda path, **kwargs: (_ for _ in ()).throw(
            ServerAlreadyRunning("already running")
        ),
    )

    read_fd, write_fd = os.pipe()
    try:
        args = SimpleNamespace(
            command="serve",
            dir=str(tmp_path),
            bcar=None,
            socket=str(tmp_path / "s.sock"),
            idle_timeout=0.0,
            daemon=True,
            log_file=str(tmp_path / "s.log"),
            daemon_notify_fd=write_fd,
        )
        assert server_module.serve_cli(args) == 1
        message = os.read(read_fd, 4096).decode("utf-8", errors="replace")
    finally:
        os.close(read_fd)

    assert message.startswith("ERROR:")
    assert "already running" in message


def test_serve_cli_daemon_child_leaves_the_launch_directory(
    tmp_path: Path, monkeypatch
):
    # A resident server is routinely launched from a scratch directory that is
    # deleted afterwards; holding it would break every later job in
    # _working_directory's os.getcwd().
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    observed: dict[str, str] = {}

    class _StopHere(Exception):
        pass

    def _boom(*args, **kwargs):
        observed["cwd"] = os.getcwd()
        raise _StopHere()

    monkeypatch.setattr(server_module, "_load_backend_for_server", _boom)
    monkeypatch.setattr(
        os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )

    read_fd, write_fd = os.pipe()
    try:
        args = SimpleNamespace(
            command="serve",
            dir=str(tmp_path),
            bcar=None,
            socket=str(tmp_path / "s.sock"),
            idle_timeout=0.0,
            daemon=True,
            log_file=str(tmp_path / "s.log"),
            daemon_notify_fd=write_fd,
        )
        with pytest.raises(SystemExit):
            server_module.serve_cli(args)
    finally:
        os.close(read_fd)

    assert observed["cwd"] == "/"


def test_daemon_start_timeout_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv("VPMDK_DAEMON_START_TIMEOUT", raising=False)
    assert server_module._daemon_start_timeout() == 600.0

    monkeypatch.setenv("VPMDK_DAEMON_START_TIMEOUT", "12.5")
    assert server_module._daemon_start_timeout() == 12.5

    for bogus in ("not-a-number", "0", "-5", "inf"):
        monkeypatch.setenv("VPMDK_DAEMON_START_TIMEOUT", bogus)
        assert server_module._daemon_start_timeout() == 600.0


def test_serve_cli_does_not_refork_when_already_daemonized(
    tmp_path: Path, monkeypatch
):
    # A re-executed daemon child is already a detached session leader, so it
    # must report readiness on the inherited fd instead of forking again.
    daemonize_calls: list[object] = []
    monkeypatch.setattr(
        server_module,
        "_daemonize",
        lambda *a, **k: daemonize_calls.append(a),
    )

    class _StopHere(Exception):
        pass

    def _boom(*args, **kwargs):
        raise _StopHere()

    monkeypatch.setattr(server_module, "_load_backend_for_server", _boom)
    # finish() calls os._exit for a daemon child; make it observable instead.
    monkeypatch.setattr(
        os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )

    read_fd, write_fd = os.pipe()
    try:
        args = SimpleNamespace(
            command="serve",
            dir=str(tmp_path),
            bcar=None,
            socket=str(tmp_path / "s.sock"),
            idle_timeout=0.0,
            daemon=True,
            log_file=str(tmp_path / "s.log"),
            daemon_notify_fd=write_fd,
        )
        with pytest.raises(SystemExit):
            server_module.serve_cli(args)
        message = os.read(read_fd, 4096).decode("utf-8", errors="replace")
    finally:
        os.close(read_fd)

    assert daemonize_calls == []
    assert "_StopHere" in message


def test_daemon_pidfile_records_socket_ownership_and_is_cleaned(tmp_path: Path):
    socket_path = tmp_path / "daemon.sock"
    daemon_pidfile = Path(pidfile_path(str(socket_path)))
    _, thread = _start_server(socket_path, pidfile=daemon_pidfile)
    try:
        lines = daemon_pidfile.read_text().splitlines()
        # Third line (R137): the writer's kernel start time, so a restart can
        # tell the recorded process apart from a pid-recycled impostor -- and
        # can positively identify a default-socket serve whose cmdline names
        # no socket at all.
        assert lines[:2] == [str(os.getpid()), f"socket={socket_path}"]
        assert len(lines) == 3
        assert lines[2] == (
            f"starttime={server_module._process_start_time(os.getpid())}"
        )
    finally:
        _stop_server(socket_path, thread)

    assert not daemon_pidfile.exists()


def test_daemon_refuses_to_overwrite_unowned_pidfile(tmp_path: Path):
    socket_path = tmp_path / "daemon.sock"
    unrelated_pidfile = Path(pidfile_path(str(socket_path)))
    unrelated_pidfile.write_text("external supervisor metadata\n")
    server = VPMDKServer(
        str(socket_path),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        pidfile=str(unrelated_pidfile),
    )

    with pytest.raises(RuntimeError, match="not owned by this VPMDK socket"):
        server.serve_forever()

    assert unrelated_pidfile.read_text() == "external supervisor metadata\n"
    assert not socket_path.exists()


def test_socket_accepting_connections_is_never_unlinked_as_stale(tmp_path: Path):
    socket_path = tmp_path / "slow.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    accepted = threading.Event()
    release = threading.Event()

    def accept_without_replying() -> None:
        connection, _ = listener.accept()
        accepted.set()
        release.wait(2.0)
        connection.close()

    accept_thread = threading.Thread(target=accept_without_replying)
    accept_thread.start()
    inode = socket_path.stat().st_ino
    try:
        with pytest.raises(ServerAlreadyRunning, match="already running"):
            prepare_socket_path(str(socket_path))
        assert accepted.wait(1.0)
        assert socket_path.stat().st_ino == inode
    finally:
        release.set()
        accept_thread.join(timeout=2.0)
        listener.close()


def test_detected_device_rebuilds_effective_backend_configuration(
    tmp_path: Path,
    monkeypatch,
):
    calculator = DummyCalculator()
    calculator.device = "cpu"
    monkeypatch.setattr(
        vpmdk,
        "_resolve_device",
        lambda value: "cuda" if value is None else value,
    )

    server = VPMDKServer(
        str(tmp_path / "server.sock"),
        calculator,
        {"MLP": "UPET", "MODEL": "pet-oam-xl"},
        backend_base_dir=str(tmp_path),
    )

    assert server.backend["device"] == "cpu"
    assert server.backend["configuration"]["DEVICE"] == "cpu"
    assert server.backend["effective_configuration"]["DEVICE"] == "cpu"
    # On a CPU model the neighbor-list device resolves to "cpu" (default "auto"
    # and explicit "model"/"cpu" all run on CPU), recorded consistently.
    assert (
        server.backend["effective_configuration"]["UPET_NEIGHBORLIST_DEVICE"]
        == "cpu"
    )
    validate_request_backend(
        server.backend,
        {"DEVICE": "cpu", "UPET_NL_DEVICE": "model"},
        request_base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch, match="DEVICE"):
        validate_request_backend(
            server.backend,
            {"DEVICE": "cuda"},
            request_base_dir=str(tmp_path),
        )


def test_protocol_errors_do_not_stop_server(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)

    def request(payload: dict[str, object]) -> dict[str, object]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(socket_path))
            connection.sendall((json.dumps(payload) + "\n").encode())
            response = connection.makefile("rb").readline()
        return json.loads(response)

    try:
        wrong_version = request({"op": "status", "version": 99})
        unknown_op = request({"op": "unknown", "version": 1})
        invalid_force_string = request(
            {"op": "stop", "version": 1, "force": "false"}
        )
        invalid_force_number = request({"op": "stop", "version": 1, "force": 1})
        relative_workdir = request(
            {"op": "run", "version": 1, "workdir": "relative/calc"}
        )
        assert wrong_version["event"] == "error"
        assert unknown_op["event"] == "error"
        assert invalid_force_string["code"] == "protocol_error"
        assert invalid_force_number["code"] == "protocol_error"
        assert relative_workdir == {
            "event": "error",
            "code": "protocol_error",
            "error": "run.workdir must be an absolute path",
        }
        assert "JSON boolean" in invalid_force_string["error"]
        assert VPMDKClient(str(socket_path)).status()["state"] == "idle"
    finally:
        _stop_server(socket_path, thread)


def test_protocol_version_must_be_a_real_integer(tmp_path: Path):
    # A bare `version != PROTOCOL_VERSION` comparison ACCEPTS JSON `true` and
    # `1.0` as protocol 1, because Python's numeric tower makes True == 1 and
    # 1.0 == 1 (bool is an int subclass). A malformed or version-skewed peer must
    # get a protocol_error instead of being treated as speaking the supported
    # protocol, matching the strict isinstance checks the other request fields use.
    socket_path = tmp_path / "server.sock"
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)

    def request(payload: dict[str, object]) -> dict[str, object]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(socket_path))
            connection.sendall((json.dumps(payload) + "\n").encode())
            response = connection.makefile("rb").readline()
        return json.loads(response)

    try:
        # Values that are `== 1` but are NOT a JSON integer, plus plain skew.
        for version in (True, False, 1.0, "1", None, 2, [1], {"v": 1}):
            event = request({"op": "status", "version": version})
            assert event["event"] == "error", (version, event)
            assert event["code"] == "protocol_error", (version, event)

        # The genuine integer version is still accepted, and the server survives.
        accepted = request({"op": "status", "version": 1})
        assert accepted["event"] == "status"
        assert VPMDKClient(str(socket_path)).status()["state"] == "idle"
    finally:
        _stop_server(socket_path, thread)


def test_client_stop_rejects_non_boolean_force(tmp_path: Path):
    with pytest.raises(TypeError, match="force must be a boolean"):
        VPMDKClient(str(tmp_path / "server.sock")).stop(force="false")


def test_handler_thread_start_failure_does_not_stall_the_server(tmp_path: Path):
    # A failed handler Thread.start() must not consume the accept-order sequence
    # or leave the in-flight count orphaned: otherwise _next_handler_sequence
    # stalls and every later connection deadlocks in _wait_for_handler_turn.
    real_thread = server_module.threading.Thread
    state = {"failed": False}

    class _MaybeFailingThread:
        def __init__(self, *args, target=None, **kwargs):
            self._thread = real_thread(*args, target=target, **kwargs)
            self._target = target

        def start(self):
            if (
                getattr(self._target, "__name__", "") == "_handle_connection"
                and not state["failed"]
            ):
                state["failed"] = True
                raise RuntimeError("simulated thread exhaustion")
            return self._thread.start()

        def join(self, *args, **kwargs):
            return self._thread.join(*args, **kwargs)

        def is_alive(self):
            # A Thread stand-in must answer this: teardown polls the worker's
            # liveness so a repeated shutdown signal can be honoured.
            return self._thread.is_alive()

    socket_path = tmp_path / "s.sock"

    def executor(workdir: str, *, calculator) -> None:
        print("Calculation completed.")

    server_module.threading.Thread = _MaybeFailingThread
    try:
        server = VPMDKServer(
            str(socket_path),
            DummyCalculator(),
            {"MLP": "CHGNET", "DEVICE": "cpu"},
            backend_base_dir=str(socket_path.parent),
            heartbeat_interval=0.05,
            executor=executor,
        )
        thread = real_thread(target=server.serve_forever, daemon=True)
        thread.start()
        # serve_forever publishes self._listener only AFTER bind() creates the
        # socket file, so waiting on the file alone leaves a window in which a
        # connect gets ECONNREFUSED.
        _wait_for(lambda: socket_path.exists() and server._listener is not None)
        try:
            # First request's handler start() fails -> connection is closed.
            with pytest.raises(ServerConnectionError):
                VPMDKClient(str(socket_path)).run(str(tmp_path), timeout=5.0)
            # The server must still serve later requests (not deadlocked).
            VPMDKClient(str(socket_path)).run(str(tmp_path), timeout=10.0)
        finally:
            _stop_server(socket_path, thread)
    finally:
        server_module.threading.Thread = real_thread

    assert state["failed"] is True


def test_worker_start_failure_cleans_up_bound_socket(tmp_path: Path, monkeypatch):
    # If the worker thread fails to start, _worker must stay None so the finally
    # does not join an unstarted thread (which would raise and skip cleanup); the
    # bound socket must be removed and the real start error surfaced.
    socket_path = tmp_path / "s.sock"
    real_thread = server_module.threading.Thread

    class _WorkerFailingThread:
        def __init__(self, *args, target=None, **kwargs):
            self._target = target
            self._thread = real_thread(*args, target=target, **kwargs)

        def start(self):
            if getattr(self._target, "__name__", "") == "_worker_loop":
                raise RuntimeError("simulated thread exhaustion")
            return self._thread.start()

        def join(self, *args, **kwargs):
            return self._thread.join(*args, **kwargs)

        def is_alive(self):
            # A Thread stand-in must answer this: teardown polls the worker's
            # liveness so a repeated shutdown signal can be honoured.
            return self._thread.is_alive()

    server = VPMDKServer(
        str(socket_path),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )
    monkeypatch.setattr(server_module.threading, "Thread", _WorkerFailingThread)

    with pytest.raises(RuntimeError, match="simulated thread exhaustion"):
        server.serve_forever()

    assert not socket_path.exists()


def test_event_stream_rejects_oversized_leftover_without_blocking():
    # An oversized unterminated event trailing a valid newline-terminated one in
    # the same recv() must be rejected after yielding the valid event, not left
    # for a subsequent (possibly no-timeout) recv() that would block forever.
    valid = json.dumps({"event": "log", "line": "x"}).encode() + b"\n"
    oversized = b"x" * (lightweight_client_module.MAX_REQUEST_BYTES + 16)

    class _FakeConn:
        def __init__(self, chunk):
            self._chunk = chunk
            self._calls = 0

        def settimeout(self, _timeout):
            pass

        def recv(self, _n):
            self._calls += 1
            if self._calls == 1:
                return self._chunk
            raise AssertionError("recv called again: oversized leftover not caught")

    events: list[dict] = []
    with pytest.raises(lightweight_client_module.ProtocolError, match="size limit"):
        for event in lightweight_client_module.VPMDKClient._event_stream(
            _FakeConn(valid + oversized), deadline=None
        ):
            events.append(event)

    assert events == [{"event": "log", "line": "x"}]


def test_event_stream_maps_nonstandard_json_failures_to_protocol_error():
    # json.loads can reject a malformed peer event line in ways that are neither
    # JSONDecodeError nor UnicodeDecodeError: RecursionError on deep nesting, and
    # (Python 3.11+) a PLAIN ValueError on an over-4300-digit integer literal. All
    # must map to ProtocolError (mirroring the server) so a malformed/hostile peer
    # yields the exit-3 contract, not an uncaught traceback + exit 1.
    class _FakeConn:
        def __init__(self, chunk):
            self._chunk = chunk

        def settimeout(self, _timeout):
            pass

        def recv(self, _n):
            chunk, self._chunk = self._chunk, b""
            return chunk

    deeply_nested = b"[" * 200_000 + b"\n"  # -> RecursionError
    huge_integer = b'{"event":"done","ok":' + b"9" * 5000 + b"}\n"  # -> ValueError
    for line in (deeply_nested, huge_integer):
        assert len(line) < lightweight_client_module.MAX_REQUEST_BYTES
        with pytest.raises(
            lightweight_client_module.ProtocolError, match="Invalid server JSON event"
        ):
            list(
                lightweight_client_module.VPMDKClient._event_stream(
                    _FakeConn(line), deadline=None
                )
            )


def test_heartbeat_thread_start_failure_does_not_wedge_the_worker(tmp_path: Path):
    # heartbeat.start() precedes _execute_job's guarded block; if it raises
    # (thread exhaustion) the exception must not escape the worker, which would
    # leave _busy set and queue every later request forever. Heartbeats are
    # best-effort, so the job still runs and the worker keeps serving.
    real_thread = server_module.threading.Thread
    state = {"failed": False}

    class _HeartbeatFailingThread:
        def __init__(self, *args, target=None, **kwargs):
            self._target = target
            self._thread = real_thread(*args, target=target, **kwargs)

        def start(self):
            if (
                getattr(self._target, "__name__", "") == "_heartbeat"
                and not state["failed"]
            ):
                state["failed"] = True
                raise RuntimeError("simulated thread exhaustion")
            return self._thread.start()

        def join(self, *args, **kwargs):
            return self._thread.join(*args, **kwargs)

        def is_alive(self):
            # A Thread stand-in must answer this: teardown polls the worker's
            # liveness so a repeated shutdown signal can be honoured.
            return self._thread.is_alive()

    socket_path = tmp_path / "s.sock"
    completed: list[str] = []

    def executor(workdir: str, *, calculator) -> None:
        completed.append(Path(workdir).name)
        print("Calculation completed.")

    server_module.threading.Thread = _HeartbeatFailingThread
    try:
        server = VPMDKServer(
            str(socket_path),
            DummyCalculator(),
            {"MLP": "CHGNET", "DEVICE": "cpu"},
            backend_base_dir=str(socket_path.parent),
            heartbeat_interval=0.05,
            executor=executor,
        )
        thread = real_thread(target=server.serve_forever, daemon=True)
        thread.start()
        # serve_forever publishes self._listener only AFTER bind() creates the
        # socket file, so waiting on the file alone leaves a window in which a
        # connect gets ECONNREFUSED.
        _wait_for(lambda: socket_path.exists() and server._listener is not None)
        try:
            first = tmp_path / "first"
            first.mkdir()
            second = tmp_path / "second"
            second.mkdir()
            # First job's heartbeat fails to start, but the job must still finish.
            VPMDKClient(str(socket_path)).run(str(first), timeout=10.0)
            # The worker must remain alive for a subsequent request.
            VPMDKClient(str(socket_path)).run(str(second), timeout=10.0)
        finally:
            _stop_server(socket_path, thread)
    finally:
        server_module.threading.Thread = real_thread

    assert state["failed"] is True
    assert completed == ["first", "second"]


def test_idle_timeout_waits_for_in_flight_connections(tmp_path: Path):
    # A connection accepted right at the timeout boundary must not be dropped:
    # while its handler is still starting up (not yet enqueued, _last_activity
    # not yet touched), the server must not consider itself idle.
    server = VPMDKServer(
        str(tmp_path / "s.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        idle_timeout=0.01,
    )
    with server._state_lock:
        server._active_connections = 1
        server._last_activity = time.monotonic() - 100.0  # well past the timeout

    assert server._should_exit() is False

    with server._state_lock:
        server._active_connections = 0
    assert server._should_exit() is True


def test_line_event_writer_splits_multi_line_writes_once():
    sent: list[str] = []

    class _FakeSender:
        def send(self, event) -> None:
            sent.append(event["line"])

    writer = server_module._LineEventWriter(_FakeSender())

    writer.write("a\nb\nc")
    assert sent == ["a", "b"]
    assert writer._pending == "c"

    writer.write("d\ne\n")
    assert sent == ["a", "b", "cd", "e"]
    assert writer._pending == ""

    writer.write("tail-no-newline")
    assert sent == ["a", "b", "cd", "e"]

    writer.flush()
    assert sent == ["a", "b", "cd", "e", "tail-no-newline"]
    assert writer._pending == ""


def test_idle_timeout_removes_socket(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    _, thread = _start_server(socket_path, idle_timeout=0.2)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert not socket_path.exists()


def test_missing_server_does_not_fall_back_to_one_shot(tmp_path: Path):
    with pytest.raises(ServerConnectionError):
        VPMDKClient(str(tmp_path / "missing.sock")).run(str(tmp_path))
    assert (
        vpmdk.main(
            ["run", "--socket", str(tmp_path / "missing.sock"), "--dir", str(tmp_path)]
        )
        == 3
    )


def test_client_timeout_does_not_cancel_server_job(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    finished = threading.Event()
    accepted = threading.Event()

    def executor(workdir: str, *, calculator) -> None:
        time.sleep(0.4)
        finished.set()
        print("Calculation completed.")

    def observe(event) -> None:
        if event.get("event") == "accepted":
            accepted.set()

    _, thread = _start_server(socket_path, executor=executor)
    client = VPMDKClient(str(socket_path))
    try:
        with pytest.raises(ClientTimeoutError):
            client.run(str(tmp_path), timeout=0.15, event_callback=observe)
        assert accepted.is_set()
        assert finished.wait(2.0)
        _wait_for(lambda: client.status()["jobs_completed"] == 1)
    finally:
        _stop_server(socket_path, thread)


class _DeadlineSocket:
    def __init__(self, *, timeout_during: str):
        self.timeout_during = timeout_during
        self.timeouts: list[float | None] = []
        self.closed = False

    def settimeout(self, value) -> None:
        self.timeouts.append(value)

    def connect(self, path) -> None:
        if self.timeout_during == "connect":
            raise socket.timeout("deliberate connect timeout")

    def sendall(self, payload) -> None:
        if self.timeout_during == "send":
            raise socket.timeout("deliberate send timeout")

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("timeout_during", ["connect", "send"])
def test_request_deadline_applies_to_connect_and_send(
    tmp_path: Path,
    monkeypatch,
    timeout_during: str,
):
    fake_socket = _DeadlineSocket(timeout_during=timeout_during)
    monkeypatch.setattr(client_module.socket, "socket", lambda *args: fake_socket)
    client = VPMDKClient(str(tmp_path / "server.sock"), connect_timeout=2.0)

    with pytest.raises(ClientTimeoutError, match="timed out"):
        client.run(str(tmp_path), timeout=0.05)

    assert fake_socket.closed
    assert fake_socket.timeouts
    assert 0 < float(fake_socket.timeouts[0]) <= 0.05
    if timeout_during == "send":
        assert 0 < float(fake_socket.timeouts[-1]) <= 0.05


def test_connect_deadline_timeout_uses_cli_exit_code_four(
    tmp_path: Path,
    monkeypatch,
):
    fake_socket = _DeadlineSocket(timeout_during="connect")
    monkeypatch.setattr(client_module.socket, "socket", lambda *args: fake_socket)

    assert (
        vpmdk.main(
            [
                "run",
                "--socket",
                str(tmp_path / "server.sock"),
                "--dir",
                str(tmp_path),
                "--timeout",
                "0.05",
            ]
        )
        == 4
    )


def test_cli_exit_codes_for_remote_failure_mismatch_and_timeout(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    failing = tmp_path / "failing"
    mismatch = tmp_path / "mismatch"
    slow = tmp_path / "slow"
    for path in (failing, mismatch, slow):
        path.mkdir()
    (mismatch / "BCAR").write_text("MLP=MACE\n")

    def executor(workdir: str, *, calculator) -> None:
        name = Path(workdir).name
        if name == "failing":
            raise RuntimeError("failed through CLI")
        if name == "slow":
            time.sleep(0.2)
        print("Calculation completed.")

    _, thread = _start_server(socket_path, executor=executor)
    try:
        assert vpmdk.main(["run", "--socket", str(socket_path), "--dir", str(failing)]) == 2
        assert vpmdk.main(["run", "--socket", str(socket_path), "--dir", str(mismatch)]) == 5
        assert (
            vpmdk.main(
                [
                    "run",
                    "--socket",
                    str(socket_path),
                    "--dir",
                    str(slow),
                    "--timeout",
                    "0.05",
                ]
            )
            == 4
        )
    finally:
        _wait_for(lambda: VPMDKClient(str(socket_path)).status()["state"] == "idle")
        _stop_server(socket_path, thread)


def test_serve_cli_builds_model_once_and_supports_json_status(
    tmp_path: Path,
    prepare_inputs,
    monkeypatch,
    capsys,
):
    workdir = tmp_path / "work"
    workdir.mkdir()
    prepare_inputs(workdir, incar_overrides={"NSW": "0"})
    socket_path = tmp_path / "cli.sock"
    builds = 0

    def build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return DummyCalculator()

    monkeypatch.setattr(vpmdk, "_build_calculator_from_tags", build)
    results: list[int] = []
    serve_thread = threading.Thread(
        target=lambda: results.append(
            vpmdk.main(
                [
                    "serve",
                    "--dir",
                    str(workdir),
                    "--socket",
                    str(socket_path),
                ]
            )
        ),
        daemon=True,
    )
    serve_thread.start()
    _wait_for(socket_path.exists)
    def is_idle() -> bool:
        try:
            return VPMDKClient(str(socket_path)).status(timeout=0.2)["state"] == "idle"
        except (ServerConnectionError, ClientTimeoutError):
            return False

    _wait_for(is_idle)

    assert vpmdk.main(["run", "--socket", str(socket_path), "--dir", str(workdir)]) == 0
    assert vpmdk.main(["status", "--socket", str(socket_path), "--json"]) == 0
    assert vpmdk.main(["stop", "--socket", str(socket_path), "--timeout", "3"]) == 0
    serve_thread.join(timeout=3.0)

    output = capsys.readouterr().out
    assert '"jobs_completed": 1' in output
    assert "Calculation completed." in output
    assert builds == 1
    assert results == [0]


def test_graceful_stop_waits_for_running_job(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    started = threading.Event()

    def executor(workdir: str, *, calculator) -> None:
        started.set()
        time.sleep(0.2)
        print("Calculation completed.")

    _, server_thread = _start_server(socket_path, executor=executor)
    run_error: list[BaseException] = []

    def submit() -> None:
        try:
            VPMDKClient(str(socket_path)).run(str(tmp_path))
        except BaseException as exc:  # pragma: no cover - assertion reports details
            run_error.append(exc)

    run_thread = threading.Thread(target=submit)
    run_thread.start()
    assert started.wait(1.0)
    began_stop = time.monotonic()
    VPMDKClient(str(socket_path)).stop(timeout=2.0)
    stop_elapsed = time.monotonic() - began_stop
    run_thread.join(timeout=2.0)
    server_thread.join(timeout=2.0)

    assert run_error == []
    assert stop_elapsed >= 0.15
    assert not socket_path.exists()


def test_second_shutdown_signal_stops_worker_before_queued_job(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    active_workdir = tmp_path / "active"
    queued_workdir = tmp_path / "queued"
    active_workdir.mkdir()
    queued_workdir.mkdir()
    active_started = threading.Event()
    release_active = threading.Event()
    queued_accepted = threading.Event()
    executed: list[str] = []

    def executor(workdir: str, *, calculator) -> None:
        executed.append(Path(workdir).name)
        if Path(workdir) == active_workdir:
            active_started.set()
            assert release_active.wait(3.0)
        print("Calculation completed.")

    server, server_thread = _start_server(socket_path, executor=executor)
    active_errors: list[BaseException] = []
    queued_errors: list[BaseException] = []

    def submit_active() -> None:
        try:
            VPMDKClient(str(socket_path)).run(str(active_workdir))
        except BaseException as exc:  # pragma: no cover - assertions report details
            active_errors.append(exc)

    def submit_queued() -> None:
        try:
            VPMDKClient(str(socket_path)).run(
                str(queued_workdir),
                event_callback=lambda event: (
                    queued_accepted.set()
                    if event.get("event") == "accepted"
                    else None
                ),
            )
        except BaseException as exc:  # pragma: no cover - assertions report details
            queued_errors.append(exc)

    active_thread = threading.Thread(target=submit_active)
    queued_thread = threading.Thread(target=submit_queued)
    active_thread.start()
    assert active_started.wait(1.0)
    queued_thread.start()
    assert queued_accepted.wait(1.0)

    server.install_signal_handlers()
    try:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        # The handler only sets a plain flag (async-signal-safe); the running
        # accept loop drains it into _stop_requested on its 0.2s cadence.
        _wait_for(server._stop_requested.is_set)
        assert not server._worker_stop.is_set()

        handler(signal.SIGINT, None)
        # Second signal escalates to force once the accept loop drains it.
        _wait_for(server._force_requested.is_set)
        _wait_for(server._worker_stop.is_set)
    finally:
        release_active.set()
        active_thread.join(timeout=3.0)
        queued_thread.join(timeout=3.0)
        server_thread.join(timeout=3.0)
        server.restore_signal_handlers()

    assert executed == ["active"]
    assert all(isinstance(error, ServerConnectionError) for error in active_errors)
    assert queued_errors and isinstance(queued_errors[0], ServerConnectionError)
    assert not active_thread.is_alive()
    assert not queued_thread.is_alive()
    assert not server_thread.is_alive()
    assert not socket_path.exists()


def test_transient_accept_error_does_not_kill_the_server(tmp_path: Path):
    # A recoverable accept() error (EMFILE/ENFILE: fd exhaustion) must be logged
    # and retried, not re-raised: re-raising unwinds serve_forever and destroys
    # the VRAM-resident model + queued jobs over a condition that clears as fds
    # free up. Inject one EMFILE, then confirm the server still serves requests.
    assert errno.EMFILE in server_module._TRANSIENT_ACCEPT_ERRNOS

    class _FlakyAcceptListener:
        def __init__(self, real):
            self._real = real
            self._raised = False

        def accept(self):
            if not self._raised:
                self._raised = True
                raise OSError(errno.EMFILE, "Too many open files")
            return self._real.accept()

        def __getattr__(self, name):
            return getattr(self._real, name)

    socket_path = tmp_path / "server.sock"
    server, thread = _start_server(socket_path)
    try:
        server._listener = _FlakyAcceptListener(server._listener)
        # The server must survive the injected EMFILE and answer a later request.
        result = VPMDKClient(str(socket_path)).status(timeout=5.0)
        assert result["protocol"] == 1
    finally:
        _stop_server(socket_path, thread)


def test_status_is_not_blocked_by_a_slow_earlier_connection(tmp_path: Path):
    # SERVER_MODE_SPEC 3.2: status/stop respond immediately. A peer that connects
    # first and then stalls mid-send must not delay a later status behind its
    # accept-order turn (which only run enqueues need) for the whole
    # REQUEST_READ_TIMEOUT window.
    socket_path = tmp_path / "server.sock"
    _, thread = _start_server(socket_path)
    staller = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        staller.connect(str(socket_path))
        # Give the staller's handler time to be accepted (earlier sequence) and
        # block in _read_request without sending a request line.
        time.sleep(0.3)
        start = time.monotonic()
        result = VPMDKClient(str(socket_path)).status(timeout=float(server_module.REQUEST_READ_TIMEOUT))
        elapsed = time.monotonic() - start
        assert result["protocol"] == 1
        assert elapsed < 2.0, (
            f"status blocked {elapsed:.2f}s behind a slow connection "
            f"(REQUEST_READ_TIMEOUT={server_module.REQUEST_READ_TIMEOUT}s)"
        )
    finally:
        staller.close()
        _stop_server(socket_path, thread)


def test_shutdown_signal_handler_is_async_signal_safe(tmp_path: Path):
    # The SIGINT/SIGTERM handler must be async-signal-safe: set plain flags ONLY,
    # taking no lock and calling no Event.set(). Two deadlocks are guarded:
    #  (R67) acquiring _enqueue_lock would invert the enqueue->state order against
    #        the worker (AB-BA);
    #  (R74) calling Event.set() would re-enter the Event's non-reentrant internal
    #        lock if a signal interrupts the main thread mid-set() on the same
    #        Event during shutdown, self-deadlocking the main thread.
    # The main thread translates the flags into the shutdown Events in
    # _should_exit.
    server = VPMDKServer(
        str(tmp_path / "s.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
    )
    server.install_signal_handlers()
    try:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)

        # Hold _enqueue_lock so the handler would block if it tried to take it.
        lock_held = threading.Event()
        release = threading.Event()

        def hold_enqueue_lock() -> None:
            with server._enqueue_lock:
                lock_held.set()
                release.wait(3.0)

        holder = threading.Thread(target=hold_enqueue_lock, daemon=True)
        holder.start()
        assert lock_held.wait(1.0)

        done = threading.Event()

        def fire_two_signals() -> None:
            handler(signal.SIGINT, None)  # first signal
            handler(signal.SIGINT, None)  # second signal -> force escalation
            done.set()

        firer = threading.Thread(target=fire_two_signals, daemon=True)
        firer.start()
        # If the handler acquired _enqueue_lock (held by ``holder``) or otherwise
        # blocked, this would time out.
        assert done.wait(2.0), "signal handler blocked (acquired a lock)"

        # The handler set only plain flags: NO shutdown Event is set yet. If it
        # (re)called Event.set()/_publish_force_stop(), these would already be set
        # -- the exact R74 reentrancy hazard.
        assert server._stop_signal is True
        assert server._force_signal is True
        assert not server._stop_requested.is_set()
        assert not server._force_requested.is_set()
        assert not server._worker_stop.is_set()

        release.set()
        holder.join(timeout=3.0)
        firer.join(timeout=3.0)

        # The main thread drains the flags into the shutdown Events in
        # _should_exit (evaluated by the accept loop in a live server).
        assert server._should_exit() is True
        assert server._stop_requested.is_set()
        assert server._force_requested.is_set()
        assert server._worker_stop.is_set()
    finally:
        server.restore_signal_handlers()
        server._cleanup()


def test_worker_rechecks_force_after_dequeuing_job(tmp_path: Path, monkeypatch):
    executed: list[str] = []
    events: list[dict[str, object]] = []

    class RecordingSender:
        closed = False

        def send(self, event) -> None:
            events.append(dict(event))

        def close(self) -> None:
            self.closed = True

    sender = RecordingSender()
    server = VPMDKServer(
        str(tmp_path / "server.sock"),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        executor=lambda workdir, *, calculator: executed.append(workdir),
    )
    server._queue.put(
        server_module._RunJob(
            workdir=str(tmp_path / "queued"),
            caller_cwd=str(tmp_path),
            sender=sender,
            enqueued_at=time.monotonic(),
        )
    )
    original_get_nowait = server._queue.get_nowait

    def dequeue_while_force_is_published():
        job = original_get_nowait()
        server._publish_force_stop()
        return job

    monkeypatch.setattr(server._queue, "get_nowait", dequeue_while_force_is_published)
    try:
        server._worker_loop()
    finally:
        server._cleanup()

    assert executed == []
    assert sender.closed is True
    assert events[-1]["code"] == "server_stopping"
    assert server._queue.unfinished_tasks == 0
    assert server._busy is False


def test_stop_acknowledgement_precedes_observable_shutdown(
    tmp_path: Path,
    monkeypatch,
):
    socket_path = tmp_path / "server.sock"
    server, server_thread = _start_server(
        socket_path,
        executor=lambda *args, **kwargs: None,
    )
    acknowledgement_started = threading.Event()
    release_acknowledgement = threading.Event()
    original_send = server_module._EventSender.send

    def delay_stop_acknowledgement(sender, event):
        if (
            event.get("event") == "done"
            and event.get("ok") is True
            and "force" in event
        ):
            acknowledgement_started.set()
            assert release_acknowledgement.wait(2.0)
        return original_send(sender, event)

    monkeypatch.setattr(
        server_module._EventSender,
        "send",
        delay_stop_acknowledgement,
    )
    responses: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def stop() -> None:
        try:
            responses.append(VPMDKClient(str(socket_path)).stop(timeout=2.0))
        except BaseException as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    stop_thread = threading.Thread(target=stop)
    stop_thread.start()
    try:
        assert acknowledgement_started.wait(1.0)
        assert not server._stop_requested.is_set()
        time.sleep(0.25)
        assert server_thread.is_alive()
        assert socket_path.exists()
    finally:
        release_acknowledgement.set()
        stop_thread.join(timeout=3.0)
        server_thread.join(timeout=3.0)

    assert errors == []
    assert responses and responses[0]["ok"] is True
    assert not stop_thread.is_alive()
    assert not server_thread.is_alive()
    assert not socket_path.exists()


def test_graceful_stop_cannot_observe_a_dequeued_job_as_idle(
    tmp_path: Path,
    monkeypatch,
):
    socket_path = tmp_path / "server.sock"
    dequeued = threading.Event()
    release_claim = threading.Event()
    execution_started = threading.Event()
    release_execution = threading.Event()

    def executor(workdir: str, *, calculator) -> None:
        execution_started.set()
        assert release_execution.wait(2.0)
        print("Calculation completed.")

    server, server_thread = _start_server(socket_path, executor=executor)
    original_get = server._queue.get
    original_get_nowait = server._queue.get_nowait
    paused = False
    pause_lock = threading.Lock()

    def pause_after_dequeue(job):
        nonlocal paused
        with pause_lock:
            should_pause = not paused
            paused = True
        if should_pause:
            dequeued.set()
            assert release_claim.wait(2.0)
        return job

    def get(*args, **kwargs):
        return pause_after_dequeue(original_get(*args, **kwargs))

    def get_nowait():
        return pause_after_dequeue(original_get_nowait())

    monkeypatch.setattr(server._queue, "get", get)
    monkeypatch.setattr(server._queue, "get_nowait", get_nowait)

    run_errors: list[BaseException] = []

    def submit() -> None:
        try:
            VPMDKClient(str(socket_path)).run(str(tmp_path))
        except BaseException as exc:  # pragma: no cover - assertion reports details
            run_errors.append(exc)

    run_thread = threading.Thread(target=submit)
    stop_returned = threading.Event()
    def stop() -> None:
        server.request_stop()
        stop_returned.set()

    stop_thread = threading.Thread(target=stop)

    try:
        run_thread.start()
        assert dequeued.wait(1.0)
        stop_thread.start()
        assert not stop_returned.wait(0.1)

        release_claim.set()
        assert stop_returned.wait(1.0)
        assert execution_started.wait(1.0)
        assert server._should_exit() is False
    finally:
        release_claim.set()
        release_execution.set()
        run_thread.join(timeout=2.0)
        stop_thread.join(timeout=2.0)
        server_thread.join(timeout=3.0)

    assert run_errors == []
    assert not server_thread.is_alive()
    assert not socket_path.exists()


def test_cli_zero_timeout_stop_returns_after_acknowledgement(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    started = threading.Event()
    release = threading.Event()

    def executor(workdir: str, *, calculator) -> None:
        started.set()
        assert release.wait(2.0)
        print("Calculation completed.")

    _, server_thread = _start_server(socket_path, executor=executor)
    run_errors: list[BaseException] = []

    def submit() -> None:
        try:
            VPMDKClient(str(socket_path)).run(str(tmp_path))
        except BaseException as exc:  # pragma: no cover - assertion reports details
            run_errors.append(exc)

    run_thread = threading.Thread(target=submit)
    run_thread.start()
    assert started.wait(1.0)
    exit_code = vpmdk.main(
        ["stop", "--socket", str(socket_path), "--timeout", "0"]
    )

    assert exit_code == 0
    assert socket_path.exists()

    release.set()
    run_thread.join(timeout=2.0)
    server_thread.join(timeout=2.0)
    assert run_errors == []
    assert not server_thread.is_alive()
    assert not socket_path.exists()


def test_force_stop_acknowledges_and_removes_socket(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
    response = VPMDKClient(str(socket_path)).stop(force=True, timeout=2.0)
    thread.join(timeout=2.0)

    assert response["force"] is True
    assert not thread.is_alive()
    assert not socket_path.exists()


def test_cli_zero_timeout_stop_reports_requested_not_stopped(
    tmp_path: Path, capsys
):
    # timeout=0 returns after acknowledgement without waiting for shutdown, so
    # the CLI must not claim the server has already stopped.
    from vpmdk_client import client_main

    socket_path = tmp_path / "server.sock"
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
    try:
        exit_code = client_main(
            ["stop", "--socket", str(socket_path), "--timeout", "0"]
        )
    finally:
        thread.join(timeout=2.0)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "stop requested" in out
    assert "server stopped" not in out.lower()


def test_cli_positive_timeout_stop_reports_stopped(tmp_path: Path, capsys):
    from vpmdk_client import client_main

    socket_path = tmp_path / "server.sock"
    _, thread = _start_server(socket_path, executor=lambda *args, **kwargs: None)
    try:
        exit_code = client_main(
            ["stop", "--socket", str(socket_path), "--timeout", "5"]
        )
    finally:
        thread.join(timeout=2.0)

    assert exit_code == 0
    assert "server stopped" in capsys.readouterr().out.lower()


def test_cleanup_removes_pidfile_before_unlinking_socket(
    tmp_path: Path, monkeypatch
):
    # A positive-timeout client treats socket disappearance as shutdown
    # completion, so the owned pidfile must be removed first; otherwise a restart
    # racing socket removal could see the stale pidfile and abort.
    socket_path = tmp_path / "server.sock"
    pidfile = tmp_path / "server.pid"
    server = VPMDKServer(
        str(socket_path),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        pidfile=str(pidfile),
    )
    socket_path.write_text("")
    server._socket_inode = os.stat(socket_path).st_ino
    pidfile.write_text("placeholder")
    server._pidfile_written = True

    order: list[str] = []
    monkeypatch.setattr(
        server_module,
        "_remove_owned_pidfile",
        lambda pf, sp, pid: order.append("pidfile"),
    )
    real_unlink = os.unlink

    def spy_unlink(path):
        if str(path) == str(socket_path):
            order.append("socket")
        return real_unlink(path)

    monkeypatch.setattr(os, "unlink", spy_unlink)

    server._cleanup()

    assert order == ["pidfile", "socket"]


def test_remove_owned_pidfile_tolerates_non_utf8_content(tmp_path: Path):
    # A corrupt/tampered pidfile with invalid UTF-8 must not raise from the
    # best-effort removal helper; the unverifiable pidfile is left in place.
    pidfile = tmp_path / "server.pid"
    pidfile.write_bytes(b"\xff\xfe not utf-8\n")

    server_module._remove_owned_pidfile(
        str(pidfile), str(tmp_path / "s.sock"), os.getpid()
    )

    assert pidfile.exists()


def test_cleanup_unlinks_socket_even_if_pidfile_is_corrupt(tmp_path: Path):
    # Now that the pidfile is removed before the socket, a decode error from a
    # corrupt pidfile must not abort cleanup before the socket is unlinked.
    socket_path = tmp_path / "server.sock"
    pidfile = tmp_path / "server.pid"
    server = VPMDKServer(
        str(socket_path),
        DummyCalculator(),
        {"MLP": "CHGNET", "DEVICE": "cpu"},
        backend_base_dir=str(tmp_path),
        pidfile=str(pidfile),
    )
    socket_path.write_text("")
    server._socket_inode = os.stat(socket_path).st_ino
    pidfile.write_bytes(b"\xff\xfe corrupt")
    server._pidfile_written = True

    server._cleanup()  # must not raise

    assert not socket_path.exists()


def test_force_stop_waits_for_active_executor_before_teardown(tmp_path: Path):
    socket_path = tmp_path / "server.sock"
    queued_workdir = tmp_path / "queued"
    queued_workdir.mkdir()
    execution_started = threading.Event()
    release_execution = threading.Event()
    execution_finished = threading.Event()
    queued_accepted = threading.Event()

    def executor(workdir: str, *, calculator) -> None:
        execution_started.set()
        assert release_execution.wait(3.0)
        execution_finished.set()
        print("Calculation completed.")

    _, server_thread = _start_server(socket_path, executor=executor)
    run_errors: list[BaseException] = []
    queued_errors: list[BaseException] = []

    def submit() -> None:
        try:
            VPMDKClient(str(socket_path)).run(str(tmp_path))
        except BaseException as exc:  # pragma: no cover - assertion reports details
            run_errors.append(exc)

    run_thread = threading.Thread(target=submit)
    run_thread.start()
    assert execution_started.wait(1.0)

    def observe_queued(event) -> None:
        if event.get("event") == "accepted":
            queued_accepted.set()

    def submit_queued() -> None:
        try:
            VPMDKClient(str(socket_path)).run(
                str(queued_workdir),
                event_callback=observe_queued,
            )
        except BaseException as exc:  # pragma: no cover - assertion reports details
            queued_errors.append(exc)

    queued_thread = threading.Thread(target=submit_queued)
    queued_thread.start()
    assert queued_accepted.wait(1.0)

    response = VPMDKClient(str(socket_path)).stop(force=True, timeout=0)
    assert response["force"] is True
    _wait_for(lambda: bool(run_errors))
    _wait_for(lambda: bool(queued_errors))

    # Force disconnects clients promptly, but embedded teardown and socket
    # removal must remain pending while the executor still owns the calculator.
    assert not execution_finished.is_set()
    assert server_thread.is_alive()
    assert socket_path.exists()

    release_execution.set()
    run_thread.join(timeout=2.0)
    queued_thread.join(timeout=2.0)
    server_thread.join(timeout=3.0)

    assert isinstance(run_errors[0], ServerConnectionError)
    assert isinstance(queued_errors[0], ServerConnectionError)
    assert execution_finished.is_set()
    assert not run_thread.is_alive()
    assert not queued_thread.is_alive()
    assert not server_thread.is_alive()
    assert not socket_path.exists()


def test_resident_calculator_is_reset_for_every_request(
    tmp_path: Path,
    prepare_inputs,
):
    class ResetCountingCalculator(DummyCalculator):
        def __init__(self):
            self.reset_count = 0
            super().__init__()

        def reset(self):
            self.reset_count += 1
            super().reset()

    socket_path = tmp_path / "server.sock"
    workdir = tmp_path / "work"
    workdir.mkdir()
    prepare_inputs(workdir, incar_overrides={"NSW": "0"})
    calculator = ResetCountingCalculator()
    initial_resets = calculator.reset_count
    _, thread = _start_server(
        socket_path,
        calculator=calculator,
    )
    try:
        client = VPMDKClient(str(socket_path))
        for _ in range(3):
            client.run(str(workdir))
    finally:
        _stop_server(socket_path, thread)
    assert calculator.reset_count == initial_resets + 3


def test_serialize_event_emits_rfc_valid_json_for_non_finite_values():
    # R125: json.dumps' default allow_nan=True writes the bare tokens NaN/Infinity,
    # which are NOT valid JSON (RFC 8259). CPython's json.loads happens to accept
    # them, so the bundled client survived -- but the protocol is specified as
    # NDJSON, so any strict or non-Python peer (jq, a Go/Rust wrapper, --json piped
    # into a validator) cannot parse the frame at all. A resident legitimately holds
    # such values, e.g. NEQUIX_CAPACITY_MULTIPLIER=inf, which status() echoes back.
    #
    # Self-audit follow-up: the sanitizer must also cover mapping KEYS, because
    # json.dumps coerces a non-string key through the same float writer, so a
    # non-finite KEY raises under allow_nan=False even when every value is clean.
    import numpy

    def reject_bare_constant(token: str) -> object:
        raise AssertionError(f"non-JSON token in payload: {token}")

    events = (
        {"event": "result", "energy": float("nan"), "forces": [[float("inf"), 1.0]]},
        {"event": "status", "configuration": {"NEQUIX_CAPACITY_MULTIPLIER": float("inf")}},
        {"event": "log", "histogram": {float("nan"): 1.0}},
        # np.float64 is a float subclass, so it must be sanitized like a float.
        {"event": "result", "stress": numpy.float64("-inf")},
    )

    decoded = []
    for event in events:
        payload = server_module._serialize_event(event)
        decoded.append(
            json.loads(payload.decode("utf-8"), parse_constant=reject_bare_constant)
        )

    assert decoded[0]["energy"] == "nan"
    assert decoded[0]["forces"][0] == ["inf", 1.0]
    assert decoded[1]["configuration"]["NEQUIX_CAPACITY_MULTIPLIER"] == "inf"
    assert list(decoded[2]["histogram"]) == ["nan"]
    assert decoded[3]["stress"] == "-inf"


def test_restore_signal_handlers_tolerates_a_non_python_handler(
    monkeypatch: pytest.MonkeyPatch,
):
    # R125: signal.getsignal() returns None when the previous handler was installed
    # outside Python (a C extension calling sigaction -- an MPI runtime, a JNI
    # library). Replaying that None raises TypeError, which escaped serve_cli's
    # finally: a clean shutdown was reported as exit 1 and the remaining signals
    # stayed bound to the dead server's handler.
    server = server_module.VPMDKServer.__new__(server_module.VPMDKServer)
    server._previous_signal_handlers = {
        signal.SIGTERM: None,
        signal.SIGHUP: signal.SIG_IGN,
    }
    restored: list[tuple[int, object]] = []

    def fake_signal(number, handler):
        if handler is None:
            raise TypeError("signal handler must be signal.SIG_IGN, ...")
        restored.append((number, handler))
        return signal.SIG_DFL

    monkeypatch.setattr(server_module.signal, "signal", fake_signal)
    server.restore_signal_handlers()

    # The unownable handler falls back to the default instead of aborting the loop,
    # and the signals after it in the map are still restored.
    assert restored == [
        (signal.SIGTERM, signal.SIG_DFL),
        (signal.SIGHUP, signal.SIG_IGN),
    ]
    assert server._previous_signal_handlers == {}


def _idle_probe_server(**overrides) -> server_module.VPMDKServer:
    """A VPMDKServer with only the state _should_exit reads, for idle probing."""

    server = server_module.VPMDKServer.__new__(server_module.VPMDKServer)
    server._state_lock = threading.RLock()
    server._enqueue_lock = threading.RLock()
    server._stop_requested = threading.Event()
    server._force_requested = threading.Event()
    server._worker_stop = threading.Event()
    server._job_available = threading.Event()
    server._queue = queue.Queue()
    server._active_connections = 0
    server._busy = False
    server._stop_signal = False
    server._force_signal = False
    server._current_sender = None
    server._last_activity = time.monotonic() - 3600.0
    server.idle_timeout = 1.0
    server.logger = logging.getLogger("vpmdk-test-idle")
    for key, value in overrides.items():
        setattr(server, key, value)
    return server


def test_server_is_not_idle_while_a_terminal_event_is_still_being_sent():
    # R126: _execute_job publishes _busy=False, _current_workdir=None and
    # _last_activity=now BEFORE delivering the terminal `done` event, so a client
    # that stopped draining left the worker parked in sendall (up to
    # EVENT_SEND_TIMEOUT=900s) while the server already looked idle. --idle-timeout
    # then exited the accept loop and _close_listener() ran, so NOTHING accepted
    # while the process stayed alive holding the resident model: status/stop
    # returned exit 3/4, `stop --force` (the one designed preemption for a blocked
    # send) became unreachable, and the live server looked stale to the next
    # `serve`, which bound over its socket. _current_sender is deliberately kept
    # published for exactly this window, so it is the precise signal.
    sending = _idle_probe_server(_current_sender=object())
    assert sending._should_exit() is False

    # A graceful stop must also wait for the terminal event rather than cutting the
    # client off; only force stop preempts a blocked send (checked before the idle
    # test, so that path is unchanged).
    sending._stop_requested.set()
    assert sending._should_exit() is False
    sending._force_requested.set()
    assert sending._should_exit() is True

    finalized = _idle_probe_server()
    assert finalized._should_exit() is True


def test_abandoned_worker_exits_without_finalizing_the_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # R126: the third shutdown signal deliberately abandons a worker that is still
    # inside native backend code (_await_worker_exit's documented ladder). Returning
    # normally then let CPython finalize the interpreter underneath that daemon
    # thread, C++ std::terminate fired ("terminate called without an active
    # exception") and a textbook-correct teardown -- warning logged, socket
    # unlinked, handlers restored -- died of SIGABRT, reporting 134 where the code
    # intends exit 0. A supervisor reads that as a crash and can restart the server
    # the operator just stopped. Measured on the real tree: pre-fix the foreground
    # process ended with WTERMSIG=6, post-fix with exit 0; the daemon path was
    # already immune because finish() used os._exit for it, so the two entry points
    # disagreed on the same shutdown path.
    launch = tmp_path / "scratch"
    launch.mkdir()
    exits: list[int] = []

    # Record instead of exiting: os._exit never returns in production, and letting
    # the stand-in raise SystemExit would be caught by serve_cli's own
    # `except BaseException` and rewritten as the error path.
    def fake_exit(code):
        exits.append(code)

    monkeypatch.setattr(
        server_module, "_load_backend_for_server",
        lambda workdir, bcar: (DummyCalculator(), {"MLP": "CHGNET"}, str(launch)),
    )
    monkeypatch.setattr(
        server_module.VPMDKServer, "install_signal_handlers", lambda self: None
    )
    monkeypatch.setattr(
        server_module.VPMDKServer, "restore_signal_handlers", lambda self: None
    )
    monkeypatch.setattr(server_module.os, "_exit", fake_exit)

    def make_args(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            socket=str(tmp_path / name),
            dir=str(launch),
            bcar=None,
            daemon=False,
            idle_timeout=0.0,
            log_file=None,
        )

    def serve_forever_abandoning(self, *, ready_callback=None):
        # What _await_worker_exit records on the third signal.
        self._worker_abandoned = True

    monkeypatch.setattr(
        server_module.VPMDKServer, "serve_forever", serve_forever_abandoning
    )
    assert server_module.serve_cli(make_args("abandoned.sock")) == 0
    assert exits == [0], "abandoned worker must skip interpreter finalization"

    # A shutdown that joined its worker still returns normally, so nothing about the
    # ordinary path (atexit handlers, buffered output, embedding callers) changes.
    monkeypatch.setattr(
        server_module.VPMDKServer,
        "serve_forever",
        lambda self, ready_callback=None: None,
    )
    assert server_module.serve_cli(make_args("clean.sock")) == 0
    assert exits == [0]


def test_stale_pidfile_cleanup_spares_a_live_server_for_the_same_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # R128: during a force-stop drain the server closes its listener BEFORE joining
    # the worker (deliberate: otherwise a client sits in recv for the whole job), so
    # a live daemon that still holds the model stops answering. The socket then looks
    # stale, _remove_stale_pidfile deleted the LIVE pidfile -- its docstring premise,
    # "an unresponsive socket means the daemon that wrote it is dead", is false in
    # this window -- and that disarmed _write_pidfile's ServerAlreadyRunning guard, so
    # a second `serve` loaded a SECOND copy of the model beside the first. Measured on
    # the real tree: pre-fix the second `serve` returned 0 and two resident processes
    # were alive at once; post-fix it is refused with "belongs to a live process".
    socket_path = tmp_path / "s.sock"
    pidfile = Path(server_module.pidfile_path(str(socket_path)))
    resolved = os.path.realpath(os.path.abspath(str(socket_path)))
    pidfile.write_text(f"4242\nsocket={resolved}\n")

    # A pid that is merely ALIVE must still be cleaned up: it may be an unrelated
    # process that recycled the number, and blocking restarts on that is the deadlock
    # this helper exists to prevent.
    monkeypatch.setattr(
        server_module,
        "_pid_is_live_server_for_socket",
        lambda pid, path, **kwargs: False,
    )
    server_module._remove_stale_pidfile(str(socket_path))
    assert not pidfile.exists()

    pidfile.write_text(f"4242\nsocket={resolved}\n")
    monkeypatch.setattr(
        server_module,
        "_pid_is_live_server_for_socket",
        lambda pid, path, **kwargs: True,
    )
    server_module._remove_stale_pidfile(str(socket_path))
    assert pidfile.exists(), "a live server's pidfile must survive"

    # And with the pidfile intact, a second serve is refused rather than starting a
    # second resident.
    monkeypatch.setattr(server_module, "_pid_is_alive", lambda pid: True)
    with pytest.raises(server_module.ServerAlreadyRunning):
        server_module._write_pidfile(str(pidfile), str(socket_path))


def test_live_server_probe_requires_a_serve_process_for_this_socket(tmp_path: Path):
    # The probe must be narrower than "the pid is alive": our own pid is alive but is
    # not a `serve` for this socket, and a dead pid must never match.
    socket_path = tmp_path / "s.sock"

    assert not server_module._pid_is_live_server_for_socket(os.getpid(), str(socket_path))
    assert not server_module._pid_is_live_server_for_socket(-1, str(socket_path))
    assert not server_module._pid_is_live_server_for_socket(2**31 - 1, str(socket_path))


def test_write_pidfile_rejects_a_fifo_with_an_actionable_message(tmp_path: Path):
    # R128: os.fdopen(fd, "r+") raises io.UnsupportedOperation("File or stream is not
    # seekable") for a FIFO BEFORE the S_ISREG check could run, so `serve --daemon`
    # died after the full model load with a message naming neither the pidfile nor
    # the cause -- while the correct message sat right below, unreachable.
    socket_path = tmp_path / "s.sock"
    pidfile = Path(server_module.pidfile_path(str(socket_path)))
    os.mkfifo(pidfile)

    with pytest.raises(RuntimeError, match="non-regular pidfile"):
        server_module._write_pidfile(str(pidfile), str(socket_path))


class _ForcelessCalculator(DummyCalculator):
    """What a documented energy-only backend does: results['forces'] is None."""

    def calculate(self, atoms=None, properties=("energy",), system_changes=()):
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        self.results["forces"] = None
        self.results["stress"] = None


class _StresslessCalculator(DummyCalculator):
    """Force-capable, stress-less: what MATRIS_TASK=ef provides."""

    def calculate(self, atoms=None, properties=("energy",), system_changes=()):
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        self.results["stress"] = None


def _capability_request_dir(root: Path, name: str, bcar: str | None, incar: str) -> str:
    directory = root / name
    directory.mkdir()
    (directory / "POSCAR").write_text(
        "Si2\n1.0\n2.715 2.715 0.0\n0.0 2.715 2.715\n2.715 0.0 2.715\n"
        "Si\n2\nDirect\n0.0 0.0 0.0\n0.25 0.25 0.25\n"
    )
    (directory / "INCAR").write_text(incar)
    if bcar is not None:
        (directory / "BCAR").write_text(bcar)
    return str(directory)


@pytest.mark.parametrize(
    "bcar",
    [
        "MLP = MATRIS\nMATRIS_TASK = e\nDEVICE = cpu\n",  # restates the resident
        "MLP = MATRIS\n",  # inherits MATRIS_TASK
        "MLP = MATRIS\nMATRIS_TASK =\n",  # present but blank == omitted
        "",  # empty BCAR file
        None,  # no BCAR at all -- the documented batch pattern
    ],
)
def test_energy_only_resident_is_input_error_however_the_request_spells_it(
    tmp_path: Path, bcar
):
    # R132: R131's capability gate resolved its BackendConfig from the REQUEST
    # BCAR alone, and BackendConfig.from_mapping({}) defaults to CHGNET. So a
    # request that RESTATED the resident's tags got exit 1 before any compute,
    # while the byte-identical request that INHERITED them (SPEC 3.4, and what
    # examples/server_batch/calculations/0001/BCAR documents as the intended
    # form) ran the full inference, wrote a partial OUTCAR/OSZICAR and failed as
    # calculation_error -> exit 2, which SPEC 2.5 documents as RETRYABLE: a
    # retry driver would resubmit a permanently broken configuration forever.
    socket_path = tmp_path / "energy-only.sock"
    workdir = _capability_request_dir(
        tmp_path, f"req{abs(hash(str(bcar))) % 997}", bcar, "IBRION = -1\nNSW = 0\n"
    )
    _, thread = _start_server(
        socket_path,
        calculator=_ForcelessCalculator(),
        tags={"MLP": "MATRIS", "MATRIS_TASK": "e", "DEVICE": "cpu"},
    )
    try:
        with pytest.raises(RemoteInputError, match="energy only"):
            VPMDKClient(str(socket_path)).run(workdir, timeout=60)
    finally:
        _stop_server(socket_path, thread)

    # Nothing was computed, so no output file was written either.
    assert not os.path.exists(os.path.join(workdir, "OUTCAR"))
    assert not os.path.exists(os.path.join(workdir, "OSZICAR"))


@pytest.mark.parametrize(
    "bcar", ["MLP = MATRIS\nMATRIS_TASK = ef\nDEVICE = cpu\n", "MLP = MATRIS\n", None]
)
def test_missing_stress_warning_does_not_depend_on_restating_resident_tags(
    tmp_path: Path, bcar
):
    # Same root cause, the §1.2 half: the warning R131 added so an omitted stress
    # block would not be silent was emitted only when the request happened to
    # spell MATRIS_TASK out, so two ways of expressing the same run produced
    # different output.
    socket_path = tmp_path / "stressless.sock"
    workdir = _capability_request_dir(
        tmp_path,
        f"req{abs(hash(str(bcar))) % 997}",
        bcar,
        "IBRION = -1\nNSW = 0\nISIF = 2\n",
    )
    logs: list[str] = []
    _, thread = _start_server(
        socket_path,
        calculator=_StresslessCalculator(),
        tags={"MLP": "MATRIS", "MATRIS_TASK": "ef", "DEVICE": "cpu"},
    )
    try:
        VPMDKClient(str(socket_path)).run(workdir, timeout=60, log_callback=logs.append)
    finally:
        _stop_server(socket_path, thread)

    assert any("does not provide stress" in line for line in logs), logs
    assert any("Calculation completed." in line for line in logs), logs


def test_resident_backend_tags_are_only_a_capability_fallback(tmp_path: Path):
    # The resident's tags must not override what the request DID spell out, and a
    # fully-capable resident must not start warning about anything.
    from vpmdk_core.cli import _capability_backend_tags

    resident = {"MLP": "MATRIS", "MATRIS_TASK": "e", "DEVICE": "cpu"}
    assert _capability_backend_tags({}, resident) == resident
    assert _capability_backend_tags({"MATRIS_TASK": "efs"}, resident)["MATRIS_TASK"] == "efs"
    # A present-but-blank tag is an omission for the builder, so the resident wins.
    assert _capability_backend_tags({"MATRIS_TASK": ""}, resident)["MATRIS_TASK"] == "e"
    # One-shot passes no resident tags and must be untouched.
    assert _capability_backend_tags({"MLP": "CHGNET"}, None) == {"MLP": "CHGNET"}

    identity = backend_identity(
        {"MLP": "MATRIS", "MATRIS_TASK": "ef", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    assert server_module._resident_backend_tags(identity)["MATRIS_TASK"] == "ef"
    assert server_module._resident_backend_tags(identity)["MLP"] == "MATRIS"


def test_unwritable_custom_socket_parent_is_rejected_before_the_model_load(tmp_path):
    # R149 (P3): an existing-but-unwritable CUSTOM socket parent passed the
    # whole pre-load gate (makedirs(exist_ok=True) succeeds on an existing
    # directory regardless of write permission, and the ownership probe runs
    # only for the DEFAULT parent), so `vpmdk serve` paid the full model load
    # and then died in listener.bind() with a PermissionError whose filename
    # is None -- a bare "Permission denied" naming no path at all. AF_UNIX
    # bind() unconditionally needs write+search on the directory, so the
    # failure is fully decidable up front.
    if os.geteuid() == 0:
        pytest.skip("directory permissions do not bind as root")

    parent = tmp_path / "sockets"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        with pytest.raises(RuntimeError, match="not writable"):
            server_module.prepare_socket_path(
                str(parent / "server.sock"), pidfile_expected=True
            )
    finally:
        parent.chmod(0o700)

    # Restored write permission passes the gate again.
    server_module.prepare_socket_path(str(parent / "server.sock"), pidfile_expected=True)


def test_calculation_stderr_and_warnings_are_relayed_to_the_client(tmp_path: Path):
    # R151 (P3): only stdout was request-scoped; the calculation's stderr --
    # exactly where third-party physics caveats go (ASE FutureWarning about
    # fixcm NVT sampling, pymatgen BadPoscarWarning, numpy RuntimeWarning) --
    # went to the server's own stderr / private 0600 log, so the submitting
    # client saw nothing while the byte-identical one-shot run printed it.
    # Worse, Python's per-process warning dedup meant a resident emitted each
    # warning for the FIRST job only; catch_warnings() per job restores the
    # one-shot per-run behavior. (warnings.warn itself cannot be asserted
    # under pytest, whose plugin captures warnings before they reach stderr;
    # the warning half is emulated with the default showwarning behavior --
    # a formatted line written to sys.stderr at warn time.)
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    def executor(workdir: str, *, calculator) -> None:
        print("stdout-disclosure")
        print("stderr-disclosure", file=sys.stderr)
        sys.stderr.write(
            "ase/md/langevin.py:102: FutureWarning: thermostat-caveat\n"
        )

    _, thread = _start_server(socket_path, executor=executor)
    try:
        for attempt in (1, 2):
            out_lines: list[str] = []
            err_lines: list[str] = []
            VPMDKClient(str(socket_path)).run(
                str(request_dir),
                log_callback=out_lines.append,
                stderr_log_callback=err_lines.append,
            )
            assert "stdout-disclosure" in out_lines, attempt
            assert "stderr-disclosure" in err_lines, attempt
            assert not any("stderr-disclosure" in line for line in out_lines), attempt
            assert not any("stdout-disclosure" in line for line in err_lines), attempt
            assert any("thermostat-caveat" in line for line in err_lines), attempt
    finally:
        _stop_server(socket_path, thread)


def test_permanent_pathology_oserrors_are_input_errors(tmp_path: Path):
    # R152 (P2): ELOOP (self-referential symlink at an artifact path),
    # ENAMETOOLONG, and FileNotFoundError (dangling symlink into a missing
    # directory) have no place in the R135 subclass tuple or the R150 EROFS
    # test, so they fell to calculation_error / exit 2 -- documented
    # RETRYABLE for conditions that are permanent properties of the
    # submitted workdir, while one-shot exits 1 (lesson xxxix).
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    for planted in (
        OSError(errno.ELOOP, "Too many levels of symbolic links", "OUTCAR"),
        OSError(errno.ENAMETOOLONG, "File name too long", "OUTCAR"),
        FileNotFoundError(2, "No such file or directory", "OUTCAR"),
    ):

        def executor(workdir: str, *, calculator, _exc=planted) -> None:
            raise _exc

        _, thread = _start_server(socket_path, executor=executor)
        try:
            with pytest.raises(RemoteInputError):
                VPMDKClient(str(socket_path)).run(str(request_dir))
        finally:
            _stop_server(socket_path, thread)


def test_relayed_stderr_is_dropped_when_client_stderr_is_closed():
    # R152 (P3): with fd 2 closed (`vpmdk run 2>&-`) CPython sets sys.stderr
    # to None; _write_line treats a None stream as stdout, so the R151
    # stderr relay injected warning text into the stdout stream scripts
    # parse, where one-shot drops those lines entirely (lesson xxxii: a None
    # stream is part of the condition's state space).
    import argparse

    lcm = lightweight_client_module

    def generator(self, request, *, timeout):
        yield {"event": "log", "line": "warning-line", "stream": "stderr"}
        yield {"event": "log", "line": "Calculation completed."}
        yield {"event": "done", "ok": True, "elapsed_s": 1.0}

    stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
    args = argparse.Namespace(command="run", socket="/s", dir=".", timeout=0.0)
    original_request = lcm.VPMDKClient._request
    saved = (sys.stdout, sys.stderr)
    try:
        lcm.VPMDKClient._request = generator
        sys.stdout, sys.stderr = stdout, None
        code = lcm.client_cli(args)
    finally:
        sys.stdout, sys.stderr = saved
        lcm.VPMDKClient._request = original_request
    stdout.flush()
    out = stdout.buffer.getvalue()
    assert code == 0
    assert b"Calculation completed." in out
    assert b"warning-line" not in out


def test_server_jobs_enable_torch_warn_always(tmp_path: Path):
    # R152 (P3): TORCH_WARN_ONCE dedups in C++ below the Python warnings
    # layer, so the R151 per-job catch_warnings() could not re-arm it and
    # torch-originated warnings reached the client on the FIRST job of a
    # resident only. Each job now enables torch's warnAlways, which forwards
    # every occurrence to the Python layer where the per-job filter scope
    # restores exactly once-per-job (measured: no intra-job over-emission).
    torch = pytest.importorskip("torch")
    previous = torch.is_warn_always_enabled()
    torch.set_warn_always(False)

    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    def executor(workdir: str, *, calculator) -> None:
        pass

    _, thread = _start_server(socket_path, executor=executor)
    try:
        VPMDKClient(str(socket_path)).run(str(request_dir))
        assert torch.is_warn_always_enabled()
    finally:
        _stop_server(socket_path, thread)
        torch.set_warn_always(previous)


def test_client_umask_governs_output_artifact_modes(tmp_path: Path, monkeypatch):
    # R152 (P2): output artifacts were created with the SERVER's launch
    # umask; the submitting client's umask was never transmitted, so
    # `umask 077; vpmdk run` silently wrote world-readable outputs from a
    # umask-022 resident and the reverse broke group pipelines with 0600
    # files after exit 0. The request now carries the client's umask and the
    # worker applies it around the calculation (process-global like the cwd
    # the job also swaps), restoring it afterwards.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()

    def executor(workdir: str, *, calculator) -> None:
        with open(os.path.join(workdir, "OUTCAR"), "w") as handle:
            handle.write("x")

    # Simulate a client whose umask differs from the server process's.
    monkeypatch.setattr(lightweight_client_module, "_read_current_umask", lambda: 0o077)
    process_umask = os.umask(0o022)
    os.umask(0o022)

    _, thread = _start_server(socket_path, executor=executor)
    try:
        VPMDKClient(str(socket_path)).run(str(request_dir))
        mode = stat.S_IMODE(os.stat(request_dir / "OUTCAR").st_mode)
        assert mode == 0o600
        # The worker restored the server's own umask after the job.
        restored = os.umask(0o022)
        os.umask(restored)
        assert restored == 0o022
    finally:
        _stop_server(socket_path, thread)
        os.umask(process_umask)


def test_daemon_fifo_log_file_fails_fast_with_a_diagnostic(
    tmp_path: Path, capsys, monkeypatch
):
    # Bound the launcher wait so a regression cannot stall the suite for the
    # default 600 s.
    monkeypatch.setenv("VPMDK_DAEMON_START_TIMEOUT", "20")
    # R154 (P2): a reader-less FIFO at an explicit --log-file made the forked
    # daemon child block forever in a plain os.open -- the launcher waited
    # the full 600 s daemon-start timeout, printed advice that cannot work
    # ('check with vpmdk status' for a socket that will never appear), and
    # leaked a permanently stuck orphan -- while the identical FOREGROUND
    # invocation fails in under a second. O_NONBLOCK turns the condition
    # into an immediate ENXIO reported through the readiness pipe
    # (set_blocking(True) afterwards keeps a FIFO WITH a reader working).
    fifo = tmp_path / "collector.fifo"
    os.mkfifo(fifo)
    args = SimpleNamespace(
        command="serve",
        dir=str(tmp_path),
        bcar=None,
        socket=str(tmp_path / "s.sock"),
        idle_timeout=0.0,
        daemon=True,
        log_file=str(fifo),
        daemon_notify_fd=None,
    )
    started = time.monotonic()
    assert server_module.serve_cli(args) == 1
    assert time.monotonic() - started < 60.0
    err = capsys.readouterr().err
    assert "daemon failed to start" in err


def test_dangling_request_bcar_symlink_is_an_input_error(tmp_path: Path):
    # R155 (P3) wiring, server half: a broken BCAR link in a submitted
    # workdir must be input_error / exit 1, not a silent fallback to the
    # resident's tags as if the request BCAR were absent.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()
    (request_dir / "BCAR").symlink_to(request_dir / "gone" / "BCAR")

    def executor(workdir: str, *, calculator) -> None:
        raise AssertionError("the job must be rejected before execution")

    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteInputError, match="cannot be resolved"):
            VPMDKClient(str(socket_path)).run(str(request_dir))
    finally:
        _stop_server(socket_path, thread)


def test_nequix_compile_tag_is_inert_under_jax_backend(tmp_path: Path):
    # R156 (P3): upstream nequix stores use_compile but reads it ONLY in its
    # torch branch, and NEQUIX_BACKEND defaults to jax -- so a request that
    # spells out NEQUIX_USE_COMPILE against a jax resident was rejected exit
    # 5 for a calculator one-shot builds bit-for-bit identically. Sibling of
    # the DEVICE-under-jax rule in _device_tag_is_inert. The torch half
    # (NEQUIX_CAPACITY_MULTIPLIER inert under torch) is deliberately NOT
    # mirrored: no torch checkpoint exists here to verify it.
    jax_resident = backend_identity(
        {"MLP": "NEQUIX", "DEVICE": "cpu"}, base_dir=str(tmp_path)
    )
    for tag in ("NEQUIX_USE_COMPILE", "NEQUIX_COMPILE"):
        validate_request_backend(
            jax_resident,
            {"MLP": "NEQUIX", tag: "1", "DEVICE": "cpu"},
            request_base_dir=str(tmp_path),
        )

    # Under an explicit torch backend the tag genuinely changes the model and
    # must still be compared.
    torch_resident = backend_identity(
        {"MLP": "NEQUIX", "NEQUIX_BACKEND": "torch", "DEVICE": "cpu"},
        base_dir=str(tmp_path),
    )
    with pytest.raises(BackendConfigurationMismatch, match="NEQUIX_USE_COMPILE"):
        validate_request_backend(
            torch_resident,
            {
                "MLP": "NEQUIX",
                "NEQUIX_BACKEND": "torch",
                "NEQUIX_USE_COMPILE": "1",
                "DEVICE": "cpu",
            },
            request_base_dir=str(tmp_path),
        )


def test_failing_request_bcar_still_streams_the_unused_input_notices(
    tmp_path: Path,
):
    # R157 (P3): the server hoists the request-BCAR parse out of run_workdir
    # (validate_request_backend needs the tags), so when that parse failed
    # the client never received the "Note: KPOINTS detected..." lines the
    # byte-identical one-shot run prints FIRST -- the very line SPEC 3.3
    # uses as its canonical log-event example.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()
    (request_dir / "KPOINTS").write_text("Automatic\n0\nGamma\n1 1 1\n")
    (request_dir / "BCAR").symlink_to(request_dir / "gone" / "BCAR")

    def executor(workdir: str, *, calculator) -> None:
        raise AssertionError("the job must be rejected before execution")

    out_lines: list[str] = []
    _, thread = _start_server(socket_path, executor=executor)
    try:
        with pytest.raises(RemoteInputError, match="cannot be resolved"):
            VPMDKClient(str(socket_path)).run(
                str(request_dir), log_callback=out_lines.append
            )
    finally:
        _stop_server(socket_path, thread)
    assert any("KPOINTS detected" in line for line in out_lines)


def test_request_bcar_typo_warning_matches_oneshot_position(tmp_path: Path):
    # R162 (P2): the server's hoisted request-BCAR parse emitted the R161
    # unknown-tag warning BEFORE run_workdir's Note: lines (one-shot prints
    # it after), and emitted it even for a request whose INCAR fails before
    # one-shot ever reads BCAR -- a SPEC 1.2 stdout divergence. The hoisted
    # parse now suppresses the warning and run_workdir re-emits it for the
    # passed tags at the one-shot position.
    socket_path = tmp_path / "server.sock"
    request_dir = tmp_path / "work"
    request_dir.mkdir()
    (request_dir / "KPOINTS").write_text("Automatic\n0\nGamma\n1 1 1\n")
    (request_dir / "BCAR").write_text("MLP = CHGNET\nMODELL = /nowhere\n")

    def executor(workdir: str, *, calculator) -> None:
        # Mirror one-shot's opening sequence: notices first, then the BCAR
        # warning for the pre-parsed tags (run_workdir does exactly this).
        import vpmdk_core

        vpmdk_core._print_unused_input_notices(workdir)
        vpmdk_core._warn_unknown_bcar_tags({"MLP": "CHGNET", "MODELL": "/nowhere"})

    out_lines: list[str] = []
    _, thread = _start_server(socket_path, executor=executor)
    try:
        VPMDKClient(str(socket_path)).run(
            str(request_dir), log_callback=out_lines.append
        )
    finally:
        _stop_server(socket_path, thread)

    note_index = next(
        i for i, line in enumerate(out_lines) if "KPOINTS detected" in line
    )
    warn_indices = [
        i for i, line in enumerate(out_lines) if "MODELL is not recognized" in line
    ]
    # Exactly ONE warning (the hoisted parse stayed silent), and it comes
    # after the Note: line, as in one-shot.
    assert len(warn_indices) == 1, out_lines
    assert warn_indices[0] > note_index, out_lines


def test_foreground_fifo_log_with_live_reader_does_not_hang(
    tmp_path: Path, monkeypatch
):
    # Cross-review (P2): the R152 probe opened and immediately CLOSED its
    # writer on the log path, which delivered EOF to a FIFO's waiting reader
    # (cat) -- the reader exited, and the post-load FileHandler then
    # reopened the FIFO with no reader left and blocked forever, with no
    # socket and no diagnostic. The probe fd is now held until the server's
    # FileHandler has opened the path.
    fifo = tmp_path / "collector.fifo"
    os.mkfifo(fifo)

    collected: list[bytes] = []

    def reader() -> None:
        with open(fifo, "rb") as handle:
            collected.append(handle.read())

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    def slow_loader(workdir, bcar):
        # Simulate the model load: in the real failure the reader's EOF exit
        # happens DURING this window, so the post-load FileHandler reopens a
        # reader-less FIFO. Without the delay the reopen can win the race
        # against the reader's exit and the regression is invisible.
        time.sleep(1.5)
        return (
            DummyCalculator(),
            {"MLP": "CHGNET", "DEVICE": "cpu"},
            str(tmp_path),
        )

    monkeypatch.setattr(server_module, "_load_backend_for_server", slow_loader)
    monkeypatch.setattr(
        server_module.VPMDKServer, "serve_forever", lambda self, ready_callback=None: None
    )
    monkeypatch.setattr(
        server_module.VPMDKServer, "install_signal_handlers", lambda self: None
    )
    args = SimpleNamespace(
        command="serve",
        dir=str(tmp_path),
        bcar=None,
        socket=str(tmp_path / "s.sock"),
        idle_timeout=0.0,
        daemon=False,
        log_file=str(fifo),
        daemon_notify_fd=None,
    )

    result: dict[str, int] = {}

    def run() -> None:
        result["rc"] = server_module.serve_cli(args)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(30)
    try:
        assert not worker.is_alive(), "foreground serve hung opening the FIFO log"
        assert result.get("rc") == 0
    finally:
        if worker.is_alive():
            # Unblock a hung open so the suite can continue.
            with contextlib.suppress(OSError):
                rescue = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
                os.close(rescue)


def test_grace_ignored_device_request_warning():
    # DEVICE is deliberately inert for GRACE (dropped from the identity
    # comparison) and the resident builder never re-runs per request, so the
    # "GRACE ignores the DEVICE tag" warning the byte-identical one-shot run
    # prints from the builder never reached a request client. The server now
    # synthesizes the builder's exact message for a DEVICE-carrying GRACE
    # request.
    resident = {"mlp": "GRACE", "model": "GRACE-2L-OMAT"}

    warn = server_module._grace_ignored_device_request_warning(
        resident, {"DEVICE": "cuda"}
    )
    assert warn == vpmdk._GRACE_DEVICE_IGNORED_WARNING

    # No DEVICE (or a blank one) stays quiet, matching the one-shot builder.
    assert (
        server_module._grace_ignored_device_request_warning(resident, {}) is None
    )
    assert (
        server_module._grace_ignored_device_request_warning(
            resident, {"DEVICE": "  "}
        )
        is None
    )

    # A non-GRACE resident is out of scope.
    assert (
        server_module._grace_ignored_device_request_warning(
            {"mlp": "CHGNET"}, {"DEVICE": "cuda"}
        )
        is None
    )

    # An explicit MLP override to another backend is left to the mismatch check.
    assert (
        server_module._grace_ignored_device_request_warning(
            resident, {"MLP": "CHGNET", "DEVICE": "cuda"}
        )
        is None
    )

    # An explicit GRACE MLP restatement still warns.
    assert (
        server_module._grace_ignored_device_request_warning(
            resident, {"MLP": "GRACE", "DEVICE": "cuda"}
        )
        == vpmdk._GRACE_DEVICE_IGNORED_WARNING
    )


def test_resolve_backend_device_mirrors_bam_upstream_collapse():
    # R177 (P2): RACECalculator.configure_device maps ONLY the literal 'cpu'
    # to the cpu and every other spelling ('cuda', 'cuda:1', 'gpu', 'CPU',
    # 'cpu:0') to cuda-if-available-else-cpu, dropping the index. Raw-string
    # identity therefore rejected byte-identical request/resident pairs with
    # exit 5 and equated pairs that differ in effect. The resolver now
    # mirrors the builder exactly for BAM.
    import torch

    accelerated = "cuda" if torch.cuda.is_available() else "cpu"

    assert server_module._resolve_backend_device("BAM", "cpu") == "cpu"
    assert server_module._resolve_backend_device("BAM", "") == "cpu"
    for spelling in ("cuda", "cuda:1", "gpu", "CPU", "cpu:0"):
        assert (
            server_module._resolve_backend_device("BAM", spelling) == accelerated
        ), spelling
    # Omitted DEVICE follows _resolve_device autodetection, which lands on
    # the same effective device as any accelerator spelling.
    assert server_module._resolve_backend_device("BAM", None) == accelerated


def test_serve_cli_closes_the_log_probe_fd_when_startup_fails(tmp_path, monkeypatch, capsys):
    # Cross-review (R179 window, P2): the foreground --log-file probe fd is
    # opened BEFORE backend loading and, on the success path, closed right
    # after VPMDKServer construction -- but when _load_backend_for_server or
    # construction raised, only the error tail ran and the fd leaked. A
    # long-lived caller invoking serve_cli repeatedly leaked one fd per
    # failed start (eventually EMFILE).
    def boom(*args, **kwargs):
        raise RuntimeError("backend load boom")

    monkeypatch.setattr(server_module, "_load_backend_for_server", boom)

    log_file = tmp_path / "server.log"
    args = SimpleNamespace(
        command="serve",
        dir=str(tmp_path),
        bcar=None,
        socket=str(tmp_path / "s.sock"),
        idle_timeout=0.0,
        daemon=False,
        log_file=str(log_file),
        daemon_notify_fd=None,
    )

    def open_fds() -> set[int]:
        return {int(name) for name in os.listdir("/proc/self/fd")}

    before = open_fds()
    for _ in range(3):
        assert server_module.serve_cli(args) == 1
    after = open_fds()

    leaked = sorted(after - before)
    assert leaked == [], f"leaked fds after failed starts: {leaked}"
    assert "backend load boom" in capsys.readouterr().err


def test_serve_cli_reserves_the_endpoint_before_the_model_load(tmp_path, monkeypatch, capsys):
    # Cross-review (R181 window, P2): prepare_socket_path is a check, not a
    # reservation, so two concurrent serves targeting the same unused socket
    # both proceeded to load a potentially GPU-sized backend and the loser
    # was only detected at _bind. serve_cli now writes the pidfile
    # (O_CREAT|O_EXCL with live-owner detection) BEFORE the model load and
    # releases it on the startup error tail.
    import vpmdk_client

    socket_path = str(tmp_path / "s.sock")
    pidfile = server_module.pidfile_path(
        vpmdk_client.resolve_socket_path(socket_path)
    )
    observed: dict[str, object] = {}

    def failing_load(*args, **kwargs):
        observed["pidfile_exists_during_load"] = os.path.exists(pidfile)
        if os.path.exists(pidfile):
            observed["recorded_pid"] = server_module._parse_pidfile_metadata(
                open(pidfile, encoding="utf-8").read()
            )[0]
        raise RuntimeError("load boom")

    monkeypatch.setattr(server_module, "_load_backend_for_server", failing_load)
    args = SimpleNamespace(
        command="serve",
        dir=str(tmp_path),
        bcar=None,
        socket=socket_path,
        idle_timeout=0.0,
        daemon=False,
        log_file=None,
        daemon_notify_fd=None,
    )

    assert server_module.serve_cli(args) == 1
    assert "load boom" in capsys.readouterr().err
    # The reservation existed while the backend was loading, recorded THIS
    # process, and was released on the failure tail.
    assert observed["pidfile_exists_during_load"] is True
    assert observed["recorded_pid"] == os.getpid()
    assert not os.path.exists(pidfile)

    # A reservation held by a live owner refuses BEFORE any model load.
    def already(*args, **kwargs):
        raise server_module.ServerAlreadyRunning("held by a live process")

    load_calls: list[int] = []
    monkeypatch.setattr(server_module, "_write_pidfile", already)
    monkeypatch.setattr(
        server_module,
        "_load_backend_for_server",
        lambda *a, **k: load_calls.append(1),
    )
    assert server_module.serve_cli(args) == 1
    assert "held by a live process" in capsys.readouterr().err
    assert load_calls == []


def test_bind_failure_releases_the_preload_reservation(tmp_path, monkeypatch, capsys):
    # R182 (P3, fixed as completion of the cross-review "release it on
    # failure" item): pidfile ownership transfers at the LAST statement of
    # _bind (_pidfile_written = True), but server_ref is populated before
    # serve_forever -- so a _bind failure (AF_UNIX path too long, parent
    # turned unwritable during the model load, EADDRINUSE) skipped BOTH
    # release paths and left the pre-load reservation <socket>.pid on disk.
    # For a long-lived in-process serve_cli caller the leaked record names a
    # LIVE pid, permanently refusing every other serve on that endpoint.
    monkeypatch.setattr(
        server_module,
        "_load_backend_for_server",
        lambda *a, **k: (DummyCalculator(), {}, str(tmp_path)),
    )
    socket_path = tmp_path / ("d" * 120)  # beyond the AF_UNIX path limit
    args = SimpleNamespace(
        command="serve",
        dir=str(tmp_path),
        bcar=None,
        socket=str(socket_path),
        idle_timeout=0.0,
        daemon=False,
        log_file=None,
        daemon_notify_fd=None,
    )

    assert server_module.serve_cli(args) == 1
    assert "path too long" in capsys.readouterr().err.lower()
    import vpmdk_client

    pidfile = server_module.pidfile_path(
        vpmdk_client.resolve_socket_path(str(socket_path))
    )
    assert not os.path.exists(pidfile)


def test_daemon_ready_print_survives_a_dead_stdout_consumer(tmp_path, monkeypatch):
    # R183 (P2): the daemon launcher's success print was unguarded, so with
    # an unbuffered stdout whose consumer had exited (PYTHONUNBUFFERED=1 |
    # head -1), EPIPE surfaced inside print() itself -- past the flush-time
    # _drain_stream_guarded in serve_cli's finally -- and a SUCCESSFULLY
    # started, model-holding resident was reported as exit 1 with a raw
    # BrokenPipeError traceback. A supervisor reads that as "failed to
    # start" and orphans the resident.
    monkeypatch.setattr(
        server_module,
        "_daemonize",
        lambda *a, **k: (True, None, "READY:VPMDK server ready at X (pid 1)"),
    )

    class BrokenStdout:
        def write(self, text):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(sys, "stdout", BrokenStdout())

    args = SimpleNamespace(
        command="serve",
        dir=str(tmp_path),
        bcar=None,
        socket=str(tmp_path / "s.sock"),
        idle_timeout=0.0,
        daemon=True,
        log_file=None,
        daemon_notify_fd=None,
    )
    assert server_module.serve_cli(args) == 0
