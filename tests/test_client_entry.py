from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_HEAVY_MODULES = (
    "ase",
    "chgnet",
    "e3nn",
    "mace",
    "numpy",
    "pymatgen",
    "torch",
    "vpmdk_core",
)


@pytest.mark.parametrize(
    "arguments",
    [
        ["status"],
        ["run", "--dir", "."],
        ["stop", "--timeout", "0"],
    ],
)
def test_client_entry_does_not_import_ml_runtime(
    tmp_path: Path,
    arguments: list[str],
):
    socket_path = tmp_path / "missing.sock"
    command_arguments = [*arguments, "--socket", str(socket_path)]
    script = """
import json
import sys

from vpmdk_entry import main

exit_code = main(json.loads(sys.argv[1]))
heavy = json.loads(sys.argv[2])
print("IMPORTS=" + json.dumps(sorted(name for name in heavy if name in sys.modules)))
raise SystemExit(exit_code)
"""
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (src_dir, env.get("PYTHONPATH")) if part
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            json.dumps(command_arguments),
            json.dumps(_HEAVY_MODULES),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 3
    assert "IMPORTS=[]" in completed.stdout
    assert "Cannot connect to VPMDK server" in completed.stderr


def test_client_run_resolves_workdir_exactly_like_one_shot(tmp_path: Path, monkeypatch):
    # SERVER_MODE_SPEC 2.2: `run --dir D` must mean exactly what one-shot
    # `vpmdk --dir D` means, and run_workdir resolves with a bare os.path.abspath.
    # Expanding ~ here made the SAME argument resolve to two different
    # directories: the request went to $HOME/calc while one-shot read ./~/calc --
    # so the identical invocation could compute a different structure, write
    # outputs elsewhere, or exit 0 through the server and 1 one-shot. SPEC 1.1
    # forbids changing the one-shot side, so the client must not expand.
    from vpmdk_client import VPMDKClient

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    captured: dict[str, object] = {}

    def fake_request(self, request, *, timeout):
        captured["request"] = request
        yield {"event": "done", "ok": True}

    monkeypatch.setattr(VPMDKClient, "_request", fake_request)
    monkeypatch.chdir(tmp_path)

    for argument in ("~/calc", "relative/calc", str(tmp_path / "abs" / "calc")):
        VPMDKClient(str(tmp_path / "s.sock")).run(argument)
        workdir = captured["request"]["workdir"]
        # Byte-identical to what run_workdir would compute for the same string.
        assert workdir == os.path.abspath(argument)
        assert os.path.isabs(workdir)

    # Specifically: a literal tilde is NOT redirected to $HOME.
    VPMDKClient(str(tmp_path / "s.sock")).run("~/calc")
    assert captured["request"]["workdir"] == str(tmp_path / "~" / "calc")


def test_root_script_uses_lightweight_client_entry(tmp_path: Path):
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "vpmdk.py"),
            "status",
            "--socket",
            str(tmp_path / "missing.sock"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "Cannot connect to VPMDK server" in completed.stderr


def test_socket_creation_failure_is_classified_as_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # R125: socket() and settimeout() were called OUTSIDE the classified try, so an
    # OSError raised there (EMFILE/ENFILE near RLIMIT_NOFILE, ENOBUFS) escaped every
    # VPMDKClientError mapping as an uncaught traceback with an off-contract exit
    # code, instead of the documented exit 3 for an unreachable server.
    import socket as socket_module

    from vpmdk_client import ServerConnectionError, VPMDKClient

    # _verify_socket_ownership runs first and rejects a non-socket path, so bind a
    # real listener before any patching.
    real_socket = socket_module.socket
    socket_path = tmp_path / "server.sock"
    listener = real_socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)

    for failing_step in ("socket", "settimeout"):
        if failing_step == "socket":
            def fake_socket(*args, **kwargs):
                raise OSError(24, "Too many open files")

            monkeypatch.setattr(socket_module, "socket", fake_socket)
        else:
            class FailingSocket(real_socket):
                def settimeout(self, value):  # type: ignore[override]
                    raise OSError(105, "No buffer space available")

            monkeypatch.setattr(socket_module, "socket", FailingSocket)

        client = VPMDKClient(str(socket_path))
        with pytest.raises(ServerConnectionError) as excinfo:
            client._connect(deadline=None)
        assert "Cannot connect to VPMDK server" in str(excinfo.value), failing_step

    listener.close()


def test_read_current_umask_does_not_mutate_the_process_mask():
    # Cross-review (R181 window, P2): the os.umask read-back dance is
    # process-global, so two concurrent run() calls could interleave -- one
    # observing the other's temporary mask 0, or restoring 0 as the process
    # mask, after which every later file was world-writable. On Linux the
    # mask is now read from /proc/self/status without mutating anything; the
    # mutating fallback is serialized behind a module lock.
    import vpmdk_client

    previous = os.umask(0o027)
    try:
        def forbidden(*args, **kwargs):
            raise AssertionError("os.umask must not be called on the /proc path")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(vpmdk_client.os, "umask", forbidden)
            assert vpmdk_client._read_current_umask() == 0o027
        os.umask(0o077)
        assert vpmdk_client._read_current_umask() == 0o077
    finally:
        os.umask(previous)


def test_read_current_umask_fallback_is_correct_and_serialized():
    import builtins
    import vpmdk_client

    real_open = builtins.open

    def no_proc(path, *args, **kwargs):
        if str(path) == "/proc/self/status":
            raise OSError("no proc here")
        return real_open(path, *args, **kwargs)

    previous = os.umask(0o027)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(builtins, "open", no_proc)
            assert vpmdk_client._read_current_umask() == 0o027
            assert vpmdk_client._UMASK_LOCK.acquire(blocking=False)
            vpmdk_client._UMASK_LOCK.release()
    finally:
        os.umask(previous)


def test_client_rejects_invalid_connect_timeouts():
    # Cross-review (R181 window, P2): float() accepted negative, NaN,
    # infinite, or beyond-time_t values and the FIRST request then died in
    # socket.settimeout() -- outside the connection-error handlers, after
    # the socket object existed, leaving its descriptor to garbage
    # collection. The constructor now validates like the request timeout.
    import vpmdk_client

    for bad in (-1.0, float("nan"), float("inf"), -float("inf"), 0.0, 1e18):
        with pytest.raises(ValueError, match="connect_timeout"):
            vpmdk_client.VPMDKClient("/tmp/x.sock", connect_timeout=bad)

    client = vpmdk_client.VPMDKClient("/tmp/x.sock", connect_timeout=2.5)
    assert client.connect_timeout == 2.5
