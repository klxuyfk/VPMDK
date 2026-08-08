"""Resident-calculator server for VPMDK's VASP-style execution path."""

from __future__ import annotations

import contextlib
import errno
import functools
import importlib.metadata
import io
import itertools
import json
import logging
import math
import os
import queue
import select
import signal
import socket
import stat
import sys
import threading
import time
import traceback as traceback_module
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from vpmdk_protocol import (
    MAX_REQUEST_BYTES,
    PROTOCOL_VERSION,
    default_socket_path,
    resolve_socket_path,
)

try:
    # ASE raises PropertyNotImplementedError (a NotImplementedError subclass) at
    # COMPUTE time when a calculator cannot provide a requested property (e.g.
    # get_stress() on a forces-only model during ISIF>=3). That is a genuine
    # calculation failure, distinct from VPMDK's own NotImplementedError raises
    # for unsupported INPUT config (VTST ICHAIN / NFREE), so _execute_job must not
    # fold it into input_error. Empty-tuple fallback => isinstance is always False
    # (all NotImplementedError then classify as input_error, the prior behavior).
    from ase.calculators.calculator import (
        PropertyNotImplementedError as _ASEPropertyNotImplementedError,
    )
except Exception:  # pragma: no cover - ASE is always present in server mode
    _ASEPropertyNotImplementedError = ()

try:
    # Used to isolate the process-global numpy RNG per request: a resident MD run
    # (IBRION=0 + T>0) draws initial velocities from MaxwellBoltzmannDistribution,
    # which consumes np.random's global state. Without per-request save/restore,
    # request B advances that state so a repeated request A (and the A->B->A
    # isolation sequence) would produce different velocities than a fresh one-shot
    # process. Saving before and restoring after each job keeps every request
    # deterministic from the resident's startup state.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is always present in server mode
    _np = None

HEARTBEAT_INTERVAL = 30.0
STALE_SOCKET_GRACE_PERIOD = 0.5
# A client must deliver its single-line request promptly; it controls when it
# sends, so a short deadline is safe here.
REQUEST_READ_TIMEOUT = 5.0
# Event delivery is paced by how fast the client drains the stream, which is
# outside the server's control (pagers, slow callbacks). Keep this generous so a
# stalled reader never turns a successful calculation into a reported failure,
# while still bounding a client that stops reading altogether.
EVENT_SEND_TIMEOUT = 900.0


def _root():
    return sys.modules["vpmdk_core"]


@functools.lru_cache(maxsize=1)
def _package_version() -> str:
    # The installed version is constant for the process, so cache it: status()
    # is polled repeatedly and importlib.metadata.version walks sys.path and
    # parses on-disk distribution metadata on every call.
    try:
        return importlib.metadata.version("vpmdk")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def pidfile_path(socket_path: str) -> str:
    """Return the daemon pidfile paired with ``socket_path``."""

    return f"{socket_path}.pid"


def _parse_pidfile_metadata(text: str) -> tuple[int, str] | None:
    """Return trusted ownership metadata from a VPMDK pidfile."""

    lines = text.splitlines()
    if len(lines) not in (2, 3) or not lines[1].startswith("socket="):
        return None
    if len(lines) == 3 and not lines[2].startswith("starttime="):
        return None
    try:
        pid = int(lines[0])
    except ValueError:
        return None
    if pid <= 0:
        return None
    socket_path = lines[1].removeprefix("socket=")
    if not socket_path:
        return None
    return pid, os.path.realpath(os.path.abspath(socket_path))


def _parse_pidfile_starttime(text: str) -> str | None:
    """Return the recorded process start time, if the pidfile carries one."""

    for line in text.splitlines():
        if line.startswith("starttime="):
            value = line.removeprefix("starttime=").strip()
            return value or None
    return None


def _process_stat_fields(pid: int) -> tuple[str, str] | None:
    """Return (state, starttime) for ``pid`` from /proc, or None."""

    if pid <= 0:
        return None
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    # The comm field may contain spaces and parentheses; the fields resume
    # after the LAST ')'. state is field 3 and starttime field 22 of proc(5),
    # i.e. tokens 0 and 19 after comm.
    tail = raw.rsplit(b")", 1)[-1].split()
    if len(tail) < 20:
        return None
    return (
        tail[0].decode("ascii", "replace"),
        tail[19].decode("ascii", "replace"),
    )


def _process_start_time(pid: int) -> str | None:
    """Return the kernel's start time for ``pid`` (jiffies since boot).

    (pid, starttime) uniquely identifies a process for the machine's uptime:
    a recycled pid gets a different starttime, so a match proves the recorded
    process itself still EXISTS -- without inspecting its cmdline, which a
    default-socket ``vpmdk serve`` does not mention its socket in at all.
    Existence is not liveness: the caller must also check the state, because
    a SIGKILLed-but-unreaped ZOMBIE keeps its /proc entry and starttime.
    """

    fields = _process_stat_fields(pid)
    return None if fields is None else fields[1]


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pidfile_content(expected_socket: str) -> str:
    """Serialize this process's pidfile record.

    The third line records the kernel start time of this process so a later
    liveness check can identify it WITHOUT parsing its cmdline (a
    default-socket serve's cmdline does not mention the socket at all, which
    made the R136 force-drain protection silently inapplicable to the plain
    `vpmdk serve` invocation). Omitted where /proc is unavailable; readers
    fall back to the cmdline heuristic then.
    """

    lines = f"{os.getpid()}\nsocket={expected_socket}\n"
    start_time = _process_start_time(os.getpid())
    if start_time is not None:
        lines += f"starttime={start_time}\n"
    return lines


def _write_pidfile(pidfile: str, socket_path: str) -> None:
    """Create or replace metadata only when it belongs to this socket."""

    expected_socket = os.path.realpath(os.path.abspath(socket_path))
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(pidfile, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            fd = os.open(pidfile, flags)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to open daemon pidfile safely: {pidfile}"
            ) from exc
        # Check the TYPE on the raw fd first. os.fdopen(fd, "r+") raises
        # io.UnsupportedOperation("File or stream is not seekable") for a FIFO
        # BEFORE the S_ISREG test below could run, so `serve --daemon` died after
        # the full model load with a message naming neither the pidfile nor the
        # cause, while the correct, actionable message right below never ran. The
        # two sibling helpers were already hardened for exactly this hazard.
        try:
            existing_stat = os.fstat(fd)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(
                f"Unable to inspect daemon pidfile safely: {pidfile}"
            ) from exc
        if not stat.S_ISREG(existing_stat.st_mode):
            os.close(fd)
            raise RuntimeError(
                f"Refusing to overwrite non-regular pidfile: {pidfile}"
            )
        with os.fdopen(fd, "r+", encoding="utf-8") as file_obj:
            if existing_stat.st_uid != os.geteuid():
                # OWNERSHIP, checked on the fd we actually opened. Only the
                # recorded socket path and PID liveness were validated before, so
                # a pidfile another local user pre-planted at the predictable
                # <socket>.pid was treated as a candidate: with mode 0644 the
                # reopen failed EACCES and surfaced as the opaque "Unable to open
                # daemon pidfile safely" AFTER the full model load, and with 0666
                # VPMDK truncated and wrote the victim's PID into the attacker's
                # file. Reject it by owner, with a message that says why -- this is
                # the check _assert_private_log_path's docstring already claimed
                # this sibling performed.
                raise RuntimeError(
                    f"Refusing to use daemon pidfile owned by uid "
                    f"{existing_stat.st_uid} (expected {os.geteuid()}): {pidfile}"
                )
            try:
                existing_text = file_obj.read()
            except UnicodeDecodeError:
                # Non-UTF-8 bytes are as unparseable as malformed text, so fall
                # through to the "not owned by this socket" rejection below rather
                # than letting a bare UnicodeDecodeError escape. Uncaught it
                # aborted `serve --daemon` AFTER the full model load with a message
                # that never names the pidfile. The two sibling helpers
                # (_remove_stale_pidfile / _remove_owned_pidfile) already tolerate
                # undecodable bytes; this one was the odd one out, and it is the
                # one that has to produce the actionable diagnostic because
                # _remove_stale_pidfile deliberately LEAVES such a file in place.
                existing_text = ""
            metadata = _parse_pidfile_metadata(existing_text)
            if metadata is None or metadata[1] != expected_socket:
                raise RuntimeError(
                    f"Refusing to overwrite pidfile not owned by this VPMDK socket: {pidfile}"
                )
            if metadata[0] != os.getpid() and _pid_is_alive(metadata[0]):
                raise ServerAlreadyRunning(
                    f"VPMDK pidfile belongs to a live process for {socket_path}"
                )
            file_obj.seek(0)
            file_obj.truncate()
            file_obj.write(_pidfile_content(expected_socket))
            file_obj.flush()
            os.fchmod(file_obj.fileno(), 0o600)
        return

    with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
        file_obj.write(_pidfile_content(expected_socket))


def _remove_owned_pidfile(pidfile: str, socket_path: str, pid: int) -> None:
    """Remove a pidfile only when its socket and process ownership match."""

    # O_NONBLOCK: opening a FIFO read-only BLOCKS until a writer appears, so a
    # non-regular entry planted at the predictable <socket>.pid would hang
    # shutdown forever. With it the open returns at once and the S_ISREG check
    # below rejects the entry -- the same hazard _assert_private_log_path guards
    # for the sibling <socket>.log. O_NONBLOCK is inert for a regular file.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(pidfile, flags)
    except OSError:
        return
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            return
        with os.fdopen(fd, "r", encoding="utf-8") as file_obj:
            fd = -1
            metadata = _parse_pidfile_metadata(file_obj.read())
        expected_socket = os.path.realpath(os.path.abspath(socket_path))
        if metadata != (pid, expected_socket):
            return
        current_stat = os.lstat(pidfile)
        if current_stat.st_ino == file_stat.st_ino and stat.S_ISREG(current_stat.st_mode):
            os.unlink(pidfile)
    except (OSError, ValueError):
        # Best-effort removal: a corrupt/tampered pidfile whose bytes are not
        # valid UTF-8 raises UnicodeDecodeError (a ValueError) from read(); an
        # unverifiable pidfile is simply left in place rather than crashing the
        # shutdown cleanup (which must still unlink the socket and close logs).
        return
    finally:
        if fd >= 0:
            os.close(fd)


def _pid_is_live_server_for_socket(
    pid: int, socket_path: str, *, start_time: str | None = None
) -> bool:
    """Whether ``pid`` is a live VPMDK ``serve`` process for this exact socket.

    Deliberately narrower than "the pid is alive": a recycled pid belonging to an
    unrelated process must NOT block a restart. When the pidfile recorded the
    writer's kernel ``start_time``, (pid, starttime) is a recycle-proof process
    identity and settles the question directly -- essential because the cmdline
    heuristic below cannot see a DEFAULT-socket serve (its cmdline carries no
    socket path), which made the force-drain protection silently inapplicable
    to the plain `vpmdk serve` invocation. Without a recorded start time (older
    pidfile, /proc unavailable) the cmdline heuristic is the fallback, and it
    answers False wherever /proc is unavailable, which reduces to the previous
    behavior rather than guessing.
    """

    if pid <= 0:
        return False
    if start_time is not None:
        fields = _process_stat_fields(pid)
        if fields is not None:
            state, current = fields
            # A SIGKILLed server whose supervisor has not called wait() stays
            # in /proc as a ZOMBIE with its original starttime -- identity
            # matches, but the process holds nothing (no model, no fds) and
            # can never answer again. Treating it as live blocked every
            # restart with "refusing to replace it while its process holds
            # the model" until the negligent parent reaped it. 'X' (dead) is
            # the same for completeness.
            if state in ("Z", "X"):
                return False
            return current == start_time
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError:
        return False
    arguments = [item for item in raw.split(b"\0") if item]
    if not any(item == b"serve" for item in arguments):
        return False
    expected = os.path.realpath(os.path.abspath(socket_path))
    for item in arguments:
        candidate = item.decode("utf-8", errors="surrogateescape")
        try:
            if os.path.realpath(os.path.abspath(candidate)) == expected:
                return True
        except (OSError, ValueError):
            continue
    return False


def _pidfile_names_live_server(socket_path: str) -> bool:
    """Whether ``<socket>.pid`` records a LIVE vpmdk serve for this socket.

    Positive identification only: a missing/foreign/unparseable pidfile, a
    different recorded socket, or a merely-alive recycled PID all return
    False (blocking restarts on those is the deadlock _remove_stale_pidfile
    exists to prevent). Same fd-based open discipline as that helper so a
    symlink or FIFO planted at the predictable path cannot redirect or hang
    the check.
    """

    pidfile = pidfile_path(socket_path)
    try:
        fd = os.open(
            pidfile,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return False
        with os.fdopen(fd, "rb") as file_obj:
            fd = -1
            raw = file_obj.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return False
        metadata = _parse_pidfile_metadata(text)
        if metadata is None:
            return False
        if metadata[1] != os.path.realpath(os.path.abspath(socket_path)):
            return False
        return _pid_is_live_server_for_socket(
            metadata[0], socket_path, start_time=_parse_pidfile_starttime(text)
        )
    except OSError:
        return False
    finally:
        if fd >= 0:
            os.close(fd)


def _remove_stale_pidfile(socket_path: str) -> None:
    """Remove a leftover pidfile paired with a socket already found stale.

    SERVER_MODE_SPEC 2.1 keys staleness solely on the socket: once the socket has
    been found unresponsive and unlinked, the daemon that wrote ``<socket>.pid``
    is dead, so the pidfile is stale too -- even if its recorded PID has since
    been recycled to an unrelated live process. Remove it (matching only the
    recorded socket, NOT PID liveness) so a legitimate restart is not blocked by a
    false ServerAlreadyRunning in _write_pidfile after an ungraceful death.
    """

    pidfile = pidfile_path(socket_path)
    fd = -1
    try:
        # O_NONBLOCK so a FIFO planted at this predictable path cannot hang
        # `serve` startup forever (a read-only FIFO open waits for a writer);
        # the S_ISREG check below then rejects it. Inert for a regular file.
        fd = os.open(
            pidfile,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError:
        # No pidfile (nothing to remove) or a symlink (O_NOFOLLOW refuses to
        # follow it): in neither case is there a regular VPMDK pidfile to clean.
        return
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            return
        if file_stat.st_size == 0:
            # A ZERO-LENGTH pidfile is the daemon's own crash residue, not an
            # external supervisor's data: _write_pidfile's O_CREAT|O_EXCL makes an
            # empty file and flushes the bytes only on close, so a SIGKILL/power
            # loss in that window (or ext4 delayed allocation after a crash) leaves
            # <socket>.pid empty. Since the paired socket has ALREADY been found
            # unresponsive/absent before this runs, that empty file is stale.
            # Remove it -- otherwise the NEXT daemon's _write_pidfile hits the
            # FileExistsError branch, reads empty (metadata None), and raises "not
            # owned by this VPMDK socket", permanently blocking `serve --daemon`
            # restart until an operator manually deletes it (the exact deadlock
            # this restart-resilience function exists to prevent). Removing an
            # empty file destroys nothing, so it never clobbers foreign data. This
            # is race-free against a concurrent starting daemon: _bind writes the
            # pidfile only AFTER the socket is listening, so any half-written empty
            # pidfile coincides with a LIVE socket that prepare_socket_path rejects
            # via ServerAlreadyRunning before ever calling this.
            pass  # fall through to the inode-guarded unlink below
        else:
            with os.fdopen(fd, "rb") as file_obj:
                fd = -1
                raw = file_obj.read()
            try:
                pidfile_text = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Non-empty, undecodable bytes are NOT our crash residue (our
                # writes are short ASCII). Conservatively leave them: an external
                # tool may keep its own data at this path.
                return
            metadata = _parse_pidfile_metadata(pidfile_text)
            if metadata is None:
                # Non-empty but unparseable (e.g. a foreign file at this path):
                # only a zero-length file is treated as VPMDK crash residue, so
                # leave it in place and let _write_pidfile validate ownership.
                return
            if metadata[1] != os.path.realpath(os.path.abspath(socket_path)):
                # A well-formed pidfile recording a DIFFERENT socket is not ours.
                return
            if _pid_is_live_server_for_socket(
                metadata[0],
                socket_path,
                start_time=_parse_pidfile_starttime(pidfile_text),
            ):
                # This function's premise -- "an unresponsive socket means the
                # daemon that wrote <socket>.pid is dead" -- is FALSE during a
                # force-stop drain: teardown closes the listener before joining the
                # worker, so the daemon stops answering while it keeps running and
                # holding the model. Deleting its live pidfile then disarmed
                # _write_pidfile's ServerAlreadyRunning guard and let a second
                # resident load a SECOND copy of the model beside it (on a GPU, a
                # second full allocation).
                #
                # Only refuse when the recorded pid is POSITIVELY identified as a
                # vpmdk `serve` for THIS socket. A merely-alive pid may be an
                # unrelated process that recycled the number, and blocking restarts
                # on that is the deadlock this function exists to prevent.
                return
        current_stat = os.lstat(pidfile)
        if current_stat.st_ino == file_stat.st_ino and stat.S_ISREG(current_stat.st_mode):
            os.unlink(pidfile)
    except OSError:
        # Best-effort: a stat/unlink race (file already gone) must not crash
        # startup. _write_pidfile still validates ownership as defense in depth.
        return
    finally:
        if fd >= 0:
            os.close(fd)


def default_log_path(socket_path: str) -> str:
    return f"{socket_path}.log"


# Distinguishes each server's private logger in diagnostics without reusing
# id(self), whose values CPython recycles after an instance is freed.
_LOGGER_SEQUENCE = itertools.count(1)


def _assert_private_log_path(path: str) -> None:
    """Refuse a DERIVED log path occupied by something that is not our own file.

    Checked with lstat BEFORE opening, because opening is not safe for every file
    type: a pre-planted readerless FIFO makes a blocking open() hang forever, and
    logging.FileHandler re-opens by path with a plain blocking open() right after
    us, so an open-based check alone cannot prevent that hang.

    O_NOFOLLOW (kept below) only refuses a final-component SYMLINK. At the
    predictable ``<socket>.log`` an attacker can instead pre-plant a regular file
    they own (mode 0666) or a hard link: the open succeeds -- the 0600 mode
    argument is inert for an existing inode -- and the daemon then appends every
    log line, backend stdout line, workdir path and traceback into it, since the
    fd is dup2'd onto stdout and stderr. The sibling ``<socket>.pid`` validates
    ownership the same way, on the fd it opened (_write_pidfile).
    """

    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return  # absent: we create it ourselves, 0600
    except OSError:
        return  # unstatable: let the open below produce the clearer error
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(
            f"Refusing to write the default server log through a symlink: {path}"
        )
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(
            f"Refusing to write the default server log to a non-regular file: {path}"
        )
    if info.st_uid != os.geteuid():
        raise RuntimeError(
            f"Refusing to write the default server log to a file owned by uid "
            f"{info.st_uid} (expected {os.geteuid()}): {path}"
        )


def _verify_private_log_fd(fd: int, path: str) -> None:
    """Re-check the fd we actually opened, closing the lstat->open race window."""

    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(
            f"Refusing to write the default server log to a non-regular file: {path}"
        )
    if info.st_uid != os.geteuid():
        raise RuntimeError(
            f"Refusing to write the default server log to a file owned by uid "
            f"{info.st_uid} (expected {os.geteuid()}): {path}"
        )


def _create_private_log_file(path: str, *, refuse_symlink: bool = False) -> None:
    """Create ``path`` mode 0600 when absent, matching the daemon log open.

    ``logging.FileHandler`` creates its file with 0666 & ~umask -- typically
    0644 -- while the daemon path opens the SAME ``--log-file`` with an explicit
    ``os.open(..., 0o600)`` in _daemonize. Without this the two modes disagree
    for one flag: on a shared host a foreground run would leave the resident
    MLP/MODEL/DEVICE, every request's workdir path and full failure tracebacks
    world-readable. Worse, a mode passed to os.open applies only at CREATION, so
    a later ``serve --daemon`` pointed at a file a foreground run already made
    0644 merely appends and its 0600 never takes effect.

    Like the daemon path, an EXISTING file keeps its mode: the creation mode is
    the contract, and force-chmod'ing a file an operator deliberately widened
    (e.g. 0640 for a monitoring group) would be a surprising side effect.

    ``refuse_symlink`` marks the DERIVED <socket>.log -- a path the user never
    named, so anything already sitting there is an attack, not a setup. An
    EXPLICIT --log-file is deliberately left alone: that path is the user's own
    choice, where a symlink (log rotation, /var/log indirection) is legitimate.
    """

    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if not refuse_symlink:
        with contextlib.suppress(OSError):
            # Failures are left to the FileHandler below, which raises a clearer
            # error naming the path.
            os.close(os.open(path, flags, 0o600))
        return
    _assert_private_log_path(path)
    try:
        fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, getattr(errno, "EMLINK", errno.ELOOP)}:
            raise RuntimeError(
                f"Refusing to write the default server log through a symlink: {path}"
            ) from exc
        # Any other failure: let the FileHandler surface the clearer message.
        return
    try:
        _verify_private_log_fd(fd, path)
    finally:
        os.close(fd)

def _probe_foreground_log_file(path: str) -> int:
    """Open-check an explicit foreground ``--log-file`` BEFORE the model load.

    The daemon path opens the same flag's target in ``_daemonize`` before any
    backend work, so an unwritable path fails in well under a second -- while
    the foreground path deferred the identical open to ``VPMDKServer.__init__``,
    AFTER the full model load (the daemon/foreground half-mirror again). Worse,
    a FIFO planted at the path made the foreground ``serve`` block forever
    inside that post-load open, with no socket bound and no diagnostic.
    ``O_NONBLOCK`` turns the reader-less-FIFO hang into an immediate ENXIO and
    is inert for regular files; mode 0600 matches the real creation in
    ``_create_private_log_file`` so the probe does not weaken it. Failures
    raise OSError for serve_cli to report with the path named.

    Returns the OPEN fd; the caller must hold it until the logger's own
    FileHandler has opened the path and close it afterwards. Closing it
    immediately delivered EOF to a FIFO's waiting reader (``cat fifo``),
    which then exited -- so the post-load FileHandler reopened the FIFO
    with no reader left and blocked forever, the exact hang the probe
    exists to prevent (cross-review finding).
    """

    return os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NONBLOCK", 0),
        0o600,
    )


def _verify_socket_parent_ownership(dir_fd: int, dir_uid: int, parent: str) -> None:
    """Confirm the directory is owned by whoever owns files we create in it.

    Creates a short-lived probe file relative to ``dir_fd`` (openat, so it is
    confined to the already-``O_NOFOLLOW``-opened directory) and compares its
    owner to the directory's owner. This proves ownership even when a plain
    ``st_uid == geteuid()`` cannot: under root (which bypasses the mode) a
    foreign directory's owner differs from the probe's, while our own directory
    matches even when the filesystem reports a remapped/squashed uid for both.
    """

    probe_name = f".vpmdk-owner-probe-{os.getpid()}"
    probe_flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_NOFOLLOW", 0)
    )
    probe_fd: int | None = None
    for attempt in range(2):
        try:
            probe_fd = os.open(probe_name, probe_flags, 0o600, dir_fd=dir_fd)
            break
        except FileExistsError:
            if attempt == 0:
                # A stale probe from a crashed same-PID process lingers in our
                # own (already-0700) directory; remove it and retry once.
                with contextlib.suppress(OSError):
                    os.unlink(probe_name, dir_fd=dir_fd)
                continue
            # Still present after removal: another process is actively holding
            # the name (e.g. an attacker in a directory they own), so fail closed
            # rather than skip the ownership check.
            raise RuntimeError(
                f"cannot verify ownership of socket directory {parent}: its probe "
                "file name is being held by another process"
            )
        # Any other OSError (a read-only or full filesystem, etc.) is unrelated to
        # ownership; let it propagate with its real errno rather than misattribute
        # it to an ownership/security failure. A socket bind would surface the same
        # unusability with an equally clear error.
    assert probe_fd is not None
    try:
        probe_uid = os.fstat(probe_fd).st_uid
    finally:
        os.close(probe_fd)
        with contextlib.suppress(OSError):
            os.unlink(probe_name, dir_fd=dir_fd)
    if probe_uid != dir_uid:
        raise RuntimeError(
            f"socket directory {parent} is owned by uid {dir_uid}, not the user "
            f"that owns files created in it (uid {probe_uid}); refusing to use a "
            "directory another user controls"
        )


def _secure_private_socket_parent(parent: str) -> None:
    """Reject a symlinked/foreign default parent and ensure it is private (``0700``).

    The default socket parent lives under a world-writable base (``/tmp`` when
    ``XDG_RUNTIME_DIR`` is unset), where another user can pre-plant a symlink or
    a directory at the predictable ``/tmp/vpmdk-<uid>`` name. Open it with
    ``O_NOFOLLOW``/``O_DIRECTORY`` so a final-component symlink fails the open
    (never followed, closing the makedirs/chmod TOCTOU), verify ownership against
    a probe file (see ``_verify_socket_parent_ownership``), and ``fchmod`` it to
    ``0700`` only if it is still group/other-accessible -- so a
    chmod-restricted-but-writable filesystem (some overlay / 9p / DrvFs mounts)
    does not block an already-private parent.
    """

    open_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(parent, open_flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError(
                f"refusing to use a symlinked socket directory: {parent}"
            ) from exc
        if exc.errno in {errno.EACCES, errno.EPERM}:
            # A 0700 directory we cannot even open belongs to another user.
            raise RuntimeError(
                f"socket directory {parent} is not accessible to the current "
                "user; refusing to use a directory another user controls"
            ) from exc
        raise
    try:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"socket directory is not a directory: {parent}")
        # Ownership check: the directory must be owned by the same identity that
        # owns files we create inside it. Opening/chmod-ing a 0700 directory does
        # NOT prove ownership when running as root (root bypasses the mode), so a
        # root server would otherwise accept a directory an unprivileged user
        # pre-created at the predictable default path (e.g. /tmp/vpmdk-0) and let
        # them capture the socket. Comparing st_uid against a probe file we
        # create is robust where a plain ``st_uid == geteuid()`` is not: it works
        # under root, under uid-mapping, and under NFS root_squash (the directory
        # and probe report the same squashed uid) without false-rejecting our own
        # freshly-created directory.
        if info.st_mode & 0o077:
            # Tighten a group/other-accessible directory to 0700 FIRST, so the
            # ownership probe below runs inside a now-private directory that only
            # its owner (or root) can create files in -- closing the window where
            # an attacker could interfere with the probe in a world-writable
            # directory. fchmod succeeds only for the owner under a non-root euid,
            # so a failure here means the directory cannot be secured.
            try:
                os.fchmod(fd, 0o700)
            except OSError as exc:
                raise RuntimeError(
                    f"socket directory {parent} is accessible to other users "
                    f"and cannot be secured to 0700: {exc}"
                ) from exc
        _verify_socket_parent_ownership(fd, info.st_uid, parent)
    finally:
        os.close(fd)


def ensure_socket_directory(socket_path: str) -> None:
    """Create a private socket parent without weakening an existing directory."""

    parent = os.path.dirname(socket_path) or "."
    # Key the hardening on the PARENT DIRECTORY, not the full socket path. The
    # squattable artifact is the predictable directory ${XDG_RUNTIME_DIR:-/tmp}/
    # vpmdk-<uid> (as _secure_private_socket_parent's own docstring says), so
    # comparing filenames let ANY other socket name in that exact directory --
    # e.g. `--socket /tmp/vpmdk-1000/gpu0.sock` for a second GPU -- skip the
    # symlink rejection, the ownership probe and the 0700 tightening entirely,
    # while `default.sock` beside it was correctly refused. A genuinely custom
    # parent is still left exactly as the user set it.
    is_default = os.path.abspath(parent) == os.path.abspath(
        os.path.dirname(default_socket_path())
    )
    try:
        # mode=0o700 already yields a private directory for a freshly created
        # parent (0o700 has no group/other bits for umask to strip), so the
        # custom-socket path needs no separate chmod: issuing one would only
        # re-trigger the chmod-restricted-filesystem (DrvFs/9p/overlay) false
        # rejection that the default-path hardening was rewritten to avoid, and
        # an existing custom parent is intentionally left as the user set it.
        os.makedirs(parent, mode=0o700, exist_ok=True)
    except FileExistsError as exc:
        # exist_ok cannot swallow a non-directory occupying the path: a dangling
        # symlink (islink true, isdir false) or a plain file. Report the clear
        # rejection _secure_private_socket_parent would raise for a live symlink,
        # instead of a cryptic FileExistsError.
        if os.path.islink(parent):
            raise RuntimeError(
                f"refusing to use a symlinked socket directory: {parent}"
            ) from exc
        raise RuntimeError(
            f"socket directory path is not a directory: {parent}"
        ) from exc
    if is_default:
        # Symlink-reject + owner-verify + 0700-tighten in one fd-based step.
        _secure_private_socket_parent(parent)


def server_is_alive(socket_path: str, *, timeout: float = 0.5) -> bool:
    """Return whether a process accepts connections on the socket."""

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        try:
            connection.connect(socket_path)
        except OSError as exc:
            # Only errors proving that no listener owns the pathname make it
            # eligible for stale-socket cleanup. A timeout, permission error,
            # full listen backlog, or foreign protocol must be preserved.
            return exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}
    return True


def _reject_unusable_pidfile(socket_path: str) -> None:
    """Fail up front on a pidfile _write_pidfile is guaranteed to refuse.

    A pre-existing unusable entry at ``<socket>.pid`` -- a symlink, a FIFO, or
    foreign/corrupt content -- with no live server used to abort ``serve``
    only AFTER the full model load, inside _write_pidfile (post-bind). The
    refusal is fully decidable before loading; same shape as the deleted-
    socket/live-server check above. Mirrors _write_pidfile's own refusal
    conditions and messages so the outcome is identical, just earlier.
    """

    pidfile = pidfile_path(socket_path)
    try:
        fd = os.open(
            pidfile,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        # A symlink (O_NOFOLLOW -> ELOOP) or otherwise unopenable entry: the
        # post-load _write_pidfile would die with this same message.
        raise RuntimeError(
            f"Unable to open daemon pidfile safely: {pidfile}"
        ) from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(
                f"Refusing to overwrite non-regular pidfile: {pidfile}"
            )
        if file_stat.st_uid != os.geteuid():
            # The THIRD half-mirror of this pair: _write_pidfile also refuses
            # a foreign-owned pidfile (planted in a shared sticky /tmp), but
            # the gate omitted that condition, so the refusal came only after
            # the full model load -- on every retry, since the stale sweep
            # cannot unlink a foreign file there either.
            raise RuntimeError(
                f"Refusing to use daemon pidfile owned by uid "
                f"{file_stat.st_uid} (expected {os.geteuid()}): {pidfile}"
            )
        if file_stat.st_size == 0:
            return  # crash residue; the stale sweeps handle it
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise RuntimeError(
                f"Refusing to overwrite pidfile not owned by this VPMDK socket: {pidfile}"
            ) from None
        metadata = _parse_pidfile_metadata(text)
        if metadata is None or metadata[1] != os.path.realpath(
            os.path.abspath(socket_path)
        ):
            # BOTH halves of _write_pidfile's refusal (unparseable content OR
            # a recorded socket that differs): the first version mirrored only
            # the first half, so a well-formed pidfile recording another
            # socket (a moved runtime directory) still paid the full model
            # load before aborting -- on every retry, since the stale sweep
            # deliberately preserves a different-socket file.
            raise RuntimeError(
                f"Refusing to overwrite pidfile not owned by this VPMDK socket: {pidfile}"
            )
    finally:
        if fd >= 0:
            os.close(fd)


def prepare_socket_path(socket_path: str, *, pidfile_expected: bool = False) -> None:
    """Reject a live server or remove an unresponsive stale socket entry."""

    _prepare_socket_path_inner(socket_path)
    if pidfile_expected:
        # This launch WILL write a pidfile; one the post-load _write_pidfile
        # is guaranteed to refuse must stop the launch NOW, before the load
        # is paid for. Servers running without a pidfile (library API) leave
        # a foreign file alone, as before.
        _reject_unusable_pidfile(socket_path)


def _require_writable_socket_parent(socket_path: str) -> None:
    """Reject an unwritable socket parent BEFORE the model load is paid for.

    AF_UNIX ``bind()`` unconditionally needs write+search permission on the
    socket's directory, so an existing-but-unwritable CUSTOM parent (a
    root-owned /run, a read-only scheduler scratch dir, a typo) is a
    guaranteed post-load failure -- and ``listener.bind()``'s PermissionError
    carries ``filename=None``, so the operator saw a bare
    ``Error: PermissionError: [Errno 13] Permission denied`` naming no path
    at all, after the full model load. The DEFAULT parent was already
    covered pre-load by _verify_socket_parent_ownership's probe file; this
    is the custom-parent half of that pair. ``os.access`` is read-only, so
    a user-chosen directory is still never modified (the reason hardening is
    scoped to the default parent). A false positive here (exotic ACL/NFS)
    merely restores the old post-load failure.
    """

    parent = os.path.dirname(socket_path) or "."
    if os.access(
        parent,
        os.W_OK | os.X_OK,
        effective_ids=os.access in os.supports_effective_ids,
    ):
        return
    raise RuntimeError(
        f"socket directory is not writable: {parent} (creating a Unix socket "
        "requires write permission on its parent directory)"
    )


def _prepare_socket_path_inner(socket_path: str) -> None:
    """Reject a live server or remove an unresponsive stale socket entry."""

    ensure_socket_directory(socket_path)
    _require_writable_socket_parent(socket_path)
    if not os.path.lexists(socket_path):
        # "No socket => no live server" is FALSE when the socket file was
        # deleted EXTERNALLY (a /tmp ager, an accidental rm) under a living
        # server: the pidfile still names the live process. Without this
        # check, a restart passed here silently, paid for the FULL model load
        # (a second VRAM-sized allocation beside the deaf resident), and only
        # then failed in _write_pidfile with a message naming no pid -- a
        # guaranteed failure that was fully decidable up front.
        if _pidfile_names_live_server(socket_path):
            pid_text = "unknown pid"
            recorded_pid = None
            with contextlib.suppress(Exception):
                with open(pidfile_path(socket_path), encoding="utf-8") as handle:
                    metadata = _parse_pidfile_metadata(handle.read())
                if metadata is not None:
                    recorded_pid = metadata[0]
                    pid_text = f"pid {metadata[0]}"
            if recorded_pid == os.getpid():
                # THIS process's own pre-load endpoint reservation:
                # serve_cli writes the pidfile before the model load, and
                # VPMDKServer._bind re-runs this check before the socket
                # exists. A foreign starter is still refused below.
                return
            raise ServerAlreadyRunning(
                f"A VPMDK server ({pid_text}) is still running for "
                f"{socket_path}, but its socket file is not present -- "
                "either it was deleted externally under the live server, or "
                "the server is still starting up and has not bound its "
                "socket yet. Kill that process (or wait for it to exit or "
                "finish starting) before starting a new server."
            )
        # No socket and no live owner => any leftover pidfile is stale.
        _remove_stale_pidfile(socket_path)
        return
    try:
        socket_stat = os.lstat(socket_path)
    except FileNotFoundError:
        # The entry vanished between lexists() and here -- the normal race when a
        # dying server's _cleanup() unlinks the socket, e.g. `vpmdk stop
        # --timeout 0` (documented to return before shutdown completes) followed
        # immediately by `vpmdk serve`. The path is simply free now, so continue
        # startup instead of letting FileNotFoundError reach serve_cli's generic
        # handler and abort with "Error: [Errno 2] ..." and exit 1.
        _remove_stale_pidfile(socket_path)
        return
    mode = socket_stat.st_mode
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(
            f"Refusing to remove non-socket path at configured socket location: {socket_path}"
        )

    # A Unix socket pathname appears at bind(), just before listen()/accept().
    # Retry briefly so a concurrently starting server is not mistaken for a
    # stale socket and unlinked during that readiness gap.
    inode = socket_stat.st_ino
    deadline = time.monotonic() + STALE_SOCKET_GRACE_PERIOD
    while True:
        if server_is_alive(socket_path, timeout=0.1):
            raise ServerAlreadyRunning(f"VPMDK server is already running at {socket_path}")
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
        try:
            current_stat = os.lstat(socket_path)
        except FileNotFoundError:
            # The socket vanished during the grace wait (a concurrent serve that
            # also found it stale, a stop's _cleanup, a /tmp reaper). Sweep the
            # paired pidfile like EVERY other exit path of this function does --
            # this branch was the only one that skipped it, so a leftover
            # <socket>.pid could survive and make the following _write_pidfile
            # abort startup with "Refusing to overwrite pidfile not owned by this
            # VPMDK socket".
            _remove_stale_pidfile(socket_path)
            return
        if not stat.S_ISSOCK(current_stat.st_mode):
            raise RuntimeError(
                f"Socket path changed to a non-socket entry during startup: {socket_path}"
            )
        if current_stat.st_ino != inode:
            inode = current_stat.st_ino
            deadline = time.monotonic() + STALE_SOCKET_GRACE_PERIOD

    # An unresponsive socket is NOT proof the server is dead: a force-stop
    # drain closes the listener while the worker keeps computing and holding
    # the model. The paired pidfile is the liveness evidence that survives
    # that state -- if it positively names a live vpmdk serve for THIS socket,
    # refuse instead of unlinking the live server's socket out from under it.
    if _pidfile_names_live_server(socket_path):
        raise ServerAlreadyRunning(
            f"VPMDK server at {socket_path} is still running (its listener is "
            "closed, e.g. while draining a job after stop --force); refusing "
            "to replace it while its process holds the model."
        )

    try:
        if os.lstat(socket_path).st_ino != inode:
            return prepare_socket_path(socket_path)
        os.unlink(socket_path)
    except FileNotFoundError:
        _remove_stale_pidfile(socket_path)
        return
    # The socket was stale and has been unlinked; its paired pidfile (if any) is
    # therefore stale too, so remove it before _write_pidfile runs.
    _remove_stale_pidfile(socket_path)


class ServerAlreadyRunning(RuntimeError):
    """Raised when a live process owns the requested server socket."""


class BackendConfigurationMismatch(RuntimeError):
    """Raised when a request asks for a different resident backend."""

    def __init__(self, differences: list[str]):
        self.differences = differences
        super().__init__("Backend configuration mismatch: " + "; ".join(differences))


# These tags affect construction of the resident force-field calculator. Output,
# force-constant, and CHARGE_* tags intentionally remain request-scoped.
BACKEND_CONFIGURATION_TAGS = frozenset(
    {
        "MLP",
        "NNP",
        "MODEL",
        "DEVICE",
        "GRAPH_CONVERTER",
        "GRAPH_CONVERTER_ALGORITHM",
        "CHGNET_GRAPH_CONVERTER",
        "CHGNET_GRAPH_CONVERTER_ALGORITHM",
        "MATRIS_GRAPH_CONVERTER",
        "MATRIS_GRAPH_CONVERTER_ALGORITHM",
        "MATLANTIS_MODEL_VERSION",
        "MODEL_VERSION",
        "MATLANTIS_PRIORITY",
        "PRIORITY",
        "MATLANTIS_CALC_MODE",
        "CALC_MODE",
        "ORB_MODEL",
        "ORB_PRECISION",
        "ORB_COMPILE",
        "EQNORM_VARIANT",
        "EQNORM_COMPILE",
        "MATRIS_TASK",
        "MATTERSIM_COMPUTE_STRESS",
        "MATTERSIM_STRESS_WEIGHT",
        "ALPHANET_CONFIG",
        "ALPHANET_PRECISION",
        "ALPHANET_DTYPE",
        "HIENET_FILE_TYPE",
        "NEQUIX_BACKEND",
        "NEQUIX_USE_KERNEL",
        "NEQUIX_KERNEL",
        "NEQUIX_USE_COMPILE",
        "NEQUIX_COMPILE",
        "NEQUIX_CAPACITY_MULTIPLIER",
        "SEVENNET_FILE_TYPE",
        "SEVENNET_MODAL",
        "SEVENNET_ENABLE_CUEQ",
        "SEVENNET_ENABLE_FLASH",
        "SEVENNET_ENABLE_OEQ",
        "UPET_VERSION",
        "UPET_NON_CONSERVATIVE",
        "UPET_NEIGHBORLIST_DEVICE",
        "UPET_NL_DEVICE",
        "TACE_DTYPE",
        "TACE_SPIN_ON",
        "TACE_NEIGHBORLIST_BACKEND",
        "TACE_FIDELITY_IDX",
        "TACE_LEVEL",
        "FAIRCHEM_TASK",
        "FAIRCHEM_INFERENCE_SETTINGS",
        "FAIRCHEM_CONFIG",
        "FAIRCHEM_V1_PREDICTOR",
        "EQUIFORMER_V3_MODULE",
        "EQUIFORMER_V3_IMPORT_MODULE",
        "GRACE_PAD_NEIGHBORS_FRACTION",
        "GRACE_PAD_ATOMS_NUMBER",
        "GRACE_MAX_RECOMPILATION",
        "GRACE_MIN_DIST",
        "GRACE_FLOAT_DTYPE",
        "DEEPMD_TYPE_MAP",
        "DEEPMD_HEAD",
    }
)

# Tag prefixes whose tags are consumed by EXACTLY ONE backend builder and by no
# other backend (verified: each such tag is referenced in a single builder, with
# no cross-backend read and no cross-module delegation into it). A resident of a
# different backend ignores such a leftover tag exactly as the one-shot builder
# does, so the request-validation path drops it instead of raising a spurious
# exit-5 mismatch (SERVER_MODE_SPEC 3.4). Each maps prefix -> owning MLP.
#
# Deliberately EXCLUDED (kept and still compared, i.e. fail-closed) because their
# tags are shared across a builder family or read via delegation, where dropping
# one could hide a tag the resident builder actually consumes:
#   - SEVENNET_ (SevenNet + FlashTP + EQUFLASH's sevennet_family delegation)
#   - FAIRCHEM_ / EQUIFORMER_V3_ (fairchem module: FAIRCHEM/V2/ESEN/EQUIFORMER_V3)
#   - GRAPH_CONVERTER* (CHGNet/MatRIS + nequip_family, read via delegation)
#   - the non-prefixed Matlantis aliases MODEL_VERSION / PRIORITY / CALC_MODE
_EXCLUSIVE_BACKEND_TAG_PREFIXES = {
    "ORB_": "ORB",
    "MATTERSIM_": "MATTERSIM",
    "MATLANTIS_": "MATLANTIS",
    "UPET_": "UPET",
    "TACE_": "TACE",
    "GRACE_": "GRACE",
    "DEEPMD_": "DEEPMD",
    "EQNORM_": "EQNORM",
    "ALPHANET_": "ALPHANET",
    "HIENET_": "HIENET",
    "NEQUIX_": "NEQUIX",
    "CHGNET_": "CHGNET",
    "MATRIS_": "MATRIS",
}

# Each tuple lists a canonical tag followed by the names accepted by the
# corresponding backend builder, in the same precedence order.  Keeping this
# knowledge here prevents a resident server from rejecting a request merely
# because the startup and request BCAR files use different documented names.
# (canonical_tag, candidate_tags). A present-but-empty candidate value is always
# omitted (treated as "use the builder default", like a blank MODEL) in the loop
# below, so no per-group "skip falsy" flag is needed.
_CONFIGURATION_ALIAS_GROUPS = (
    ("MATLANTIS_MODEL_VERSION", ("MATLANTIS_MODEL_VERSION", "MODEL_VERSION")),
    ("MATLANTIS_PRIORITY", ("MATLANTIS_PRIORITY", "PRIORITY")),
    ("MATLANTIS_CALC_MODE", ("MATLANTIS_CALC_MODE", "CALC_MODE")),
    ("ALPHANET_PRECISION", ("ALPHANET_PRECISION", "ALPHANET_DTYPE")),
    ("NEQUIX_USE_KERNEL", ("NEQUIX_USE_KERNEL", "NEQUIX_KERNEL")),
    ("NEQUIX_USE_COMPILE", ("NEQUIX_USE_COMPILE", "NEQUIX_COMPILE")),
    ("UPET_NEIGHBORLIST_DEVICE", ("UPET_NEIGHBORLIST_DEVICE", "UPET_NL_DEVICE")),
    ("TACE_FIDELITY_IDX", ("TACE_FIDELITY_IDX", "TACE_LEVEL")),
)

_BOOLEAN_CONFIGURATION_TAGS = frozenset(
    {
        "ORB_COMPILE",
        "EQNORM_COMPILE",
        "MATTERSIM_COMPUTE_STRESS",
        "NEQUIX_USE_KERNEL",
        "NEQUIX_USE_COMPILE",
        "SEVENNET_ENABLE_CUEQ",
        "SEVENNET_ENABLE_FLASH",
        "SEVENNET_ENABLE_OEQ",
        "UPET_NON_CONSERVATIVE",
        "TACE_SPIN_ON",
        "FAIRCHEM_V1_PREDICTOR",
    }
)
_INTEGER_CONFIGURATION_TAGS = frozenset(
    {
        "MATLANTIS_PRIORITY",
        "TACE_FIDELITY_IDX",
        "GRACE_PAD_ATOMS_NUMBER",
        "GRACE_MAX_RECOMPILATION",
    }
)
_FLOAT_CONFIGURATION_TAGS = frozenset(
    {
        "MATTERSIM_STRESS_WEIGHT",
        "NEQUIX_CAPACITY_MULTIPLIER",
        "GRACE_PAD_NEIGHBORS_FRACTION",
        "GRACE_MIN_DIST",
    }
)
_LOWERCASE_CONFIGURATION_TAGS = frozenset(
    {
        "GRAPH_CONVERTER_ALGORITHM",
        "MATRIS_TASK",
        "HIENET_FILE_TYPE",
        "NEQUIX_BACKEND",
        "SEVENNET_FILE_TYPE",
    }
)
# Tags whose one-shot builders pass a blank "" straight into a strict coercer or
# validator (``_coerce_bool_tag`` / ``_coerce_int_tag`` / a strict ``float()`` /
# a ``_normalize_*`` that rejects unrecognized values) and therefore REJECT a
# present-but-empty value. For these a blank request value must NOT be treated as
# "use default": it is kept so ``_normalize_configuration_value`` reproduces the
# same rejection the one-shot builder would (a coerce error for bool/int/float,
# or a mismatch against the resident's real value for the string tags), keeping
# server and one-shot acceptance equivalent (SERVER_MODE_SPEC 3.4). Every OTHER
# optional tag defaults a blank -- via ``or DEFAULT`` (ORB_MODEL, MATRIS_TASK,
# MATLANTIS_PRIORITY, ALPHANET_PRECISION, SEVENNET_FILE_TYPE ...), an ``if raw:``
# guard (EQNORM_VARIANT), or ``_parse_optional_float`` returning ``None``
# (MATTERSIM_STRESS_WEIGHT, GRACE_PAD_NEIGHBORS_FRACTION, GRACE_MIN_DIST) -- so a
# blank is treated as omitted.
#
# Membership is per-tag by *builder behaviour*, NOT by value category: some
# int/float tags (MATLANTIS_PRIORITY; the three ``_parse_optional_float`` floats)
# default a blank and are therefore ACCEPTed (omitted, not listed here), while
# some string tags (NEQUIX_BACKEND, HIENET_FILE_TYPE, GRAPH_CONVERTER_ALGORITHM)
# pass a blank into a ``_normalize_*`` that raises and so ARE listed here.
_BLANK_REJECTED_TAGS = frozenset(
    _BOOLEAN_CONFIGURATION_TAGS
    | {
        # strict ``_coerce_int_tag`` (MATLANTIS_PRIORITY uses ``or`` -> excluded)
        "TACE_FIDELITY_IDX",
        "GRACE_PAD_ATOMS_NUMBER",
        "GRACE_MAX_RECOMPILATION",
        # strict ``float()`` (the other three floats use _parse_optional_float)
        "NEQUIX_CAPACITY_MULTIPLIER",
        # ``_normalize_*`` that raise on an unrecognized (incl. blank) value
        "NEQUIX_BACKEND",
        "HIENET_FILE_TYPE",
        "GRAPH_CONVERTER_ALGORITHM",
        # SevenNet passes the value straight through with ``if modal is not None``
        # (no ``or``/truthy gate), so a blank SEVENNET_MODAL= is an explicit empty
        # modal selection, NOT "use default" -- it must be kept and compared, not
        # omitted and silently reused from the resident.
        "SEVENNET_MODAL",
    }
)

# Sentinel returned by _normalize_configuration_value for a value the one-shot
# builder resolves to "ignore this tag / use the constructor default" (e.g. a
# non-numeric lenient-float tag). _canonical_configuration drops it, so the
# canonical config matches the calculator the builder actually constructs.
_OMIT_TAG = object()

# accept() errnos that are transient/recoverable: the resident server must log
# and keep serving rather than tear down the model + queued jobs. EMFILE/ENFILE
# (process/system fd exhaustion) clear as connections close; ECONNABORTED is a
# peer that vanished before accept; EINTR/EAGAIN/EWOULDBLOCK are spurious wakeups.
_TRANSIENT_ACCEPT_ERRNOS = frozenset(
    {
        errno.EMFILE,
        errno.ENFILE,
        errno.ECONNABORTED,
        errno.EINTR,
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.ENOBUFS,
        errno.ENOMEM,
    }
)
# Brief backoff after a transient accept() error so fd-exhaustion cases do not
# spin the CPU while connections drain (accept() would otherwise return the same
# error immediately).
_ACCEPT_RETRY_BACKOFF_S = 0.1

# Alias canonical tags whose builders resolve ``get(primary) or get(secondary)``
# and then feed the result into a STRICT coercer/validator that raises on a blank
# "" (but defaults a None). The ``or`` gives the FIRST truthy candidate, else the
# LAST operand's value: so a blank on the PRIMARY falls through to the secondary
# (``'' or None`` -> None -> default), but a blank SECONDARY with the primary
# absent resolves to ``''`` and the builder raises. These are NOT in
# _BLANK_REJECTED_TAGS (a blank primary must still default), so the alias loop
# keeps a blank *trailing* operand for them specifically, reproducing the same
# rejection instead of silently omitting it.
_OR_STRICT_ALIAS_TAGS = frozenset(
    {"MATLANTIS_PRIORITY", "ALPHANET_PRECISION", "UPET_NEIGHBORLIST_DEVICE"}
)

# Defaults set by VPMDK's backend builders before calculator construction.
# Constructor-owned defaults are intentionally not guessed here.
_FORCED_FLASH_CONFIGURATION_DEFAULTS = {
    "SEVENNET_FILE_TYPE": "checkpoint",
    "SEVENNET_ENABLE_CUEQ": False,
    "SEVENNET_ENABLE_FLASH": True,
    "SEVENNET_ENABLE_OEQ": False,
}
_BACKEND_CONFIGURATION_DEFAULTS: dict[str, dict[str, Any]] = {
    "MATLANTIS": {
        "MATLANTIS_MODEL_VERSION": "v8.0.0",
        "MATLANTIS_PRIORITY": 50,
        "MATLANTIS_CALC_MODE": "PBE",
    },
    "ORB": {"ORB_PRECISION": "float32-high"},
    "EQNORM": {"EQNORM_COMPILE": False},
    "MATRIS": {"MATRIS_TASK": "efs"},
    "ALPHANET": {"ALPHANET_PRECISION": "32"},
    "HIENET": {"HIENET_FILE_TYPE": "checkpoint"},
    "NEQUIX": {
        "NEQUIX_BACKEND": "jax",
        "NEQUIX_USE_KERNEL": False,
        "NEQUIX_USE_COMPILE": False,
        "NEQUIX_CAPACITY_MULTIPLIER": 1.1,
    },
    "SEVENNET": {"SEVENNET_FILE_TYPE": "checkpoint"},
    "FLASHTP": _FORCED_FLASH_CONFIGURATION_DEFAULTS,
    "EQUFLASH": _FORCED_FLASH_CONFIGURATION_DEFAULTS,
    "FAIRCHEM": {"FAIRCHEM_INFERENCE_SETTINGS": "default"},
    "FAIRCHEM_V2": {"FAIRCHEM_INFERENCE_SETTINGS": "default"},
    "ESEN": {"FAIRCHEM_INFERENCE_SETTINGS": "default"},
    "FAIRCHEM_V1": {"FAIRCHEM_V1_PREDICTOR": False},
    "EQUIFORMER_V3": {"FAIRCHEM_V1_PREDICTOR": False},
}

# MODEL is resolved through _normalize_model_identity, never through
# _normalize_configuration_value, so it is intentionally not listed here.
_PATH_CONFIGURATION_TAGS = frozenset({"ALPHANET_CONFIG", "FAIRCHEM_CONFIG"})
_MLP_IDENTITY_ALIASES = {
    "MATGL": "MATGL",
    "M3GNET": "MATGL",
    "FAIRCHEM": "FAIRCHEM",
    "FAIRCHEM_V2": "FAIRCHEM",
    "ESEN": "FAIRCHEM",
}


def _canonical_mlp_identity(mlp: str) -> str:
    """Return one identity for documented names using the same builder."""

    normalized = str(mlp).strip().upper()
    return _MLP_IDENTITY_ALIASES.get(normalized, normalized)


# Backends whose builders resolve a present-but-blank DEVICE to CPU via the
# `_resolve_device(...) or "cpu"` idiom (alphanet.py, eqnorm.py, hienet.py,
# sevennet_family.py -- the last shared by SevenNet/FlashTP/EquFlash). For these,
# `_resolve_device("")` returns "" and the trailing `or "cpu"` makes the device
# variable itself "cpu" BEFORE it reaches the calculator, so a blank-DEVICE
# resident actually runs on CPU and must ADVERTISE "cpu" -- otherwise it reports
# "" and rejects a later explicit DEVICE=cpu request as exit 5 even though both
# select the identical CPU calculator (SERVER_MODE_SPEC 3.4).
#
# MatGL/M3GNet is INCLUDED for the same "where does it actually land" reason,
# reached differently: `matgl.load_model()` builds the potential on CPU, and a
# blank DEVICE makes _move_module_to_device return WITHOUT relocating (a blank is
# not a device `.to()` accepts), so the potential stays on CPU. Letting the blank
# fall through to autodetect made a CUDA host advertise device="cuda" for a
# potential still on the CPU: a request explicitly naming "cuda" was ACCEPTED and
# then silently ran on the CPU, while "cpu" -- the device it really uses -- was
# rejected with exit 5. Fixing it here rather than by autodetecting inside the
# builder keeps one-shot behavior byte-identical (SPEC 1.1); the server simply
# stops describing a placement that never happened.
# MatRIS is INCLUDED even though only two of its three build paths apply the
# fallback, because those two are the ones real runs take: a local checkpoint and
# a named model (including the DEFAULT matris_10m_oam) both go through
# _load_matris_checkpoint_model (`model.to(device or "cpu")`) and
# _instantiate_matris_calculator (`calculator.device = device or "cpu"`), so a
# blank-DEVICE MatRIS resident genuinely runs on CPU. The third path -- passing
# the blank verbatim to MatRISCalculator(device="") -- is reached only for a model
# name that is neither a local path nor in the download registry. Excluding
# MatRIS sent its blank DEVICE down the generic "blank == autodetect" branch, so
# on a CUDA host a request repeating the resident's own blank BCAR canonicalized
# to "cuda" against a resident correctly advertising "cpu" -- a permanent exit 5
# for a config one-shot builds identically (and, with the roles reversed, a silent
# accept of a device one-shot would not have used).
# Backends whose builders never read the DEVICE tag at all: GRACE passes only
# pad/dtype kwargs, Matlantis only model_version/priority/calc_mode, and DeePMD
# only model/type_map/head (verified: no DEVICE reference, direct or indirect, in
# any of the three builders). For them DEVICE is a tag the one-shot builder
# IGNORES, so comparing it produced a permanent exit 5 for a request naming any
# device other than the host's autodetected one -- while `vpmdk --dir` on that
# same directory builds a byte-identical calculator. SERVER_MODE_SPEC 3.4 rejects
# only tags that DIFFER in effect, so it is dropped from the comparison (status
# still reports the resident's actual device, which stays informational).
_DEVICE_IGNORING_IDENTITIES = frozenset(
    _canonical_mlp_identity(m) for m in ("GRACE", "MATLANTIS", "DEEPMD")
)


_DEVICE_BLANK_TO_CPU_IDENTITIES = frozenset(
    _canonical_mlp_identity(m)
    for m in (
        # RACECalculator(device=None) selects cpu, and the BAM builder drops
        # a blank DEVICE before it reaches the calculator.
        "BAM",
        "EQNORM",
        "ALPHANET",
        "HIENET",
        "SEVENNET",
        "FLASHTP",
        "EQUFLASH",
        "MATRIS",
        "MATGL",
    )
)


def _resolve_backend_device(mlp: str | None, raw_device: Any) -> str:
    """Resolve DEVICE exactly as the selected backend's builder does.

    Mirrors ``_resolve_device`` and, for the backends in
    ``_DEVICE_BLANK_TO_CPU_IDENTITIES``, the trailing ``or "cpu"`` those builders
    apply, so a present-but-blank DEVICE canonicalizes to the same "cpu" the
    calculator actually runs on instead of "". ``raw_device`` is passed through
    verbatim (None => autodetect, "" => the blank the builder sees), so this is a
    faithful stand-in for the builder's own device resolution.
    """

    if mlp is not None and _canonical_mlp_identity(mlp) == _canonical_mlp_identity("BAM"):
        # bam-torch's RACECalculator.configure_device maps ONLY the literal
        # 'cpu' (case-sensitive, no index) to the cpu and EVERY other spelling
        # -- 'cuda', 'cuda:1', 'gpu', even 'CPU' or 'cpu:0' -- to
        # torch.device('cuda') when CUDA is available and to the cpu when it
        # is not, dropping the index entirely. Comparing raw spellings
        # therefore rejected (exit 5) request/resident pairs that build
        # byte-identical calculators (cuda vs cuda:1; cuda vs cpu on a
        # CUDA-less host) and silently equated pairs that differ in effect
        # (cpu vs cpu:0 on a CUDA host). Mirror the builder exactly: the
        # generic cpu:N/:0 normalizations below must NOT apply first.
        raw = "" if raw_device is None else str(raw_device).strip()
        if raw_device is None:
            raw = str(_root()._resolve_device(None)).strip()
        if not raw or raw == "cpu":
            # Blank: the builder drops it and RACECalculator selects cpu.
            return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    resolved = str(_root()._resolve_device(raw_device)).strip().lower()
    if resolved.startswith("cpu:") and resolved[4:].isdigit():
        # torch's cpu type has exactly ONE underlying device: 'cpu:1' builds
        # and computes byte-identically to 'cpu' in one-shot, so rejecting
        # the pair as a mismatch (exit 5) violated SPEC 3.4's
        # reject-only-what-DIFFERS rule. Every index collapses, not just 0.
        resolved = "cpu"
    if len(resolved) > 2 and resolved.endswith(":0"):
        # Index 0 IS the default device for every torch device type:
        # torch.device('cpu:0') == torch.device('cpu') (and measured: CHGNet
        # yields byte-identical energies for both spellings). Normalizing only
        # the literal 'cuda:0' left the cpu:0/cpu pair permanently rejected
        # with exit 5 in BOTH directions while one-shot computed it -- the
        # same one-sided-application-surface failure as ever. ':1' and higher
        # stay distinct: those are genuinely different devices.
        resolved = resolved[:-2]
    if not resolved and mlp is not None:
        if _canonical_mlp_identity(mlp) in _DEVICE_BLANK_TO_CPU_IDENTITIES:
            return "cpu"
        # Every OTHER builder forwards the blank to a calculator that treats a
        # falsy device as "autodetect", so a blank DEVICE selects exactly what an
        # omitted one does. Verified for the default backend: chgnet's
        # determine_device("") == determine_device(None) (its first line is
        # `use_device = use_device or os.getenv(...)`), and FairChem v1 likewise
        # drops a falsy device. Canonicalizing the blank to "" instead made a
        # resident started from a template BCAR with `DEVICE=` (an unset
        # ${VPMDK_DEVICE}) advertise device="" and then PERMANENTLY reject every
        # request naming the device it actually runs on -- exit 5 for a config the
        # one-shot builder accepts, breaking SERVER_MODE_SPEC 3.4 ("reject only
        # tags that DIFFER"). Resolve it the same way an omitted DEVICE resolves.
        return _resolve_backend_device(mlp, None)
    return resolved


def _normalize_path_or_name(value: str, base_dir: str) -> str:
    root = _root()
    text = os.path.expanduser(str(value).strip())
    candidate = text if os.path.isabs(text) else os.path.join(base_dir, text)
    looks_like_path = (
        root._has_path_separator_shape(text)
        or text.lower().endswith(root._CONFIG_PATH_SUFFIXES)
        or os.path.exists(candidate)
    )
    return os.path.realpath(candidate) if looks_like_path else text


def _normalize_model_identity(mlp: str, value: Any, base_dir: str) -> str:
    """Return the MODEL identity selected by the shared backend resolver."""

    reference = _root()._resolve_backend_model_reference(
        mlp,
        str(value),
        base_dir=base_dir,
    )
    if reference.value is None:  # pragma: no cover - explicit MODEL is nonempty
        raise ValueError(f"{mlp} MODEL did not resolve to a model identity.")
    return str(reference.identity or reference.value)


def _normalize_configuration_value(
    tag: str,
    value: Any,
    base_dir: str,
    *,
    device: str | None = None,
    mlp: str | None = None,
) -> Any:
    root = _root()
    text = str(value).strip()
    if tag in _PATH_CONFIGURATION_TAGS:
        return _normalize_path_or_name(text, base_dir)
    if tag == "DEVICE":
        # Resolve exactly as the selected backend's builder does (including the
        # `or "cpu"` blank fallback for the backends that apply it), so a blank
        # DEVICE compares equal to the CPU those builders actually run on. `text`
        # is the stripped tag value; a present-but-blank DEVICE ("") stays "" here
        # unless `mlp` marks a cpu-defaulting backend.
        return _resolve_backend_device(mlp, text)
    if tag in {"MLP", "NNP"}:
        return text.upper()
    if tag in _BOOLEAN_CONFIGURATION_TAGS:
        return root._coerce_bool_tag(value, tag)
    if tag in _INTEGER_CONFIGURATION_TAGS:
        return root._coerce_int_tag(value, tag)
    if tag in _FLOAT_CONFIGURATION_TAGS:
        if tag == "NEQUIX_CAPACITY_MULTIPLIER":
            # The Nequix builder uses strict float(); mirror it so a malformed
            # value like "junk1.1tail" is rejected here too rather than leniently
            # parsed to 1.1 and accepted, which would compute a request one-shot
            # mode rejects. (Other float tags use _parse_optional_float in their
            # builders too, so the lenient path below matches them.)
            try:
                return float(text)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid {tag} value: {value!r}") from None
        # The other float tags (MATTERSIM_STRESS_WEIGHT, GRACE_MIN_DIST,
        # GRACE_PAD_NEIGHBORS_FRACTION) are parsed by their builders with
        # _parse_optional_float, which returns None for BOTH a blank AND a
        # non-numeric value ("none", "auto", ...) and the builder then DROPS the
        # kwarg and uses the constructor default. Mirror that exactly: a None
        # result means "the builder ignores this tag", so canonicalize it to
        # _OMIT_TAG (treated as omitted) instead of raising. Raising here would
        # reject a request the one-shot builder accepts, and -- for a resident
        # started with such a value -- crash backend_identity AFTER the model is
        # already loaded (SERVER_MODE_SPEC 3.4 non-equivalence).
        parsed = root._parse_optional_float(value, key=tag)
        if parsed is None:
            return _OMIT_TAG
        return float(parsed)
    if tag == "ALPHANET_PRECISION":
        normalized = text.lower()
        if normalized in {"32", "float32", "fp32"}:
            return "32"
        if normalized in {"64", "float64", "fp64"}:
            return "64"
        raise ValueError(f"Invalid ALPHANET_PRECISION value: {value!r}")
    if tag == "EQNORM_VARIANT":
        return root._normalize_eqnorm_variant(text)
    if tag == "MATLANTIS_CALC_MODE":
        return text.upper()
    if tag == "UPET_NEIGHBORLIST_DEVICE":
        resolved = root._normalize_upet_neighborlist_device(text, device)
        if resolved is None:
            # None means "run the neighbor list on the model's device". For
            # config comparison, canonicalize it to that device so the default
            # "auto"/"model" (recorded as None) and an equivalent explicit "cpu"
            # on a CPU model compare equal instead of falsely mismatching.
            return "cuda" if str(device or "").strip().lower().startswith("cuda") else "cpu"
        return resolved
    if tag == "DEEPMD_TYPE_MAP":
        return tuple(item for item in root.re.split(r"[\s,]+", text) if item)
    if tag in {"EQUIFORMER_V3_MODULE", "EQUIFORMER_V3_IMPORT_MODULE"}:
        return tuple(
            module.strip()
            for module in text.replace(";", ",").split(",")
            if module.strip()
        )
    if tag in _LOWERCASE_CONFIGURATION_TAGS:
        return text.lower()
    return text


def _canonical_configuration(
    tags: Mapping[str, Any],
    *,
    base_dir: str,
    mlp: str,
    device: str | None = None,
    tolerant: bool = False,
) -> dict[str, Any]:
    """Return explicitly supplied construction tags under effective names.

    With ``tolerant=True`` (the resident/startup path), a tag whose value fails
    to canonicalize is DROPPED instead of raising. That is safe there because the
    resident calculator has already been built successfully, so any tag whose
    value the strict coercers reject is necessarily a leftover the resident's
    builder ignored (a foreign tag) -- exactly what the one-shot builder ignores
    too. The request path keeps ``tolerant=False`` so a malformed request tag
    still raises and is reported as an exit-5 backend mismatch.
    """

    normalized = {str(key).upper(): value for key, value in tags.items()}

    def _normalize_or_drop(tag: str, value: Any) -> Any:
        # Returns _OMIT_TAG when the value cannot be canonicalized and we are in
        # tolerant mode (drop the offending foreign tag); otherwise normalizes.
        if not tolerant:
            return _normalize_configuration_value(
                tag, value, base_dir, device=device, mlp=mlp
            )
        try:
            return _normalize_configuration_value(
                tag, value, base_dir, device=device, mlp=mlp
            )
        except (ValueError, OverflowError, FileNotFoundError):
            return _OMIT_TAG
    alias_groups = list(_CONFIGURATION_ALIAS_GROUPS)

    # MODEL is Matlantis' final model-version fallback.  For other backends it
    # remains the checkpoint/model selector and must retain path normalization.
    if mlp == "MATLANTIS":
        alias_groups[0] = (
            "MATLANTIS_MODEL_VERSION",
            ("MATLANTIS_MODEL_VERSION", "MODEL_VERSION", "MODEL"),
        )

    # CHGNet and MatRIS share the graph converter resolver.  Backend-specific
    # and generic spellings therefore describe one effective setting.
    if mlp in {"CHGNET", "MATRIS"}:
        alias_groups.append(
            (
                "GRAPH_CONVERTER_ALGORITHM",
                (
                    f"{mlp}_GRAPH_CONVERTER_ALGORITHM",
                    f"{mlp}_GRAPH_CONVERTER",
                    "GRAPH_CONVERTER_ALGORITHM",
                    "GRAPH_CONVERTER",
                ),
            )
        )

    configuration: dict[str, Any] = {}
    aliases: set[str] = set()
    for canonical_tag, candidate_tags in alias_groups:
        aliases.update(candidate_tags)
        for candidate_tag in candidate_tags:
            if candidate_tag not in normalized:
                continue
            value = normalized[candidate_tag]
            is_or_strict_trailing = (
                canonical_tag in _OR_STRICT_ALIAS_TAGS
                and candidate_tag == candidate_tags[-1]
            )
            if (
                not str(value).strip()
                and canonical_tag not in _BLANK_REJECTED_TAGS
                and not is_or_strict_trailing
            ):
                # A present-but-empty value for a tag whose builder defaults a
                # blank (via ``or``, an ``if raw:`` guard, or _parse_optional_float
                # -> None) means "use the builder default", exactly like a blank
                # MODEL, so omit it. Tags in _BLANK_REJECTED_TAGS fall through so
                # the same coerce/validation rejection the one-shot builder raises
                # is preserved instead of silently accepting the request.
                #
                # Exception: for an _OR_STRICT_ALIAS_TAGS tag, a blank on the FINAL
                # ``or`` operand (with no earlier truthy candidate) is what the
                # builder actually coerces -- ``get(primary) or get(secondary)`` ==
                # ``None or ''`` == ``''`` -- and its strict coercer raises. Keep
                # that blank so the server rejects it too; a blank on the primary
                # (a non-final operand) is still skipped above and defaults.
                continue
            if canonical_tag == "MODEL":
                configuration[canonical_tag] = _normalize_model_identity(
                    mlp, value, base_dir
                )
            else:
                normalized_value = _normalize_or_drop(canonical_tag, value)
                # _OMIT_TAG => the builder ignores this value (use its default),
                # so drop it from the canonical config (equivalent to omitting).
                if normalized_value is not _OMIT_TAG:
                    configuration[canonical_tag] = normalized_value
            break

    if mlp == "EQUIFORMER_V3":
        module_tags = ("EQUIFORMER_V3_MODULE", "EQUIFORMER_V3_IMPORT_MODULE")
        modules: list[str] = []
        aliases.update(module_tags)
        for module_tag in module_tags:
            if module_tag in normalized:
                module_values = _normalize_or_drop(module_tag, normalized[module_tag])
                if module_values is not _OMIT_TAG:
                    modules.extend(module_values)
        if modules:
            configuration["EQUIFORMER_V3_MODULES"] = tuple(dict.fromkeys(modules))

    for tag in sorted(BACKEND_CONFIGURATION_TAGS - {"MLP", "NNP"}):
        if tag not in normalized or tag in aliases:
            continue
        if (
            not str(normalized[tag]).strip()
            and tag not in _BLANK_REJECTED_TAGS
            and tag != "DEVICE"
        ):
            # A present-but-empty tag whose builder defaults a blank (via
            # ``X = get(tag) or DEFAULT``, an ``if raw:`` guard, or
            # _parse_optional_float -> None) means "use the backend default",
            # exactly like omitting it (and like a blank MODEL). Recording "" here
            # would overwrite the effective default and make status/the exit-5
            # check disagree with the calculator the builder constructed. Tags in
            # _BLANK_REJECTED_TAGS fall through so a blank raises the same
            # validation rejection the builder does, keeping server/one-shot
            # acceptance equal.
            #
            # DEVICE is EXCLUDED from this rule, but it is NOT canonicalized to
            # "": _resolve_backend_device resolves a blank the way the selected
            # builder does -- "cpu" for the `or "cpu"` family, otherwise the same
            # autodetect an omitted DEVICE gets (chgnet's determine_device("") ==
            # determine_device(None); FairChem v1 also drops a falsy device). So a
            # blank DEVICE is compared against a REAL device rather than being
            # either silently accepted or turned into a permanent exit-5.
            continue
        if tag == "MODEL":
            configuration[tag] = _normalize_model_identity(
                mlp, normalized[tag], base_dir
            )
            continue
        normalized_value = _normalize_or_drop(tag, normalized[tag])
        # _OMIT_TAG => the builder ignores this value (use its default), so drop
        # it from the canonical config (equivalent to omitting the tag).
        if normalized_value is not _OMIT_TAG:
            configuration[tag] = normalized_value
    return configuration


def _effective_configuration(
    tags: Mapping[str, Any],
    *,
    base_dir: str,
    mlp: str,
    device: str,
    explicit_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return builder defaults overlaid with explicit canonical settings."""

    root = _root()
    normalized = {str(key).upper(): value for key, value in tags.items()}
    defaults = dict(_BACKEND_CONFIGURATION_DEFAULTS.get(mlp, {}))
    if mlp == "ORB":
        defaults["ORB_MODEL"] = root.DEFAULT_ORB_MODEL
    if mlp == "ALPHANET" and not str(normalized.get("ALPHANET_CONFIG", "")).strip():
        # AlphaNet's builder INFERS the config JSON when ALPHANET_CONFIG is
        # omitted (the single .json beside the checkpoint, or the named model's
        # cached config). Record that inferred path so the resident advertises the
        # config it actually loaded: otherwise a request whose BCAR spells out the
        # very same file -- which `vpmdk --dir` runs with a byte-identical
        # calculator -- compares request='<path>' against server=None and is
        # rejected with exit 5, contradicting SERVER_MODE_SPEC 3.4 ("reject only
        # tags that DIFFER") and the documented "Matching backend tags repeated |
        # Accepted". Same shape as the EQNORM_VARIANT inference below. The helper
        # never downloads and returns None when it cannot infer without side
        # effects, leaving the previous (unadvertised) behavior intact.
        inferred_config = root._infer_alphanet_config_path(normalized, base_dir=base_dir)
        if inferred_config:
            defaults["ALPHANET_CONFIG"] = inferred_config
    if mlp == "EQNORM" and not str(normalized.get("EQNORM_VARIANT", "")).strip():
        # Treat a present-but-blank EQNORM_VARIANT= like an omitted one: the
        # builder's `if explicit_variant:` guard ignores a blank and infers the
        # variant from the model, so the effective config must infer it too.
        # Otherwise the resident would advertise no EQNORM_VARIANT and a request
        # naming the identical inferred variant would be wrongly rejected (exit 5).
        model = str(normalized.get("MODEL") or root.DEFAULT_EQNORM_MODEL)
        spec = root._resolve_eqnorm_named_model_spec(model)
        if spec is not None:
            defaults["EQNORM_VARIANT"] = spec["model_variant"]
        else:
            candidate = os.path.basename(model)
            while candidate:
                variant = root._match_eqnorm_variant(candidate)
                if variant is not None:
                    defaults["EQNORM_VARIANT"] = variant
                    break
                stem, extension = os.path.splitext(candidate)
                if not extension:
                    break
                candidate = stem
    if mlp == "UPET":
        defaults["UPET_NEIGHBORLIST_DEVICE"] = "auto"
    if mlp in {"FAIRCHEM", "FAIRCHEM_V2", "ESEN"}:
        model = normalized.get("MODEL") or root.DEFAULT_FAIRCHEM_MODEL
        if model == root.DEFAULT_FAIRCHEM_MODEL:
            defaults["FAIRCHEM_TASK"] = root.DEFAULT_FAIRCHEM_TASK
    if mlp == "SEVENNET":
        # Reuse the builder's resolver so the "one enabled forces the other two
        # off" semantics have a single source of truth. Record those implied
        # False flags so an equivalent request that spells out e.g.
        # SEVENNET_ENABLE_FLASH=false is not rejected as request=False,
        # server=None despite selecting the same calculator. Resident tags are
        # already conflict-validated by the builder; guard the conflict case so
        # config computation never raises regardless.
        try:
            cueq, flash, oeq = root._resolve_sevennet_accelerators(normalized)
        except ValueError:
            cueq = flash = oeq = None
        # Always record the RESOLVED accelerator state (False when unset), not
        # only when one is enabled. A plain resident (all flags None) otherwise
        # has no SEVENNET_ENABLE_* keys, so a request that explicitly spells out a
        # no-op disable (e.g. SEVENNET_ENABLE_CUEQ=0 -> False) is rejected as
        # request=False vs server=None even though both select the identical
        # non-accelerated calculator (SERVER_MODE_SPEC 3.4). Recording the implied
        # False makes the equivalent request match; an omitted flag is still not
        # compared, and an enabling request (True) still mismatches a plain
        # resident, as it must.
        defaults["SEVENNET_ENABLE_CUEQ"] = bool(cueq)
        defaults["SEVENNET_ENABLE_FLASH"] = bool(flash)
        defaults["SEVENNET_ENABLE_OEQ"] = bool(oeq)

    effective = {
        tag: _normalize_configuration_value(
            tag, value, base_dir, device=device, mlp=mlp
        )
        for tag, value in defaults.items()
    }
    if explicit_configuration is None:
        explicit_configuration = _canonical_configuration(
            normalized, base_dir=base_dir, mlp=mlp, device=device
        )
    effective.update(explicit_configuration)
    effective["MLP"] = mlp
    effective["DEVICE"] = device
    return effective


def backend_identity(tags: Mapping[str, Any], *, base_dir: str) -> dict[str, Any]:
    """Return effective status metadata plus explicit construction tags."""

    root = _root()
    normalized_tags = {str(key).upper(): value for key, value in tags.items()}
    mlp = root._resolve_mlp_tag(dict(normalized_tags))
    # Drop leftover tags that belong to a DIFFERENT backend before canonicalizing,
    # exactly as validate_request_backend does for requests. The resident's own
    # builder ignores such a foreign tag (one-shot `vpmdk --dir` runs fine), so
    # normalizing it here would raise for a strict-typed foreign value (e.g.
    # ALPHANET_PRECISION=16 or NEQUIX_CAPACITY_MULTIPLIER=auto under an MLP=CHGNET
    # resident) and crash `serve` AFTER a full model load -- a startup failure
    # with no one-shot analog (SERVER_MODE_SPEC 1-2). Keeping this symmetric with
    # the request path also ensures a foreign tag is absent from both sides of the
    # comparison. Only verified-exclusive foreign prefixes are stripped, so a tag
    # the resident builder actually consumes is never dropped.
    #
    # _strip_foreign_backend_tags alone is NOT enough to prevent the startup
    # crash: it deliberately does not strip family-shared (SEVENNET_/FAIRCHEM_)
    # or non-prefixed (bare PRIORITY) foreign tags. So canonicalization below runs
    # with tolerant=True, which DROPS any surviving foreign tag whose value the
    # strict coercers reject (e.g. PRIORITY=high, SEVENNET_ENABLE_CUEQ=maybe).
    # That is safe because the model already built successfully, so such a tag is
    # necessarily one the resident's builder ignored.
    relevant_tags = _strip_foreign_backend_tags(
        normalized_tags, _canonical_mlp_identity(mlp)
    )
    # Resolve as the builder does: for cpu-defaulting backends a blank DEVICE
    # becomes "cpu" (not ""), so the resident advertises the CPU it actually runs
    # on and an equivalent DEVICE=cpu request is accepted rather than exit-5'd.
    device = _resolve_backend_device(mlp, relevant_tags.get("DEVICE"))
    canonical_explicit = _canonical_configuration(
        relevant_tags, base_dir=base_dir, mlp=mlp, device=device, tolerant=True
    )
    explicit = dict(canonical_explicit)
    explicit["MLP"] = mlp
    explicit["DEVICE"] = device
    effective = _effective_configuration(
        relevant_tags,
        base_dir=base_dir,
        mlp=mlp,
        device=device,
        explicit_configuration=canonical_explicit,
    )

    model: str | None = None
    if mlp == "MATLANTIS":
        model = effective["MATLANTIS_MODEL_VERSION"]
    elif str(normalized_tags.get("MODEL", "")).strip():
        # _canonical_configuration already resolved the explicit MODEL identity;
        # reuse it instead of repeating the same filesystem resolution.
        model = canonical_explicit["MODEL"]
    elif mlp == "ORB" and str(normalized_tags.get("ORB_MODEL", "")).strip():
        # ORB_MODEL only names the resident model for ORB itself; a leftover tag
        # must not make another backend advertise ORB's model. A present-but-blank
        # ORB_MODEL= is NOT a name: the builder resolves it to DEFAULT_ORB_MODEL
        # (`get("ORB_MODEL") or DEFAULT_ORB_MODEL`), so fall through to the default
        # branch below rather than advertising an empty model in `status`.
        model = str(normalized_tags["ORB_MODEL"]).strip()
    elif mlp == "ORB":
        # ORB's resident named-model identity comes from ORB_MODEL (default when
        # omitted or blank). MODEL is reserved for an optional local checkpoint.
        model = root.DEFAULT_ORB_MODEL
    else:
        try:
            # Resolve omission directly. Reclassifying a default model name as
            # an explicit value could mistake a same-named local entry in the
            # startup directory for the model actually selected by the builder.
            model = root._resolve_backend_model_reference(mlp, None).value
        except ValueError:
            model = None
    return {
        "mlp": mlp,
        "model": model,
        "device": device,
        "configuration": explicit,
        "effective_configuration": effective,
        "base_dir": os.path.realpath(base_dir),
    }


# Backend "families": tag groups read by a family of MLPs that share (or delegate
# to) a builder. A request tag from a family the resident is NOT a member of is
# ignored -- the resident's builder never reads it, exactly as one-shot does.
# Verified empirically from the backend builders + delegations (grep):
#   SEVENNET_* : sevennet_family (SEVENNET, FLASHTP) AND misc's EQUFLASH, which
#                delegates via _build_sevennet_family_calculator. HIENET has its
#                own builder and does NOT read SEVENNET_*, so it is excluded.
#   FAIRCHEM_*/EQUIFORMER_V3_* : the fairchem module (FAIRCHEM/FAIRCHEM_V2/ESEN/
#                EQUIFORMER_V3/FAIRCHEM_V1).
#   PRIORITY/MODEL_VERSION/CALC_MODE : the non-prefixed Matlantis aliases.
# Membership uses canonical MLP identities. GRAPH_CONVERTER* is deliberately NOT a
# family rule: CHGNET/MATRIS (and nequip_family's NEQUIP/ALLEGRO) all read and
# apply it via the shared resolver, so it stays compared (fail-closed) -- a
# GRAPH_CONVERTER override is a genuine config difference, not a foreign tag.
_SEVENNET_FAMILY_IDENTITIES = frozenset(
    # SEVENNET_* are all read by the shared _build_sevennet_family_calculator, so
    # SevenNet, FlashTP AND EquFlash (which delegates to it) consume every one --
    # not disjoint, so a single family set is correct.
    _canonical_mlp_identity(m) for m in ("SEVENNET", "FLASHTP", "EQUFLASH")
)
_FAIRCHEM_FAMILY_IDENTITIES = frozenset(
    _canonical_mlp_identity(m)
    for m in ("FAIRCHEM", "FAIRCHEM_V2", "ESEN", "EQUIFORMER_V3", "FAIRCHEM_V1")
)
_MATLANTIS_FAMILY_IDENTITIES = frozenset({_canonical_mlp_identity("MATLANTIS")})
# Attribute each FairChem-family tag to the EXACT builders that read it. The
# modern FAIRCHEM/FAIRCHEM_V2/ESEN builder reads only FAIRCHEM_TASK/
# FAIRCHEM_INFERENCE_SETTINGS; EQUIFORMER_V3_MODULE/_IMPORT_MODULE only by
# _build_equiformer_v3_calculator. FAIRCHEM_CONFIG/FAIRCHEM_V1_PREDICTOR are read
# by _build_fairchem_v1_calculator, which is reached BOTH from the FAIRCHEM_V1
# path AND from _build_equiformer_v3_calculator (it delegates unconditionally,
# forwarding the full bcar_tags) -- so EQUIFORMER_V3 is a real consumer of these
# two tags and MUST be an owner, else an EQUIFORMER_V3 resident strips them and
# never compares them, silently accepting a mismatched predictor/config (an
# under-comparison / SERVER_MODE_SPEC 3.4 breach). _build_fairchem_calculator
# (modern) does NOT read them, so it stays out.
_FAIRCHEM_TAG_OWNERS = {
    "FAIRCHEM_TASK": frozenset({_canonical_mlp_identity("FAIRCHEM")}),
    "FAIRCHEM_INFERENCE_SETTINGS": frozenset({_canonical_mlp_identity("FAIRCHEM")}),
    "EQUIFORMER_V3_MODULE": frozenset({_canonical_mlp_identity("EQUIFORMER_V3")}),
    "EQUIFORMER_V3_IMPORT_MODULE": frozenset({_canonical_mlp_identity("EQUIFORMER_V3")}),
    "FAIRCHEM_CONFIG": frozenset(
        {_canonical_mlp_identity("FAIRCHEM_V1"), _canonical_mlp_identity("EQUIFORMER_V3")}
    ),
    "FAIRCHEM_V1_PREDICTOR": frozenset(
        {_canonical_mlp_identity("FAIRCHEM_V1"), _canonical_mlp_identity("EQUIFORMER_V3")}
    ),
}
# The generic graph-converter spellings are read (and applied) ONLY by the CHGNet
# and MatRIS builders' _resolve_graph_converter_algorithm call; NEQUIP/ALLEGRO
# only DEFINE that helper and never call it. So a MACE/ORB/... resident ignores
# them and must not be forced into an exit-5 mismatch. (The backend-prefixed
# CHGNET_/MATRIS_ spellings are already owned via _EXCLUSIVE_BACKEND_TAG_PREFIXES.)
_GRAPH_CONVERTER_OWNERS = frozenset(
    _canonical_mlp_identity(m) for m in ("CHGNET", "MATRIS")
)


def _tag_owner_mlp_identities(tag: str) -> "frozenset[str] | None":
    """Return the canonical MLP identities whose builders read ``tag``.

    Returns None for a tag that is generic (MODEL/DEVICE/MLP/NNP) or simply
    unknown -- those are always kept and compared (fail-closed). Otherwise the
    returned set is the exact family that consumes the tag; a resident outside it
    ignores the tag, so it is dropped from the comparison to match one-shot.
    """

    for prefix, owner in _EXCLUSIVE_BACKEND_TAG_PREFIXES.items():
        if tag.startswith(prefix):
            return frozenset({_canonical_mlp_identity(owner)})
    if tag.startswith("SEVENNET_"):
        return _SEVENNET_FAMILY_IDENTITIES
    if tag in _FAIRCHEM_TAG_OWNERS:
        return _FAIRCHEM_TAG_OWNERS[tag]
    if tag.startswith("FAIRCHEM_") or tag.startswith("EQUIFORMER_V3_"):
        # Unknown FairChem-family tag: keep for the whole family (fail-closed).
        return _FAIRCHEM_FAMILY_IDENTITIES
    if tag in {"GRAPH_CONVERTER", "GRAPH_CONVERTER_ALGORITHM"}:
        return _GRAPH_CONVERTER_OWNERS
    if tag in {"PRIORITY", "MODEL_VERSION", "CALC_MODE"}:
        return _MATLANTIS_FAMILY_IDENTITIES
    return None


def _strip_foreign_backend_tags(
    normalized: Mapping[str, Any], resident_mlp_identity: str
) -> dict[str, Any]:
    """Drop request tags that belong to a DIFFERENT backend than the resident.

    A tag consumed only by backend family F is ignored by the one-shot builder of
    any resident outside F, so keeping it would raise a spurious exit-5 mismatch
    (SERVER_MODE_SPEC 3.4). Relevance is by verified family membership
    (_tag_owner_mlp_identities): a resident inside the tag's family keeps it (it
    is a genuine config), and generic / shared-resolver / unknown tags are always
    kept -- so the server never ignores a tag its resident builder might consume.
    """

    result: dict[str, Any] = {}
    for tag, value in normalized.items():
        owners = _tag_owner_mlp_identities(tag)
        if owners is not None and resident_mlp_identity not in owners:
            continue
        result[tag] = value
    return result


def _infer_request_type_map(request_base_dir: str) -> tuple[str, ...] | None:
    """Return the DeepMD type map one-shot would infer for this request.

    Mirrors ``_build_deepmd_calculator``'s tagless path (``_infer_type_map`` on
    the calculation structure) so the comparison sees the value one-shot would
    actually use. Returns None when the structure cannot be read, so an
    unreadable POSCAR stays an input error from the run itself instead of being
    reported as a backend mismatch.
    """

    root = _root()
    potcar_path = os.path.join(request_base_dir, "POTCAR")
    potcar_arg = potcar_path if os.path.exists(potcar_path) else None
    try:
        # An NEB band legitimately has NO top-level POSCAR: run_workdir dispatches
        # to run_neb_images on the numbered image directories, and one-shot builds
        # a calculator PER IMAGE from that image's structure. Inferring only from
        # a top-level POSCAR therefore skipped the comparison entirely for a band
        # (returning None), letting every image run under the resident's ordering
        # -- and where a top-level POSCAR did exist it validated a structure the
        # calculation never uses. Infer from the structures the run actually
        # evaluates.
        image_dirs = []
        if root._is_neb_like_incar(root._load_incar(os.path.join(request_base_dir, "INCAR"))):
            image_dirs = root._discover_neb_image_directories(request_base_dir)
        if image_dirs:
            maps = set()
            for image_dir in image_dirs:
                structure = root.read_structure(
                    root._resolve_neb_image_structure_path(image_dir), potcar_arg
                )
                image_map = tuple(root._infer_type_map(structure))
                if image_map:
                    maps.add(image_map)
            if len(maps) != 1:
                # No single ordering describes the band (or none could be read).
                # A band whose images disagree is rejected by the NEB validator
                # itself, so leave the classification to the run.
                return None
            return maps.pop()

        poscar_path = os.path.join(request_base_dir, "POSCAR")
        if not os.path.exists(poscar_path):
            return None
        structure = root.read_structure(poscar_path, potcar_arg)
        inferred = root._infer_type_map(structure)
    except Exception:
        return None
    if not inferred:
        return None
    # Same canonical shape as the tag's own normalization, so the two compare.
    return tuple(inferred)


def _device_tag_is_inert(resident_mlp_identity: str, resident_config: Mapping[str, Any]) -> bool:
    """Whether the DEVICE tag cannot change the resident backend's calculator.

    Single home for the whole rule, so the unconditional and CONDITIONAL cases
    cannot drift apart:

    * GRACE / Matlantis / DeePMD builders never read DEVICE at all.
    * NEQUIX reads it ONLY under ``if backend == "torch" and requested_device:``
      (backends/nequix.py), and an unset NEQUIX_BACKEND resolves to "jax" -- so on
      the default backend the tag is inert, while on torch it genuinely moves the
      model and must still be compared.

    Deciding this at COMPARISON time (rather than stripping the tag earlier) also
    keeps the resident's advertised device faithful to what the user configured.
    """

    if resident_mlp_identity in _DEVICE_IGNORING_IDENTITIES:
        return True
    if resident_mlp_identity == _canonical_mlp_identity("NEQUIX"):
        backend = str(resident_config.get("NEQUIX_BACKEND", "jax")).strip().lower()
        return backend != "torch"
    return False


def _nequix_torch_only_tag_is_inert(
    resident_mlp_identity: str, resident_config: Mapping[str, Any]
) -> bool:
    """Whether NEQUIX's torch-only compile tag cannot change the calculator.

    Upstream nequix stores ``use_compile`` but reads it ONLY in its torch
    branch, and an unset NEQUIX_BACKEND resolves to "jax" -- so under the
    default backend two calculators differing only in NEQUIX_USE_COMPILE /
    NEQUIX_COMPILE are bit-for-bit identical, and comparing the tag rejected
    (exit 5) a request `vpmdk --dir` builds and runs identically. The same
    conditional-inertness rule _device_tag_is_inert already applies to
    NEQUIX's DEVICE under jax; this is its sibling. The symmetric torch half
    (NEQUIX_CAPACITY_MULTIPLIER is read only in _pad_graph_jax, so it would
    be inert under torch) is deliberately NOT mirrored: no torch nequix
    checkpoint is available to verify it empirically, and unverifiable
    premises are not built on (the UPET/MACE precedent).
    """

    if resident_mlp_identity != _canonical_mlp_identity("NEQUIX"):
        return False
    backend = str(resident_config.get("NEQUIX_BACKEND", "jax")).strip().lower()
    return backend != "torch"


def _configuration_values_match(requested: Any, server_value: Any) -> bool:
    """Compare two canonical configuration values, treating NaN as self-equal.

    NEQUIX_CAPACITY_MULTIPLIER mirrors its builder's strict ``float()``, which
    accepts "nan" -- so a resident started with that tag records a NaN in its own
    effective configuration. Plain ``!=`` then makes ``nan != nan`` True, and the
    resident PERMANENTLY rejects a request whose BCAR is byte-identical to its own
    startup BCAR, reporting the self-contradictory
    "NEQUIX_CAPACITY_MULTIPLIER request=nan, server=nan". No request could ever
    satisfy it, while ``vpmdk --dir`` on the same directory runs fine.

    Fixing it HERE rather than by dropping NaN during canonicalization keeps the
    comparison fail-closed: two NaNs describe the same configuration, but a NaN
    request against a numeric resident (or the reverse) still differs and is still
    reported, instead of silently vanishing from the comparison.
    """

    if requested == server_value:
        return True
    return (
        isinstance(requested, float)
        and isinstance(server_value, float)
        and math.isnan(requested)
        and math.isnan(server_value)
    )


def _resident_backend_tags(resident: Mapping[str, Any]) -> dict[str, str]:
    """Return the resident's backend tags for capability resolution.

    ``effective_configuration`` already carries the builder defaults the resident
    actually runs with (e.g. MATRIS_TASK), which is exactly what a request that
    omits those tags inherits. MLP is forced from the resident record so the
    mapping never resolves to BackendConfig's CHGNET default.
    """

    tags = {
        str(key).upper(): value
        for key, value in dict(
            resident.get("effective_configuration")
            or resident.get("configuration")
            or {}
        ).items()
    }
    mlp = str(resident.get("mlp", "") or "").strip()
    if mlp:
        tags["MLP"] = mlp
    return tags


def validate_request_backend(
    resident: Mapping[str, Any],
    request_tags: Mapping[str, Any],
    *,
    request_base_dir: str,
) -> None:
    """Reject explicitly requested calculator settings that are not resident."""

    normalized = {str(key).upper(): value for key, value in request_tags.items()}
    resident_config = dict(
        resident.get("effective_configuration", resident.get("configuration", {}))
    )
    differences: list[str] = []
    resident_mlp = str(resident.get("mlp", ""))
    resident_mlp_identity = _canonical_mlp_identity(resident_mlp)

    try:
        # Inside the guard below: a blank MLP=/NNP= in a request BCAR is a
        # request-side selector problem (exit 5), not a calculation failure.
        if "MLP" in normalized or "NNP" in normalized:
            requested_mlp = _root()._resolve_mlp_tag(dict(normalized))
            if _canonical_mlp_identity(requested_mlp) != resident_mlp_identity:
                differences.append(
                    f"MLP request={requested_mlp!r}, server={resident_mlp!r}"
                )
                # Report the MLP difference NOW and stop. Canonicalizing the rest
                # of the request would run it through the RESIDENT's backend
                # policy, which is the wrong policy for these tags: e.g. a valid
                # CHGNet named MODEL resolved under MACE's local-only policy
                # raises FileNotFoundError, and the except below would replace the
                # real difference with "MACE MODEL path not found: <workdir>/
                # CHGNet-v0.3.0" -- naming a backend and a checkpoint path the
                # user never wrote, instead of enumerating the differing tag as
                # SERVER_MODE_SPEC 3.4 requires. Comparisons across two different
                # backends are meaningless anyway; the MLP is the difference to fix.
                raise BackendConfigurationMismatch(differences)

        requested_config = _canonical_configuration(
            _strip_foreign_backend_tags(normalized, resident_mlp_identity),
            base_dir=request_base_dir,
            mlp=resident_mlp,
            device=str(resident.get("device", "")),
        )
    except (FileNotFoundError, ValueError) as exc:
        # Request-side backend selectors are part of resident compatibility,
        # not a calculation failure. Preserve exit code 5 even when an
        # explicit MODEL is unknown or points to a missing checkpoint.
        raise BackendConfigurationMismatch(
            [f"Request configuration is invalid: {exc}"]
        ) from exc
    device_is_inert = _device_tag_is_inert(resident_mlp_identity, resident_config)
    nequix_compile_is_inert = _nequix_torch_only_tag_is_inert(
        resident_mlp_identity, resident_config
    )
    for tag, requested in sorted(requested_config.items()):
        if (
            tag in ("NEQUIX_USE_COMPILE", "NEQUIX_COMPILE")
            and nequix_compile_is_inert
        ):
            # See _nequix_torch_only_tag_is_inert: dead under jax, so a
            # differing value does not make the calculators differ (SPEC 3.4).
            continue
        if tag == "DEVICE" and device_is_inert:
            # The resident's builder cannot act on DEVICE, so a differing value
            # does not make the request's calculator differ (SERVER_MODE_SPEC 3.4
            # rejects only tags that DIFFER in effect). Comparing it rejected a
            # configuration `vpmdk --dir` builds identically.
            continue
        server_value = resident_config.get(tag)
        if tag == "DEVICE" and server_value is None:
            server_value = resident.get("device")
        if tag == "MODEL" and server_value is None:
            server_value = resident.get("model")
        if not _configuration_values_match(requested, server_value):
            differences.append(f"{tag} request={requested!r}, server={server_value!r}")

    if (
        resident_mlp_identity == _canonical_mlp_identity("DEEPMD")
        and "DEEPMD_TYPE_MAP" not in requested_config
    ):
        # DEEPMD_TYPE_MAP is the one tag whose ABSENCE does not mean "reuse the
        # resident". One-shot's builder INFERS the map from the calculation's own
        # structure when the tag is missing, so an omitted tag means "use MY
        # species ordering" -- which a resident whose type_map is baked into the
        # loaded DP calculator cannot honor. Left uncompared, a request whose
        # POSCAR lists species in a different order was accepted and silently
        # evaluated under the RESIDENT's ordering: wrong species mapping, wrong
        # energies and forces, no diagnostic. _validate_resident_backend_tags
        # already refuses to infer at startup for exactly this reason.
        #
        # Infer the same map one-shot would and compare it, rather than rejecting
        # every tagless request: a batch whose structures share the resident's
        # ordering is the normal use case and must keep working. If the structure
        # cannot be read, leave it alone -- the run itself will report that as an
        # input error rather than a backend mismatch.
        inferred = _infer_request_type_map(request_base_dir)
        if inferred is not None:
            server_value = resident_config.get("DEEPMD_TYPE_MAP")
            if not _configuration_values_match(inferred, server_value):
                differences.append(
                    f"DEEPMD_TYPE_MAP request={inferred!r} (inferred from the "
                    f"request POSCAR), server={server_value!r}; set an explicit "
                    "DEEPMD_TYPE_MAP matching the resident server"
                )

    # A present-but-blank (or omitted) MODEL carries no model intent: per
    # SERVER_MODE_SPEC §3.4 only backend tags that *differ* from the resident are
    # rejected, and an unspecified MODEL means "reuse the resident model". It is
    # therefore intentionally not compared here — resolving it to a default and
    # comparing against resident['model'] mis-rejects backends whose model
    # identity is not derived from MODEL (ORB's ORB_MODEL, Matlantis' version).

    if differences:
        raise BackendConfigurationMismatch(differences)


def _grace_ignored_device_request_warning(
    resident: Mapping[str, Any],
    request_tags: Mapping[str, Any],
) -> str | None:
    """Warn when a GRACE request carries the inert DEVICE tag.

    DEVICE is deliberately dropped from the GRACE identity comparison
    (_DEVICE_IGNORING_IDENTITIES) because the builder never reads it, and the
    resident builder never re-runs per request -- so the "GRACE ignores the
    DEVICE tag" warning the byte-identical one-shot run prints from the
    builder would otherwise never reach a request client. Synthesize the same
    message, preserving one-shot output equivalence.
    """

    root = _root()
    if _canonical_mlp_identity(str(resident.get("mlp", ""))) != "GRACE":
        return None
    normalized = {str(key).upper(): value for key, value in request_tags.items()}
    if "MLP" in normalized or "NNP" in normalized:
        try:
            requested_mlp = root._resolve_mlp_tag(dict(normalized))
        except (ValueError, FileNotFoundError):
            return None
        if _canonical_mlp_identity(requested_mlp) != "GRACE":
            # A genuine backend switch is handled by the mismatch check, not here.
            return None
    if not str(normalized.get("DEVICE") or "").strip():
        return None
    return root._GRACE_DEVICE_IGNORED_WARNING


def _unknown_grace_request_warning(
    resident: Mapping[str, Any],
    request_tags: Mapping[str, Any],
    *,
    base_dir: str,
) -> str | None:
    """Warn when a request names an unknown GRACE model reused as the resident default.

    GRACE's resolver maps an unrecognized foundation-model name to the installed
    default instead of raising, so a MODEL typo resolves to the resident identity
    and slips past the exit-5 mismatch check. A one-shot run would print this
    warning from the calculator builder, but the resident builder never re-runs,
    so the substitution would otherwise be entirely silent for a request client.
    Surface the same warning the builder emits, preserving one-shot equivalence.
    """

    root = _root()
    if _canonical_mlp_identity(str(resident.get("mlp", ""))) != "GRACE":
        return None
    normalized = {str(key).upper(): value for key, value in request_tags.items()}
    if "MLP" in normalized or "NNP" in normalized:
        try:
            requested_mlp = root._resolve_mlp_tag(dict(normalized))
        except (ValueError, FileNotFoundError):
            return None
        if _canonical_mlp_identity(requested_mlp) != "GRACE":
            # A genuine backend switch is handled by the mismatch check, not here.
            return None
    requested_model = str(normalized.get("MODEL") or "").strip()
    if not requested_model:
        return None
    # A recognized foundation name is never a substitution: short-circuit on this
    # one registry scan for the common valid request, so only the rare
    # foundation-unknown request (a typo or an explicit local checkpoint path)
    # falls through to the reference resolution below.
    foundation_model = root._resolve_grace_foundation_model(requested_model)
    if foundation_model is not None:
        return None
    # foundation_model is None for both a name typo and a local checkpoint path,
    # so resolve the reference to tell them apart: only a NAMED_MODEL that GRACE
    # silently mapped to the resident default warrants the warning; a real path
    # (which validation already confirmed matches the resident) must not warn.
    # The resolution reuses foundation_model below so the registry is not scanned
    # again inside the shared predicate.
    try:
        reference = root._resolve_backend_model_reference(
            "GRACE", requested_model, base_dir=base_dir
        )
    except (ValueError, FileNotFoundError):
        # An unresolvable request MODEL is rejected by validate_request_backend.
        return None
    if not root._grace_substitutes_unknown_model(
        root, reference, requested_model, foundation_model=foundation_model
    ):
        return None
    return (
        f"Warning: Unknown GRACE model '{requested_model}', reusing resident "
        f"default {reference.value} instead."
    )


def _validate_resident_backend_tags(tags: Mapping[str, Any]) -> str:
    """Validate startup tags whose safety requirements differ in server mode."""

    normalized = {str(key).upper(): value for key, value in tags.items()}
    mlp = _root()._resolve_mlp_tag(dict(normalized))
    if mlp == "DEEPMD" and not str(normalized.get("DEEPMD_TYPE_MAP", "")).strip():
        raise ValueError(
            "DEEPMD server mode requires an explicit DEEPMD_TYPE_MAP in the "
            "startup BCAR; inferring it from the startup POSCAR is unsafe when "
            "the resident calculator is reused for other structures."
        )
    return mlp


def _validate_startup_model_path(
    tags: Mapping[str, Any], *, base_dir: str, mlp: str
) -> Any:
    """Resolve startup MODEL using the same policy as its calculator builder."""

    normalized = {str(key).upper(): value for key, value in tags.items()}
    return _root()._resolve_backend_model_reference(
        mlp,
        normalized.get("MODEL"),
        base_dir=base_dir,
    )


def _detect_calculator_device(calculator) -> str | None:
    """Return a calculator/model device attribute when one is exposed."""

    candidates = _root()._calculator_candidates(calculator)
    for candidate in list(candidates):
        for attribute in ("calculator", "model"):
            try:
                nested = getattr(candidate, attribute, None)
            except Exception:
                continue
            if nested is not None and all(nested is not existing for existing in candidates):
                candidates.append(nested)
    for candidate in candidates:
        for attribute in ("device", "use_device"):
            try:
                value = getattr(candidate, attribute, None)
            except Exception:
                continue
            if value is not None:
                text = str(value).strip().lower()
                if text:
                    # ':0' is the default index for every device type; see
                    # _resolve_backend_device.
                    return text[:-2] if text.endswith(":0") else text
    return None


def _detect_calculator_graph_converter_algorithm(calculator) -> str | None:
    """Return the loaded model's graph-converter algorithm when exposed.

    Same READ-not-guess pattern as _detect_calculator_device: a tagless
    CHGNET resident used to advertise GRAPH_CONVERTER_ALGORITHM=None, so a
    request spelling out the algorithm the bundled default model ALREADY uses
    (measured: CHGNet v0.3.0 ships algorithm='fast') was rejected exit 5
    while one-shot computed byte-identical numbers -- a SPEC 3.4 violation.
    Reading the value from the already-loaded calculator advertises reality:
    an explicit matching request now matches, and a resident genuinely loaded
    with a legacy converter still correctly rejects 'fast'.
    """

    candidates = _root()._calculator_candidates(calculator)
    for candidate in list(candidates):
        for attribute in ("calculator", "model"):
            try:
                nested = getattr(candidate, attribute, None)
            except Exception:
                continue
            if nested is not None and all(nested is not existing for existing in candidates):
                candidates.append(nested)
    for candidate in candidates:
        try:
            converter = getattr(candidate, "graph_converter", None)
        except Exception:
            continue
        if converter is None:
            continue
        try:
            algorithm = getattr(converter, "algorithm", None)
        except Exception:
            continue
        if algorithm is not None:
            text = str(algorithm).strip().lower()
            if text:
                return text
    return None


_TRUNCATION_MARKER = "\n...[truncated to VPMDK protocol limit]"


def _encode_utf8_lenient(text: str) -> bytes:
    """UTF-8 bytes for text that may hold surrogate-escaped bytes.

    Exception/log strings often embed filesystem paths decoded with
    errors="surrogateescape" (lone surrogates U+DC80-DCFF), which have no plain
    UTF-8 form. A strict ``.encode("utf-8")`` would raise; used on the event or
    daemon-notify paths that raise would be swallowed and drop the message, so
    every raw-text encode goes through this surrogate-tolerant path.
    """

    return text.encode("utf-8", errors="surrogatepass")


def _text_has_surrogate(text: str) -> bool:
    """Return whether text contains a byte with no plain UTF-8 form."""

    try:
        text.encode("utf-8")
        return False
    except UnicodeEncodeError:
        return True


def _sanitize_json_floats(value: Any) -> Any:
    """Replace non-finite floats with their text form, recursively.

    json.dumps' default allow_nan=True emits the tokens ``Infinity``/``NaN``,
    which are NOT valid JSON, so a strict peer (or any non-Python client) cannot
    parse the frame at all. A resident legitimately CAN hold such a value -- e.g.
    NEQUIX_CAPACITY_MULTIPLIER=inf mirrors the builder's strict float(), which is
    why _configuration_values_match handles NaN -- and status() echoes the
    configuration verbatim. Emitting the value as a string keeps the information
    and keeps the frame parseable; dropping to allow_nan=False instead would raise
    and lose the whole event.
    """

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        # repr() via float(): a numpy scalar is a float subclass whose own repr is
        # 'np.float64(-inf)' under numpy 2, which would make the wire text depend on
        # which backend produced the number. 'inf'/'-inf'/'nan' is stable.
        return repr(float(value))
    if isinstance(value, Mapping):
        # Keys too: json.dumps coerces a non-string key through the same float
        # writer, so a non-finite KEY raises under allow_nan=False even when every
        # value is clean.
        return {
            _sanitize_json_floats(key): _sanitize_json_floats(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_floats(item) for item in value]
    return value


def _serialize_event(event: Mapping[str, Any]) -> bytes:
    # allow_nan=False cannot raise here: json's encoder routes only ``float``
    # instances (np.float64 included, being a subclass) through the writer that
    # rejects non-finite values, and _sanitize_json_floats has replaced every one
    # of those -- keys as well as values. A type json cannot encode at all still
    # raises TypeError, which _EventSender.send swallows as it always has.
    payload = _sanitize_json_floats(dict(event))
    try:
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except UnicodeEncodeError:
        # Messages can carry surrogate-escaped bytes (any path decoded with
        # errors="surrogateescape"), which have no UTF-8 encoding. Escaping them
        # keeps the event valid JSON and lossless for the client's json.loads,
        # instead of raising out of the worker thread.
        return json.dumps(
            payload, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")



def _fit_event_text(
    event: Mapping[str, Any],
    key: str,
    text: str,
    *,
    suffix: str = "",
) -> str:
    """Return the longest character-safe prefix fitting one serialized event."""

    candidate_event = dict(event)

    # Every scan below is bounded to the largest prefix that can ever fit
    # (<= MAX_REQUEST_BYTES chars, since each char is at least one byte). This
    # keeps splitting a multi-megabyte line linear: _split_log_event calls this
    # once per chunk with the whole remaining tail, so an unbounded scan of that
    # tail would be O(N^2) and CPU-pin the job thread.
    scan_limit = MAX_REQUEST_BYTES

    # _serialize_event escapes the WHOLE event with ensure_ascii=True as soon as
    # ANY field carries a surrogate, which also lengthens other non-ASCII fields.
    # Size candidates with the same decision — derived from every field, not just
    # the one being fit — or a surrogate in a sibling field would make the real
    # send exceed the budget. A plain encode detects surrogates far more cheaply
    # than building a full json.dumps escaped copy.
    candidate_event[key] = ""
    ascii_escape = _text_has_surrogate(text[:scan_limit]) or any(
        isinstance(value, str) and _text_has_surrogate(value)
        for value in candidate_event.values()
    )

    def _event_size(value: str) -> int:
        candidate_event[key] = value
        return len(
            json.dumps(
                candidate_event, ensure_ascii=ascii_escape, separators=(",", ":")
            ).encode("utf-8")
        )

    # Skip the full-text serialization when the text alone already fills the
    # limit; it cannot fit once event overhead is added, so go straight to the
    # (clamped) binary search. Using ``<`` (not ``<=``) matters for the MAX-char
    # windows _split_log_event passes: an exactly-MAX-char text can never
    # early-return, so probing it would waste a full ~1 MB json.dumps per chunk.
    if len(text) < MAX_REQUEST_BYTES and _event_size(text) <= MAX_REQUEST_BYTES:
        return text

    # Only the value at ``key`` varies between probes, so size the surrounding
    # event once and each candidate on its own. Overhead uses the same
    # ensure_ascii as the send.
    overhead = _event_size("") - len(b'""')
    budget = MAX_REQUEST_BYTES - overhead

    low = 0
    # UTF-8 uses at least one byte per character, so no prefix longer than the
    # byte budget can ever fit. Clamping keeps the first probe from copying and
    # encoding half of a pathologically long line.
    high = min(len(text), max(budget, 0))
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle] + suffix
        encoded = len(
            json.dumps(candidate, ensure_ascii=ascii_escape).encode("utf-8")
        )
        if encoded <= budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _split_log_event(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split an oversized logical log line into protocol-safe log events."""

    full_text = str(event.get("line", ""))
    if not full_text:
        return [dict(event)]
    # Reserve space for continuation metadata while selecting every chunk.
    # ``false`` is one byte longer than ``true`` in JSON, so fitting against
    # the terminal form also guarantees that intermediate chunks fit.
    template = dict(event)
    template["continued"] = False
    total = len(full_text)
    chunks: list[dict[str, Any]] = []
    offset = 0
    while offset < total:
        # Advance by an integer offset and only ever hand _fit_event_text a
        # bounded window. Re-slicing the whole remaining tail each iteration
        # (remaining = remaining[len(chunk):]) copies O(N) bytes per chunk and
        # is O(N^2) for a multi-hundred-MB newline-free line. Any single chunk
        # is at most one byte per character, so it cannot exceed MAX_REQUEST_BYTES
        # characters; a MAX-character window therefore always contains it.
        window = full_text[offset : offset + MAX_REQUEST_BYTES]
        chunk = _fit_event_text(template, "line", window)
        if not chunk:
            break
        chunk_event = dict(template)
        chunk_event["line"] = chunk
        offset += len(chunk)
        chunk_event["continued"] = offset < total
        chunks.append(chunk_event)
    if offset < total:
        fallback = {"event": "log", "line": _TRUNCATION_MARKER.lstrip()}
        if len(_serialize_event(fallback)) <= MAX_REQUEST_BYTES:
            chunks.append(fallback)
    return chunks


def _truncate_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Bound a non-log event while retaining its terminal error contract."""

    bounded = dict(event)
    traceback_text = bounded.get("traceback")
    if isinstance(traceback_text, str):
        bounded["traceback"] = _TRUNCATION_MARKER.lstrip()

    error_text = bounded.get("error")
    if isinstance(error_text, str):
        # Preserve a useful exception summary without allowing it to consume
        # the entire event budget before the traceback is considered.
        error_budget = max(64, min(65536, MAX_REQUEST_BYTES // 4))
        # Slice the string to error_budget characters first: each encodes to at
        # least one byte, so the first error_budget bytes are unchanged while a
        # multi-megabyte error is never fully encoded just to keep ~64 KB.
        shortened_error = _encode_utf8_lenient(error_text[:error_budget])[
            :error_budget
        ].decode("utf-8", errors="ignore")
        if shortened_error != error_text:
            shortened_error += _TRUNCATION_MARKER
        bounded["error"] = shortened_error
        if len(_serialize_event(bounded)) > MAX_REQUEST_BYTES:
            bounded["error"] = _fit_event_text(
                bounded,
                "error",
                error_text,
                suffix=_TRUNCATION_MARKER,
            )

    if isinstance(traceback_text, str):
        bounded["traceback"] = _fit_event_text(
            bounded,
            "traceback",
            traceback_text,
            suffix=_TRUNCATION_MARKER,
        )

    if len(_serialize_event(bounded)) <= MAX_REQUEST_BYTES:
        return bounded
    return {
        "event": "error",
        "code": "protocol_error",
        "error": "Server event exceeded the VPMDK protocol size limit.",
    }


def _event_payloads(event: Mapping[str, Any]) -> list[bytes]:
    serialized = _serialize_event(event)
    if len(serialized) <= MAX_REQUEST_BYTES:
        return [serialized + b"\n"]
    if event.get("event") == "log" and isinstance(event.get("line"), str):
        bounded_events = _split_log_event(event)
    else:
        bounded_events = [_truncate_event(event)]
    return [_serialize_event(item) + b"\n" for item in bounded_events]


class _EventSender:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.lock = threading.Lock()
        self.connected = True

    def send(self, event: Mapping[str, Any]) -> bool:
        with self.lock:
            if not self.connected:
                return False
            try:
                # Serialization stays inside the guard: an encoding failure here
                # used to escape send(), escape _execute_job's finally and kill
                # the worker thread, leaving the server alive but unable to run
                # anything ever again. Any failure (socket or encoding) simply
                # marks the connection unusable.
                for payload in _event_payloads(event):
                    self.connection.sendall(payload)
                return True
            except Exception:
                self.connected = False
                return False

    def close(self) -> None:
        # Break any in-flight sendall *before* contending for the lock. A client
        # that stopped draining holds it for the whole send timeout, and this is
        # the only preemption path force-shutdown has; shutdown() makes the
        # blocked sendall fail immediately so the lock is released at once.
        self.connected = False
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        with self.lock:
            self.connected = False
            try:
                self.connection.close()
            except OSError:
                pass


class _ThreadScopedStdout(io.TextIOBase):
    """Route the worker thread's stdout to its job, everyone else's to the real one.

    ``contextlib.redirect_stdout`` swaps the PROCESS-GLOBAL ``sys.stdout``, so for
    the duration of a job every other thread's ``print`` was diverted into that
    job's client stream -- or dropped entirely once the client had gone (a
    timed-out ``run``), which is how an embedded caller using the documented
    ``VPMDKServer``/``serve_forever`` primitives silently lost its own output.
    Binding the redirect to the thread that owns the job keeps request-scoped
    output streaming to the client while leaving the rest of the process alone.
    """

    def __init__(self, target: "io.TextIOBase", owner_thread_id: int, fallback):
        self._target = target
        self._owner_thread_id = owner_thread_id
        self._fallback = fallback

    def _stream(self):
        if threading.get_ident() == self._owner_thread_id:
            return self._target
        return self._fallback

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        stream = self._stream()
        if stream is None:
            return len(text)
        return stream.write(text)

    def flush(self) -> None:
        stream = self._stream()
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.flush()


class _LineEventWriter(io.TextIOBase):
    """Turn request-scoped stdout/stderr writes into streamed ``log`` events.

    ``stream`` tags the event when the origin is not stdout ("stderr" for the
    warning half), so the client can put each line back on the SAME stream the
    one-shot CLI would have used. Older clients ignore the extra key and print
    to stdout, so the line is still delivered rather than lost.
    """

    def __init__(self, sender: _EventSender, *, stream: str | None = None):
        self.sender = sender
        self._stream = stream
        self._pending = ""

    def _event(self, line: str) -> dict:
        event = {"event": "log", "line": line}
        if self._stream is not None:
            event["stream"] = self._stream
        return event

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        self._pending += text
        # _pending never holds a newline at rest (it is only ever assigned the
        # segment after the last newline, or cleared by flush), so a newline-free
        # write cannot complete any line.
        if "\n" not in text:
            return len(text)
        # Split the buffer once. Looping with `"\n" in ...` and `split("\n", 1)`
        # rescans and copies the whole remaining string per line, so a single
        # write carrying many newlines (a big model summary/diagnostic block) is
        # O(N^2) and stalls the worker. The trailing element is the incomplete
        # final line, kept until the next newline or flush.
        parts = self._pending.split("\n")
        self._pending = parts[-1]
        for line in parts[:-1]:
            self.sender.send(self._event(line))
        return len(text)

    def flush(self) -> None:
        if self._pending:
            self.sender.send(self._event(self._pending))
            self._pending = ""


@dataclass
class _RunJob:
    workdir: str
    caller_cwd: str
    sender: _EventSender
    enqueued_at: float
    # The CLIENT's umask, applied around the calculation so output artifacts
    # get the modes the byte-identical one-shot run would create. None when an
    # older client omits it: the server's own umask then applies, as before.
    umask: "int | None" = None


class VPMDKServer:
    """One-process, one-calculator Unix-domain socket server."""

    def __init__(
        self,
        socket_path: str,
        calculator,
        backend_tags: Mapping[str, Any],
        *,
        backend_base_dir: str,
        idle_timeout: float = 0.0,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        pidfile: str | None = None,
        log_file: str | None = None,
        log_file_named: bool = False,
        executor: Callable[..., None] | None = None,
    ):
        if not math.isfinite(idle_timeout) or idle_timeout < 0:
            raise ValueError("idle timeout must be a finite non-negative number")
        if not math.isfinite(heartbeat_interval) or heartbeat_interval <= 0:
            raise ValueError("heartbeat interval must be a finite positive number")
        _validate_resident_backend_tags(backend_tags)
        self.socket_path = os.path.abspath(socket_path)
        self.calculator = calculator
        effective_backend_tags = dict(backend_tags)
        # A present-but-BLANK `DEVICE=` counts as absent here. It names no device
        # (a template's unset ${VPMDK_DEVICE}), and the builders resolve it to the
        # same device an omitted one selects, so the authoritative value is what
        # the loaded calculator reports -- not "". Treating the key's mere
        # presence as "explicit" skipped detection and made the resident advertise
        # device="" and reject every request naming its real device (exit 5).
        if not any(
            str(key).upper() == "DEVICE" and str(value).strip()
            for key, value in effective_backend_tags.items()
        ):
            detected_device = _detect_calculator_device(calculator)
            if detected_device is not None:
                effective_backend_tags["DEVICE"] = detected_device
        # Same read-from-the-loaded-calculator pattern for the graph
        # converter: only when no spelling of the tag was given explicitly
        # (an explicit tag stays authoritative), advertise the algorithm the
        # resident model actually carries so an explicit matching request is
        # not rejected exit 5 (SPEC 3.4). Only spellings the RESIDENT's
        # builder actually reads suppress detection: a stale FOREIGN spelling
        # (a CHGNET resident with a leftover MATRIS_GRAPH_CONVERTER from an
        # edited template) is ignored by the builder AND stripped as foreign
        # by backend_identity, so letting it suppress detection left the
        # resident advertising None and rejecting every explicit converter
        # request in both directions.
        resident_mlp = _canonical_mlp_identity(
            _root()._resolve_mlp_tag(
                {
                    str(key).upper(): value
                    for key, value in effective_backend_tags.items()
                }
            )
        )
        relevant_spellings = {
            "GRAPH_CONVERTER",
            "GRAPH_CONVERTER_ALGORITHM",
            f"{resident_mlp}_GRAPH_CONVERTER",
            f"{resident_mlp}_GRAPH_CONVERTER_ALGORITHM",
        }
        if not any(
            str(key).upper() in relevant_spellings and str(value).strip()
            for key, value in effective_backend_tags.items()
        ):
            detected_algorithm = _detect_calculator_graph_converter_algorithm(
                calculator
            )
            if detected_algorithm is not None:
                effective_backend_tags["GRAPH_CONVERTER_ALGORITHM"] = (
                    detected_algorithm
                )
        self.backend = backend_identity(
            effective_backend_tags, base_dir=backend_base_dir
        )
        self.idle_timeout = float(idle_timeout)
        self.heartbeat_interval = float(heartbeat_interval)
        self.pidfile = pidfile
        self.executor = executor

        self._queue: queue.Queue[_RunJob] = queue.Queue()
        # Reentrant so a signal delivered while the main thread owns the lock
        # can publish force-stop without deadlocking that same thread.
        self._enqueue_lock = threading.RLock()
        self._job_available = threading.Event()
        self._state_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._force_requested = threading.Event()
        self._worker_stop = threading.Event()
        # Plain flags set by the SIGINT/SIGTERM handler. The handler must be
        # async-signal-safe, so it only writes these (a plain attribute store is
        # atomic under the GIL) and NEVER calls Event.set(): re-entering an
        # Event's non-reentrant internal lock from a signal that interrupted the
        # main thread mid-set() on the same Event would self-deadlock. The main
        # thread translates these into the shutdown Events in _should_exit, which
        # the accept loop evaluates on its 0.2s cadence.
        self._stop_signal = False
        self._force_signal = False
        # Counts every SIGINT/SIGTERM delivery. The flags above are level-
        # triggered, so once force is published a further press changes nothing;
        # teardown needs to tell a REPEAT press apart (see _await_worker_exit).
        self._signal_deliveries = 0
        self._busy = False
        # Connections accepted but whose handler has not finished yet. Counted
        # so the idle-timeout check cannot tear the server down in the window
        # between accept() and the handler enqueueing its run / touching
        # _last_activity, which would drop a request at the timeout boundary.
        self._active_connections = 0
        self._current_workdir: str | None = None
        self._current_sender: _EventSender | None = None
        self._worker_abandoned = False
        self._jobs_completed = 0
        self._jobs_failed = 0
        self._started_at = time.monotonic()
        self._last_activity = self._started_at
        self._listener: socket.socket | None = None
        self._socket_inode: int | None = None
        self._pidfile_written = False
        self._worker: threading.Thread | None = None
        self._previous_signal_handlers: dict[int, Any] = {}
        self._connection_order = threading.Condition()
        self._next_accept_sequence = 0
        self._next_handler_sequence = 0

        # OWN the logger rather than registering it in logging's global manager.
        # logging.getLogger(f"vpmdk.server.{id(self)}") kept every server's logger
        # -- and the open FileHandler attached to it -- alive in
        # Logger.manager.loggerDict forever, which grows without bound in a
        # long-lived embedding process and never releases the log descriptor of an
        # instance that was constructed but never served (so _cleanup never ran).
        # Worse, CPython recycles addresses: a NEW server could be handed a DEAD
        # server's logger and inherit its handler, writing this server's request
        # paths and tracebacks into the previous server's log file. A private
        # Logger is unreachable from that registry, so it and its handler are
        # collected with the server, and the sequence number cannot collide.
        self.logger = logging.Logger(f"vpmdk.server.{next(_LOGGER_SEQUENCE)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler: logging.Handler
        if log_file:
            # Expand ~ so a foreground `--log-file '~/logs/s.log'` lands in $HOME
            # rather than a literal '~' directory under the current directory.
            log_file = _abspath_user(log_file)
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            # Create it 0600 first, so a foreground log is as private as the
            # daemon's (see _create_private_log_file). Refuse a symlink only
            # for a DERIVED <socket>.log the caller never actually named:
            # ``log_file_named`` is threaded from serve_cli (True whenever
            # --log-file was given), because keying on NAME EQUALITY alone
            # refused a user who deliberately spelled out the default name --
            # after paying the full model load -- while the identical
            # --daemon line accepted it (the daemon launcher keys its own
            # refuse_symlink on args.log_file is None). Library callers that
            # pass the derived path without setting log_file_named keep the
            # hardened refusal, as the planted-symlink test pins.
            _create_private_log_file(
                log_file,
                refuse_symlink=(not log_file_named)
                and log_file == _abspath_user(default_log_path(self.socket_path)),
            )
            # errors="backslashreplace": a workdir or model path can carry
            # surrogate-escaped bytes (non-UTF-8 filesystem names), and a strict
            # encoder raises inside Handler.emit -- logging swallows that via
            # handleError, so the record would be silently LOST from the log.
            # Escaping keeps the diagnostic instead of dropping it.
            handler = logging.FileHandler(
                log_file, encoding="utf-8", errors="backslashreplace"
            )
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)

    def status(self) -> dict[str, Any]:
        # Snapshot only the mutable, lock-guarded fields under the same enqueue->
        # state lock order the worker uses, so a snapshot cannot slip into the
        # window where the worker has dequeued the last job but not yet set
        # _busy (there the job is neither queued nor marked in-flight, and a
        # naive read reports a fully idle server). The worker holds _enqueue_lock
        # across that whole transition. Version metadata, pid, uptime, and the
        # immutable backend dict are assembled *outside* the locks so a status
        # poll cannot stall the worker from claiming its next job while an
        # uncached version read and the JSON build run.
        with self._enqueue_lock, self._state_lock:
            stopping = self._force_requested.is_set() or self._stop_requested.is_set()
            busy = self._busy
            jobs_completed = self._jobs_completed
            jobs_failed = self._jobs_failed
            queue_length = self._queue.qsize()
            current_workdir = self._current_workdir
        state = "stopping" if stopping else ("busy" if busy else "idle")
        response: dict[str, Any] = {
            "event": "status",
            "state": state,
            "backend": {
                "mlp": self.backend["mlp"],
                "model": self.backend["model"],
                "device": self.backend["device"],
                "options": {
                    key: value
                    for key, value in self.backend["configuration"].items()
                    if key not in {"MLP", "MODEL", "DEVICE"}
                },
            },
            "jobs_completed": jobs_completed,
            "jobs_failed": jobs_failed,
            "queue_length": queue_length,
            "uptime_s": round(time.monotonic() - self._started_at, 3),
            "pid": os.getpid(),
            "vpmdk_version": _package_version(),
            "protocol": PROTOCOL_VERSION,
        }
        if current_workdir is not None:
            response["current_workdir"] = current_workdir
        return response

    def request_stop(self, *, force: bool = False) -> None:
        with self._enqueue_lock:
            self._stop_requested.set()
            if force:
                self._publish_force_stop()

    def _publish_force_stop(self) -> None:
        """Publish force shutdown to the worker and serve loop immediately."""

        self._worker_stop.set()
        self._job_available.set()
        self._force_requested.set()

    def install_signal_handlers(self) -> None:
        """Request graceful shutdown on SIGINT/SIGTERM in the main thread."""

        if threading.current_thread() is not threading.main_thread():
            return

        def handle_signal(signum, _frame) -> None:
            # Async-signal-safe: set plain flags ONLY. A second signal escalates
            # to the equivalent of --force. Do NOT call Event.set() /
            # _publish_force_stop() / acquire any lock here:
            #  * Event.set() takes the Event's non-reentrant internal Condition
            #    lock. The main thread sets these same Events during shutdown
            #    (serve_forever's finally, _should_exit's force path). A signal
            #    delivered while the main thread is mid-set() would re-acquire the
            #    already-held lock on the same thread and self-deadlock, wedging
            #    the daemon before cleanup (stale socket/pidfile, held VRAM).
            #  * Taking _enqueue_lock would invert the enqueue->state lock order
            #    against the worker (the original AB-BA hazard).
            # The main thread drains these flags into the shutdown Events in
            # _should_exit (evaluated on the 0.2s accept-loop cadence), so a first
            # signal requests a graceful stop and a second escalates to force.
            if self._stop_signal:
                self._force_signal = True
            self._stop_signal = True
            self._signal_deliveries += 1

        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)

    def restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, handler in self._previous_signal_handlers.items():
            if handler is None:
                # signal.getsignal() returns None when the PREVIOUS handler was
                # not installed from Python (a C extension calling sigaction --
                # an MPI runtime, a JNI library). Replaying that None raises
                # TypeError, which escaped serve_cli's finally and reported a
                # fully clean shutdown as exit 1 -- and aborted the loop, leaving
                # the remaining signals bound to this dead server's handler. We
                # cannot reinstate a handler Python never owned; restore the
                # default so the host at least stops going through us.
                with contextlib.suppress(OSError, ValueError):
                    signal.signal(signum, signal.SIG_DFL)
                continue
            with contextlib.suppress(OSError, ValueError, TypeError):
                signal.signal(signum, handler)
        self._previous_signal_handlers.clear()

    def _bind(self) -> None:
        prepare_socket_path(
            self.socket_path, pidfile_expected=self.pidfile is not None
        )
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # Create the socket inode ALREADY private. bind() applies
            # 0777 & ~umask, and on Linux connect() to an AF_UNIX socket needs
            # only write permission on that inode -- so with a permissive umask
            # (002/000, common on HPC login nodes) the endpoint was briefly
            # connectable by other users in the two syscalls before the chmod
            # below, and the server never re-checks permissions once a connection
            # exists. The path-based chmod also follows symlinks, so in a parent
            # another user can write to, an unlink+symlink swap in that window
            # would apply 0600 to a file of their choosing. A umask makes the mode
            # correct at creation instead of after the fact; the chmod stays as
            # belt and braces (and to tighten a pre-existing permissive inode).
            # _bind runs before the worker thread starts, so this process-global
            # setting cannot leak into concurrent file creation.
            previous_umask = os.umask(0o177)
            try:
                listener.bind(self.socket_path)
            finally:
                os.umask(previous_umask)
            self._socket_inode = os.stat(self.socket_path).st_ino
            os.chmod(self.socket_path, 0o600)
            listener.listen(128)
            listener.settimeout(0.2)
        except Exception:
            listener.close()
            raise
        self._listener = listener
        if self.pidfile:
            _write_pidfile(self.pidfile, self.socket_path)
            self._pidfile_written = True

    def _read_request(self, connection: socket.socket) -> dict[str, Any]:
        # A socket timeout applies per recv, so it would reset on every byte and
        # let a peer that dribbles input hold this connection's accept-order
        # turn indefinitely, wedging every later status/stop. Bound the whole
        # read with one deadline instead.
        deadline = time.monotonic() + REQUEST_READ_TIMEOUT
        buffer = bytearray()
        try:
            while True:
                newline = buffer.find(b"\n")
                if newline >= 0:
                    break
                if len(buffer) > MAX_REQUEST_BYTES + 1:
                    raise ValueError("request exceeds size limit")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("request read timed out")
                connection.settimeout(remaining)
                chunk = connection.recv(65536)
                if not chunk:
                    break
                buffer.extend(chunk)

            newline = buffer.find(b"\n")
            payload = bytes(buffer if newline < 0 else buffer[:newline])
            if not payload:
                raise ValueError("empty request")
            if payload.endswith(b"\r"):
                payload = payload[:-1]
            if len(payload) > MAX_REQUEST_BYTES:
                raise ValueError("request exceeds size limit")
            try:
                request = json.loads(payload.decode("utf-8"))
            except RecursionError:
                # Client-supplied nesting depth, not a server defect: report it
                # as a protocol error so it is neither logged as an internal bug
                # nor able to flood the daemon log with tracebacks.
                raise ValueError("request JSON is nested too deeply") from None
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            return request
        finally:
            # The request deadline must never govern the event stream. A client
            # that stops reading briefly (a pager, a slow event callback) fills
            # the send buffer, and a 5 s sendall deadline would tear down the
            # connection of a calculation that actually succeeded.
            connection.settimeout(EVENT_SEND_TIMEOUT)

    def _wait_for_handler_turn(self, sequence: int) -> None:
        """Wait until every earlier accepted connection has been resolved."""

        with self._connection_order:
            self._connection_order.wait_for(
                lambda: sequence == self._next_handler_sequence
            )

    def _finish_handler_turn(self, sequence: int) -> None:
        """Publish completion of one accept-ordered handler turn."""

        with self._connection_order:
            if sequence != self._next_handler_sequence:
                raise RuntimeError("VPMDK connection sequence advanced out of order")
            self._next_handler_sequence += 1
            self._connection_order.notify_all()

    def _handle_connection(self, connection: socket.socket, sequence: int) -> None:
        sender: _EventSender | None = None
        handoff = False
        turn_acquired = False
        try:
            # Construct the sender INSIDE the try: _EventSender allocates a
            # threading.Lock(), which can raise MemoryError under pressure. If
            # that escaped the try, the finally would never run -- leaving this
            # connection's accept-order turn unfinished (wedging every later run
            # in _wait_for_handler_turn) and _active_connections never decremented
            # (so the idle test never holds and graceful stop / idle-timeout never
            # complete), leaving the server alive-looking but functionally dead.
            sender = _EventSender(connection)
            request = self._read_request(connection)
            version = request.get("version")
            # The version must be a real JSON integer. A bare `!=` comparison
            # would ACCEPT JSON `true` and `1.0` as protocol 1, because Python's
            # numeric tower makes True == 1 and 1.0 == 1 (and bool is an int
            # subclass) -- so a malformed or version-skewed peer would be treated
            # as speaking the supported protocol instead of getting the
            # protocol_error it is owed. Require a non-boolean int, matching the
            # strict isinstance checks the other request fields already use
            # (stop.force, run.workdir, run.caller_cwd).
            if (
                not isinstance(version, int)
                or isinstance(version, bool)
                or version != PROTOCOL_VERSION
            ):
                sender.send(
                    {
                        "event": "error",
                        "code": "protocol_error",
                        "error": f"Unsupported protocol version {version!r}; expected {PROTOCOL_VERSION}",
                    }
                )
                return
            op = request.get("op")
            # status/stop are answered immediately, WITHOUT waiting for this
            # connection's accept-order turn. The turn only exists to keep run
            # enqueues in accept order; gating status/stop on it lets an earlier
            # connection that is slow or silent in _read_request (up to
            # REQUEST_READ_TIMEOUT) stall a monitoring status or an operator stop,
            # violating SERVER_MODE_SPEC 3.2 ("respond immediately"). The finally
            # below still advances this connection's sequence in accept order, so
            # the run-queue ordering invariant is preserved.
            if op == "status":
                with self._state_lock:
                    self._last_activity = time.monotonic()
                sender.send(self.status())
                return
            if op == "stop":
                force = request.get("force", False)
                if not isinstance(force, bool):
                    sender.send(
                        {
                            "event": "error",
                            "code": "protocol_error",
                            "error": "stop.force must be a JSON boolean",
                        }
                    )
                    return
                # Holding the enqueue lock prevents new runs and shutdown
                # checks from crossing the acknowledgement boundary. Publish
                # the reply before exposing either graceful or force shutdown;
                # otherwise an idle CLI process can exit while this handler is
                # still trying to acknowledge the request.
                with self._enqueue_lock:
                    sender.send({"event": "done", "ok": True, "force": force})
                    self._stop_requested.set()
                    if force:
                        self._publish_force_stop()
                return
            if op != "run":
                sender.send(
                    {
                        "event": "error",
                        "code": "protocol_error",
                        "error": f"Unknown operation: {op!r}",
                    }
                )
                return
            # A run enqueues work, so it MUST observe accept order: wait for this
            # connection's turn before validating and enqueuing, so a later handler
            # can never enqueue its run before an earlier accepted run request.
            self._wait_for_handler_turn(sequence)
            turn_acquired = True
            workdir = request.get("workdir")
            if not isinstance(workdir, str) or not os.path.isabs(workdir):
                sender.send(
                    {
                        "event": "error",
                        "code": "protocol_error",
                        "error": "run.workdir must be an absolute path",
                    }
                )
                return
            caller_cwd = request.get("caller_cwd", workdir)
            if not isinstance(caller_cwd, str) or not os.path.isabs(caller_cwd):
                sender.send(
                    {
                        "event": "error",
                        "code": "protocol_error",
                        "error": "run.caller_cwd must be an absolute path",
                    }
                )
                return
            # Resolve BEFORE taking the lock. os.path.realpath walks the path one
            # component at a time with lstat, so on an autofs/NFS mount that must
            # be triggered (or whose server is unresponsive) it blocks for as long
            # as the filesystem takes. _enqueue_lock is the server's single global
            # gate -- status(), the stop handler, _should_exit() on every accept
            # iteration, and _worker_loop's next dequeue all take it -- so holding
            # it across that walk froze status/stop (which SERVER_MODE_SPEC 3.2
            # requires to answer even while busy), stalled the accept loop, left
            # the worker unable to start the next job on an idle GPU, and blocked
            # both shutdown paths, so an operator could not even stop the server.
            # These results only populate _RunJob, so hoisting them changes no
            # lock-protected state.
            request_umask = request.get("umask")
            if request_umask is not None and (
                isinstance(request_umask, bool)
                or not isinstance(request_umask, int)
                or not 0 <= request_umask <= 0o777
            ):
                # Same strictness as stop.force: wrong types are protocol
                # errors, never coerced.
                sender.send(
                    {
                        "event": "error",
                        "code": "protocol_error",
                        "error": "run.umask must be an integer between 0 and 511",
                    }
                )
                return
            resolved_workdir = os.path.realpath(workdir)
            resolved_caller_cwd = os.path.realpath(caller_cwd)
            with self._enqueue_lock:
                if self._stop_requested.is_set():
                    sender.send(
                        {
                            "event": "done",
                            "ok": False,
                            "code": "server_stopping",
                            "error": "Server is stopping and is not accepting new calculations.",
                        }
                    )
                    return
                with self._state_lock:
                    queue_position = self._queue.qsize() + (1 if self._busy else 0)
                sender.send({"event": "accepted", "queue_position": queue_position})
                self._queue.put(
                    _RunJob(
                        workdir=resolved_workdir,
                        caller_cwd=resolved_caller_cwd,
                        sender=sender,
                        enqueued_at=time.monotonic(),
                        umask=request_umask,
                    )
                )
                self._job_available.set()
            handoff = True
        except Exception as exc:
            # Deliberately broad: json.loads can raise beyond the protocol error
            # types (RecursionError on deeply nested input, MemoryError, ...),
            # and any escape would leave this connection's turn unfinished.
            if not turn_acquired:
                self._wait_for_handler_turn(sequence)
                turn_acquired = True
            if isinstance(
                exc, (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError)
            ):
                code = "protocol_error"
            else:
                # Not a malformed request: record it, otherwise a server-side
                # defect is indistinguishable from a client protocol violation.
                code = "internal_error"
                self.logger.exception(
                    "Unexpected failure while handling a connection"
                )
            error_event: dict[str, Any] = {
                "event": "error",
                "code": code,
                "error": str(exc),
            }
            if code == "internal_error":
                error_event["traceback"] = traceback_module.format_exc()
            if sender is not None:
                sender.send(error_event)
        finally:
            # The accept-ordered sequence must advance no matter how this
            # handler ended: a turn left unfinished blocks every later
            # connection forever in _wait_for_handler_turn.
            if not turn_acquired:
                self._wait_for_handler_turn(sequence)
                turn_acquired = True
            self._finish_handler_turn(sequence)
            if not handoff:
                if sender is not None:
                    sender.close()
                else:
                    # _EventSender construction failed; close the raw connection.
                    with contextlib.suppress(OSError):
                        connection.close()
            # Clear the in-flight mark. A handed-off run keeps the server busy
            # via the queue/_busy flag, so it is safe to drop the count here.
            with self._state_lock:
                self._active_connections -= 1

    def _heartbeat(self, job: _RunJob, started_at: float, finished: threading.Event) -> None:
        while not finished.wait(self.heartbeat_interval):
            job.sender.send(
                {
                    "event": "heartbeat",
                    "elapsed_s": round(time.monotonic() - started_at, 3),
                }
            )

    @staticmethod
    def _recover_cuda_oom(exc: BaseException) -> None:
        description = f"{type(exc).__name__}: {exc}".lower()
        if "cuda" not in description or "out of memory" not in description:
            return
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _execute_job(self, job: _RunJob) -> None:
        started_at = time.monotonic()
        # Everything the except/finally below may touch is initialized to a safe
        # default BEFORE the try; the actual (allocating) setup happens INSIDE it.
        # A MemoryError from threading.Event()/Thread()/_LineEventWriter
        # allocation must not escape this method's finally -- that would kill the
        # sole worker with _busy still True and wedge every later request (the
        # same Lock/Event-allocation hazard guarded for _EventSender in
        # _handle_connection). Caught here, it is reported and the worker survives.
        heartbeat_finished: threading.Event | None = None
        heartbeat: threading.Thread | None = None
        heartbeat_started = False
        writer: _LineEventWriter | None = None
        stderr_writer: _LineEventWriter | None = None
        previous_umask: "int | None" = None
        rng_state = None
        ok = False
        # None (not a dict literal) so NOTHING before the try allocates: a
        # MemoryError here must not skip the finally (_busy reset + terminal event
        # + sender close). The real allocations -- the RNG snapshot and this
        # default terminal event -- happen INSIDE the try below.
        terminal_event: "dict[str, Any] | None" = None
        try:
            rng_state = _np.random.get_state() if _np is not None else None
            terminal_event = {
                "event": "done",
                "ok": False,
                "code": "internal_error",
                "error": "VPMDK failed to start the calculation.",
                "elapsed_s": 0.0,
            }
            if job.umask is not None:
                # Output artifacts must get the modes the CLIENT's umask
                # dictates (one-shot parity): a `umask 077; vpmdk run` client
                # got world-readable OUTCAR/CONTCAR from a umask-022 resident,
                # and the reverse broke group post-processing pipelines with
                # 0600 files after exit 0. Process-global like the cwd this
                # job also swaps (_working_directory); the worker is the only
                # calculation thread, and the finally below restores it.
                previous_umask = os.umask(job.umask)
            heartbeat_finished = threading.Event()
            heartbeat = threading.Thread(
                target=self._heartbeat,
                args=(job, started_at, heartbeat_finished),
                daemon=True,
            )
            # Heartbeats are a keep-alive nicety; the calculation runs in this
            # worker thread and needs no extra thread. Start best-effort and only
            # join a thread that actually started.
            try:
                heartbeat.start()
                heartbeat_started = True
            except Exception:
                self.logger.warning(
                    "Heartbeat thread could not start for %s; continuing without "
                    "heartbeats",
                    job.workdir,
                )
            writer = _LineEventWriter(job.sender)
            stderr_writer = _LineEventWriter(job.sender, stream="stderr")
            try:
                # TORCH_WARN_ONCE dedups in C++ BELOW the Python warnings layer,
                # so the per-job catch_warnings() below cannot re-arm it: a
                # torch-originated warning (e.g. CHGNet's requires_grad-to-scalar
                # UserWarning on every forward pass) reached the client on the
                # FIRST job of a resident only. warnAlways forwards every
                # occurrence to the Python layer, where the per-job filter scope
                # restores exactly once-per-job -- one-shot parity, measured with
                # no intra-job over-emission.
                import torch as _torch

                _torch.set_warn_always(True)
            except Exception:
                pass
            bcar_path = os.path.join(job.workdir, "BCAR")
            # Thread-scoped: only THIS worker thread's stdout becomes client log
            # events; concurrent threads keep writing to the real stdout.
            with contextlib.redirect_stdout(
                _ThreadScopedStdout(writer, threading.get_ident(), sys.stdout)
            ), contextlib.redirect_stderr(
                # The stderr HALF of the same 3.3 relay: third-party warnings
                # (ASE FutureWarning, pymatgen BadPoscarWarning, numpy
                # RuntimeWarning) go to stderr, which was never redirected --
                # the submitting client saw nothing while the byte-identical
                # one-shot run printed them, and they landed in the server's
                # private 0600 log instead.
                _ThreadScopedStdout(
                    stderr_writer, threading.get_ident(), sys.stderr
                )
            ), warnings.catch_warnings():
                # catch_warnings() bumps the warnings filter version, which
                # invalidates every module's __warningregistry__ dedup cache:
                # without it a resident emitted each once-per-process warning
                # for the FIRST job only, and jobs 2..N lost it from client
                # AND log, where one-shot (a fresh process per run) re-emits
                # it on every run.
                # Validate INSIDE the writer redirect so any warning it emits
                # reaches the client, exactly as one-shot mode shows it. e.g.
                # MATTERSIM_STRESS_WEIGHT=not-a-number makes _parse_optional_float
                # warn-and-ignore during canonicalization; run before the redirect,
                # that warning went only to the foreground terminal / daemon log.
                try:
                    _root()._reject_broken_input_link(bcar_path, "BCAR")
                    request_tags = (
                        # warn_unknown_tags=False: this parse is hoisted ahead
                        # of run_workdir, and warning here put the line before
                        # the Note: lines and emitted it even when one-shot
                        # fails at the INCAR before ever reading BCAR;
                        # run_workdir warns for the passed tags at the
                        # one-shot position instead.
                        _root().parse_key_value_file(
                            bcar_path, warn_unknown_tags=False
                        )
                        if os.path.exists(bcar_path)
                        else {}
                    )
                except Exception as exc:
                    # One-shot prints the unused-input notices as run_workdir's
                    # FIRST output, BEFORE reading BCAR; this hoisted parse
                    # (needed early because validate_request_backend consumes
                    # request_tags) skipped them on the failure path, so a
                    # failing request lost the Note: lines the byte-identical
                    # one-shot prints -- SPEC 3.3's own log-event example is
                    # exactly the KPOINTS line. Inside the redirect, so the
                    # notices stream to the client before the diagnostic, in
                    # one-shot's order. A malformed/unreadable request BCAR
                    # (bad encoding, a directory, permission error) is invalid
                    # user input: input_error (exit 1), matching one-shot.
                    _root()._print_unused_input_notices(job.workdir)
                    raise _root().WorkdirInputError(
                        f"Failed to read BCAR: {exc}"
                    ) from exc
                validate_request_backend(
                    self.backend,
                    request_tags,
                    request_base_dir=job.workdir,
                )
                device_warning = _grace_ignored_device_request_warning(
                    self.backend, request_tags
                )
                if device_warning:
                    # Before the unknown-model warning: the one-shot builder
                    # prints the DEVICE line first.
                    print(device_warning)
                backend_warning = _unknown_grace_request_warning(
                    self.backend, request_tags, base_dir=job.workdir
                )
                if backend_warning:
                    print(backend_warning)
                if self.executor is None:
                    _root().run_workdir(
                        job.workdir,
                        calculator=self.calculator,
                        bcar_tags=request_tags,
                        charge_base_dir=job.caller_cwd,
                        # The RESIDENT is what actually computes, and §3.4 lets a
                        # request inherit its backend tags instead of restating
                        # them (the documented batch pattern). Without this, the
                        # capability gate resolved an inheriting request against
                        # BackendConfig's CHGNET default: an energy-only resident
                        # answered exit 1 to a request that spelled the tags out
                        # but exit 2 -- documented RETRYABLE -- after a full
                        # forward pass to the byte-identical inheriting one, and
                        # the missing-stress warning appeared for one and not the
                        # other (a §1.2 divergence).
                        backend_tags=_resident_backend_tags(self.backend),
                    )
                else:
                    self.executor(job.workdir, calculator=self.calculator)
            writer.flush()
            stderr_writer.flush()
            ok = True
            terminal_event = {
                "event": "done",
                "ok": True,
                "elapsed_s": round(time.monotonic() - started_at, 3),
            }
        except BaseException as exc:
            if writer is not None:
                writer.flush()
            if stderr_writer is not None:
                stderr_writer.flush()
            self._recover_cuda_oom(exc)
            traceback_text = traceback_module.format_exc()
            if isinstance(exc, BackendConfigurationMismatch):
                code = "backend_mismatch"
            elif isinstance(
                exc, (_root().WorkdirInputError, _root().UnsupportedInputError)
            ):
                # VPMDK raises UnsupportedInputError for "this requested INCAR
                # config/mode is not supported" (VTST ICHAIN!=0, unsupported
                # NFREE, ...): a fix-your-input condition, not a retryable
                # calculation failure, so classify it as input_error (exit 1).
                # Matching the BUILTIN NotImplementedError here instead would also
                # capture one raised MID-CALCULATION by a third-party backend
                # (torch's "Could not run 'aten::...' with arguments from the
                # 'CUDA' backend" for an unregistered kernel), which
                # SERVER_MODE_SPEC 2.5 defines as exit 2 -- and the exit-1 branch
                # additionally suppresses the traceback the user needs.
                # ASE's PropertyNotImplementedError (a NotImplementedError
                # subclass) is likewise NOT input: it is raised mid-calculation by a
                # calculator lacking a requested property and is a genuine
                # calculation failure (exit 2, with traceback), not bad input.
                code = "input_error"
            elif isinstance(
                exc,
                (
                    PermissionError,
                    IsADirectoryError,
                    NotADirectoryError,
                    # A dangling symlink at an artifact path pointing into a
                    # missing directory: open("w") raises FileNotFoundError,
                    # a permanent property of the submitted tree.
                    FileNotFoundError,
                ),
            ) or (
                isinstance(exc, OSError)
                # errnos WITHOUT a dedicated Python subclass (lesson xxxix):
                # a read-only mount, a self-referential symlink loop at an
                # artifact path, an over-long symlink target. All are
                # deterministic properties of the submitted workdir.
                and exc.errno in (errno.EROFS, errno.ELOOP, errno.ENAMETOOLONG)
            ):
                # EROFS spelled out: Python has no dedicated OSError subclass
                # for a read-only FILESYSTEM (a read-only NFS export, `mount -o
                # remount,ro`, a container image layer), so "a read-only tree"
                # -- which the comment below has promised exit 1 for since
                # R135 -- only actually matched when read-only-ness came from
                # permission bits (EACCES -> PermissionError), not from the
                # mount itself.
                # Writing outputs into the client's workdir failed because of the
                # workdir itself (an OUTCAR directory in the way, a read-only
                # tree). These are deterministic properties of the submitted
                # workdir -- retrying reproduces them byte-for-byte -- so
                # advertising exit 2 ("retryable calculation failure",
                # SERVER_MODE_SPEC 2.5) sends batch drivers into a retry loop
                # that can never succeed. The one-shot CLI dies with an
                # uncaught-OSError exit 1 for the same tree, so exit 1 here also
                # keeps the two paths in agreement. Other OSErrors (ENOSPC,
                # network filesystems flapping) stay exit 2: those genuinely can
                # clear up between attempts.
                code = "input_error"
            else:
                code = "calculation_error"
            terminal_event = {
                "event": "done",
                "ok": False,
                "code": code,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback_text,
                "elapsed_s": round(time.monotonic() - started_at, 3),
            }
            self.logger.error("Calculation failed for %s\n%s", job.workdir, traceback_text)
        finally:
            if previous_umask is not None:
                with contextlib.suppress(Exception):
                    os.umask(previous_umask)
            # Restore the process-global numpy RNG so this request's stochastic
            # draws (MD initial velocities, Langevin noise) do not leak into the
            # next one -- keeping a repeated identical request (A->B->A) and the
            # one-shot result reproducible from the resident's startup RNG state.
            if rng_state is not None and _np is not None:
                with contextlib.suppress(Exception):
                    _np.random.set_state(rng_state)
            if heartbeat_finished is not None:
                heartbeat_finished.set()
            if heartbeat_started and heartbeat is not None:
                heartbeat.join(timeout=0.2)
            with self._state_lock:
                if ok:
                    self._jobs_completed += 1
                else:
                    self._jobs_failed += 1
                self._busy = False
                self._current_workdir = None
                self._last_activity = time.monotonic()
            # Keep _current_sender published until the terminal event is out:
            # clearing it first would leave a concurrent force-stop with no
            # sender to close, so it could not preempt this final send.
            try:
                # terminal_event is None only if even its default dict allocation
                # failed (extreme OOM); still close the sender so the client's
                # recv() ends (ServerConnectionError -> exit 3) rather than
                # blocking forever, and _busy has already been reset above.
                if terminal_event is not None:
                    job.sender.send(terminal_event)
                job.sender.close()
            finally:
                with self._state_lock:
                    if self._current_sender is job.sender:
                        self._current_sender = None

    def _worker_loop(self) -> None:
        while not (self._worker_stop.is_set() or self._force_requested.is_set()):
            job: _RunJob | None = None
            with self._enqueue_lock:
                # Force stop publishes _worker_stop while holding this same
                # lock. Recheck after acquiring it so an iteration that began
                # just before publication cannot claim one more queued job.
                if self._worker_stop.is_set() or self._force_requested.is_set():
                    break
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    self._job_available.clear()
                else:
                    # A force request can be published by an embedded caller
                    # or signal test hook while Queue.get_nowait() is in
                    # progress. Never turn that just-dequeued queued job into
                    # active work after force-stop became observable.
                    if self._worker_stop.is_set() or self._force_requested.is_set():
                        # Suppress per job, like _reject_pending_jobs: a peer that
                        # already vanished must not raise out of the worker loop
                        # here, which would skip task_done() and end the loop
                        # through the catch-all instead of stopping cleanly.
                        with contextlib.suppress(Exception):
                            job.sender.send(
                                {
                                    "event": "done",
                                    "ok": False,
                                    "code": "server_stopping",
                                    "error": (
                                        "Server was force-stopped before this "
                                        "calculation started."
                                    ),
                                }
                            )
                        with contextlib.suppress(Exception):
                            job.sender.close()
                        self._queue.task_done()
                        job = None
                        break
                    if self._queue.empty():
                        self._job_available.clear()
                    with self._state_lock:
                        self._busy = True
                        self._current_workdir = job.workdir
                        self._current_sender = job.sender
            if job is None:
                self._job_available.wait(0.2)
                continue
            try:
                self._execute_job(job)
            except BaseException:
                # _execute_job's own finally already reset _busy and delivered a
                # terminal event; a BaseException still escaping here (e.g. a
                # MemoryError raised while formatting the error/traceback inside
                # its except handler) must NOT kill the sole worker and wedge the
                # server permanently. Recover and keep serving. The logging is
                # itself allocation-guarded so an OOM cannot re-escape here.
                with contextlib.suppress(Exception):
                    self.logger.exception(
                        "Worker recovered from an unexpected _execute_job failure"
                    )
            finally:
                self._queue.task_done()

    def _should_exit(self) -> bool:
        # Translate pending signal-handler flags into the shutdown Events here,
        # on the main thread. The SIGINT/SIGTERM handler only sets these plain
        # flags (calling Event.set() from the handler is not re-entrant-safe); the
        # accept loop evaluates _should_exit on its 0.2s cadence, so a first
        # signal becomes a graceful stop and a second escalates to force.
        if self._stop_signal and not self._stop_requested.is_set():
            self._stop_requested.set()
        if self._force_signal and not self._force_requested.is_set():
            # Publish the signal-triggered force state UNDER _enqueue_lock, like
            # request_stop(force=True). Without the lock, the worker could hold
            # _enqueue_lock, pass its post-dequeue force recheck (_worker_loop),
            # and mark a queued job busy while this main thread sets
            # _force_requested outside the lock -- letting a long calculation
            # start after forced teardown was requested and delaying shutdown.
            with self._enqueue_lock:
                self._publish_force_stop()
        if self._force_requested.is_set():
            # Serialize with job claiming so a queued job cannot become active
            # after force shutdown has been published by a signal or embedded
            # caller.
            with self._enqueue_lock:
                self._worker_stop.set()
                self._job_available.set()
            return True
        with self._enqueue_lock:
            with self._state_lock:
                idle = (
                    not self._busy
                    and self._active_connections == 0
                    and self._queue.empty()
                    # _execute_job publishes _busy=False and refreshes
                    # _last_activity BEFORE delivering the terminal event, so
                    # without this the server counted as idle while the worker was
                    # still parked in sendall (up to EVENT_SEND_TIMEOUT=900s) for a
                    # client that stopped draining. The accept loop then exited and
                    # _close_listener() ran, so nothing accepted while the process
                    # stayed alive holding the model: status/stop returned exit 3/4,
                    # `stop --force` -- the one designed preemption for a blocked
                    # send -- became unreachable, and the live server looked stale
                    # to the next `serve`, which bound over its socket.
                    # _current_sender is deliberately kept published for exactly
                    # this window (see _execute_job), so it is the precise signal
                    # that a job is not finalized yet.
                    and self._current_sender is None
                )
                last_activity = self._last_activity
            if self._stop_requested.is_set() and idle:
                return True
            if (
                self.idle_timeout > 0
                and idle
                and time.monotonic() - last_activity >= self.idle_timeout
            ):
                self.logger.info("Idle timeout reached; stopping server")
                self._stop_requested.set()
                return True
        return False

    def _reject_pending_jobs(self, reason: str) -> None:
        """Terminate every queued job no worker will ever run.

        A queued _RunJob OWNS its client's accepted socket (_handle_connection
        hands ownership over and skips sender.close() once handoff is True), so a
        job left in the queue means a client blocked forever on recv -- it gets no
        terminal event and no EOF -- plus a leaked descriptor for the lifetime of
        the server object. Give each one the documented ``server_stopping``
        terminal event and close it. Failures are suppressed per job: a peer that
        already vanished must not abort teardown of the rest.

        Shared by the force path and serve_forever's teardown so the two cannot
        drift; the drain is idempotent, so it is a no-op on the graceful path
        (which only exits with a provably empty queue). _enqueue_lock is an RLock,
        so nesting inside _disconnect_forced_jobs is safe.
        """

        with self._enqueue_lock:
            while True:
                try:
                    queued_job = self._queue.get_nowait()
                except queue.Empty:
                    break
                with contextlib.suppress(Exception):
                    queued_job.sender.send(
                        {
                            "event": "done",
                            "ok": False,
                            "code": "server_stopping",
                            "error": reason,
                        }
                    )
                with contextlib.suppress(Exception):
                    queued_job.sender.close()
                self._queue.task_done()

    def _disconnect_forced_jobs(self) -> None:
        """Disconnect active work and reject queued jobs before force join."""

        with self._enqueue_lock:
            with self._state_lock:
                current_sender = self._current_sender
            if current_sender is not None:
                # Suppressed for the same reason as the queued jobs below: a peer
                # that already vanished must not abort force teardown.
                with contextlib.suppress(Exception):
                    current_sender.close()
            self._reject_pending_jobs(
                "Server was force-stopped before this calculation started."
            )

    def _close_listener(self) -> None:
        """Stop accepting new connections. Idempotent.

        Split out of _cleanup so teardown can stop listening BEFORE joining the
        worker. The socket FILE is deliberately not unlinked here: a positive-
        timeout `stop` client treats socket disappearance as shutdown completion,
        so that must still wait for the executor to return in _cleanup.
        """

        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

    def _await_worker_exit(self) -> None:
        """Wait for the worker, staying responsive to a repeated stop signal.

        An unconditional join() made the server DEAF once the accept loop had
        exited: _should_exit is the only place that drains the signal flags, and
        it is never called again, so a further SIGINT/SIGTERM was swallowed by our
        handler (which deliberately neither raises nor chains to the saved
        default). Meanwhile the listener is already closed, so `status`/`stop`
        get ECONNREFUSED. With a long job in flight the operator's only recourse
        was SIGKILL -- which skips _cleanup entirely and leaves the socket file
        and pidfile behind, exactly what SERVER_MODE_SPEC's shutdown checklist
        forbids, with the model still holding VRAM.

        So poll the join and watch for a NEW delivery, completing the escalation
        ladder: first signal = graceful, second = force, third = stop waiting for
        the in-flight executor. Abandoning the wait is safe here because the
        worker thread is a daemon: teardown continues, the socket and pidfile ARE
        removed, and the process exits promptly instead of hanging.
        """

        worker = self._worker
        if worker is None:
            return
        is_alive = getattr(worker, "is_alive", None)
        if not callable(is_alive):
            # A thread stand-in without is_alive (the threading.Thread test seam):
            # fall back to the plain blocking join rather than breaking teardown.
            worker.join()
            return
        deliveries_at_entry = self._signal_deliveries
        while True:
            worker.join(timeout=0.2)
            if not is_alive():
                return
            if self._signal_deliveries > deliveries_at_entry:
                self.logger.warning(
                    "Repeated shutdown signal: abandoning the wait for the "
                    "in-flight calculation and completing teardown."
                )
                # The worker is a daemon thread we are deliberately leaving inside
                # native backend code. Interpreter finalization would then run C++
                # destructors under it and abort the process, so record this so the
                # caller can leave without finalizing (see serve_cli.finish).
                self._worker_abandoned = True
                return

    def _cleanup(self) -> None:
        self._close_listener()
        # Remove the owned pidfile BEFORE unlinking the socket: a positive-timeout
        # client treats socket disappearance as shutdown completion, so the
        # pidfile must already be gone. Otherwise a restart that races socket
        # removal could observe the stale (now-dead) pidfile and abort with
        # ServerAlreadyRunning. _remove_owned_pidfile resolves the socket path
        # lexically, so it does not depend on the socket file still existing.
        if self.pidfile and self._pidfile_written:
            _remove_owned_pidfile(self.pidfile, self.socket_path, os.getpid())
        try:
            if (
                self._socket_inode is not None
                and os.path.lexists(self.socket_path)
                and os.stat(self.socket_path).st_ino == self._socket_inode
            ):
                os.unlink(self.socket_path)
        except OSError:
            pass
        for handler in list(self.logger.handlers):
            # Guarded per handler, like every other step in this function. A
            # buffered log write that only fails at flush time (ENOSPC/EDQUOT on
            # a quota-limited log filesystem, or EPIPE on a foreground server
            # whose stderr consumer exited) would otherwise escape _cleanup and
            # serve_forever's finally, turning an already-complete shutdown --
            # pidfile removed, socket unlinked -- into exit 1, and leaving the
            # remaining handlers attached and their descriptors open.
            with contextlib.suppress(Exception):
                handler.flush()
            with contextlib.suppress(Exception):
                handler.close()
            with contextlib.suppress(Exception):
                self.logger.removeHandler(handler)

    def serve_forever(self, *, ready_callback: Callable[[], None] | None = None) -> None:
        """Bind after calculator construction, serve requests, and clean up."""
        try:
            self._bind()
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            # Publish _worker only after a successful start(): if start() raises
            # (thread exhaustion), leaving _worker None keeps the finally from
            # calling join() on an unstarted thread (which raises "cannot join
            # thread before it is started" and would mask the failure and skip
            # socket/pidfile cleanup).
            worker.start()
            self._worker = worker
            self.logger.info(
                "VPMDK server ready at %s (MLP=%s, MODEL=%s, DEVICE=%s)",
                self.socket_path,
                self.backend["mlp"],
                self.backend["model"],
                self.backend["device"],
            )
            if ready_callback is not None:
                ready_callback()

            while not self._should_exit():
                try:
                    assert self._listener is not None
                    connection, _ = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if self._should_exit():
                        break
                    if exc.errno in _TRANSIENT_ACCEPT_ERRNOS:
                        # Recoverable (fd exhaustion, interrupted syscall, peer
                        # aborted): log and keep serving. Tearing the server down
                        # here would drop the VRAM-resident model and every queued
                        # job over a condition that clears as soon as fds free up.
                        self.logger.warning(
                            "accept() failed transiently (%s); retrying",
                            exc,
                        )
                        time.sleep(_ACCEPT_RETRY_BACKOFF_S)
                        continue
                    raise
                # Assign, but do not yet consume, the accept-order sequence:
                # only a started handler advances _next_handler_sequence via its
                # finally. Mark the connection in-flight before the next
                # _should_exit() check so the idle timeout cannot fire while the
                # handler is still starting up.
                sequence = self._next_accept_sequence
                with self._state_lock:
                    self._active_connections += 1
                try:
                    threading.Thread(
                        target=self._handle_connection,
                        args=(connection, sequence),
                        daemon=True,
                    ).start()
                except Exception:
                    # Thread creation failed (e.g. resource exhaustion): no
                    # handler will run to finish this turn or clear the in-flight
                    # mark, so undo both here and DO NOT consume the sequence
                    # (leaving _next_handler_sequence stalled would deadlock every
                    # later connection). Close the connection and keep serving.
                    with self._state_lock:
                        self._active_connections -= 1
                    with contextlib.suppress(OSError):
                        connection.close()
                    self.logger.exception(
                        "Failed to start a connection handler thread"
                    )
                    continue
                self._next_accept_sequence += 1
        finally:
            # Mark the server as stopping FIRST. On an ABNORMAL exit from the
            # accept loop (a BaseException such as KeyboardInterrupt out of
            # accept(), or a non-transient accept errno) _stop_requested was never
            # set, so a handler thread still inside _read_request would sail past
            # the run guard, answer {"event":"accepted"} and enqueue onto a queue
            # whose worker is about to exit -- hanging that client forever.
            self._stop_requested.set()
            self._worker_stop.set()
            self._job_available.set()
            if self._force_requested.is_set():
                self._disconnect_forced_jobs()
            # Stop LISTENING before the join. The accept loop has already exited,
            # so nothing will ever accept again -- but the socket stayed bound
            # until _cleanup, which runs only AFTER the join. On a force stop that
            # join waits for the whole in-flight calculation, so a client
            # connecting in that window landed in the kernel backlog, had its
            # request accepted by sendall, and then blocked in recv with no
            # `accepted` event and no terminal event for the rest of the job
            # (`run --timeout 0` waits forever). Closing the listener now makes
            # those connects fail fast as unreachable (exit 3) instead. The socket
            # FILE is still unlinked later in _cleanup, so a positive-timeout
            # `stop` client cannot mistake this for completed shutdown.
            self._close_listener()
            if self._worker is not None:
                # Python threads and in-flight GPU kernels cannot be cancelled
                # safely. Do not remove the socket or report embedded teardown
                # until the active executor has actually returned -- unless the
                # operator asks again (see _await_worker_exit).
                self._await_worker_exit()
            # The worker has exited, so anything still queued will NEVER run. Reject
            # it now (after the join, so the queue is stable and a job cannot be
            # stolen from a worker mid-dequeue) instead of stranding those clients
            # blocked on recv with their sockets held open. No-op on the graceful
            # path (queue provably empty) and on the force path (already drained).
            self._reject_pending_jobs(
                "Server stopped before this calculation started."
            )
            self._cleanup()


def _load_backend_for_server(workdir: str, bcar_path: str | None):
    root = _root()
    # Expand ~ before absolutizing so a quoted `--dir '~/calc'` (literal ~) is
    # resolved to $HOME/calc, consistently in the foreground and the daemon.
    workdir_abs = _abspath_user(workdir)
    selected_bcar = (
        _abspath_user(bcar_path)
        if bcar_path
        else os.path.join(workdir_abs, "BCAR")
    )
    root._reject_broken_input_link(selected_bcar, "BCAR")
    if os.path.exists(selected_bcar):
        tags = root.parse_key_value_file(selected_bcar)
        base_dir = os.path.dirname(selected_bcar)
    else:
        if bcar_path:
            raise FileNotFoundError(f"BCAR not found: {selected_bcar}")
        tags = {}
        base_dir = workdir_abs
        print("Warning: BCAR not found; starting server with default MLP=CHGNET.")

    mlp = _validate_resident_backend_tags(tags)
    model_reference = _validate_startup_model_path(
        tags, base_dir=base_dir, mlp=mlp
    )
    if (
        model_reference.explicit
        and model_reference.value is not None
        and model_reference.kind is root.ModelReferenceKind.LOCAL_PATH
    ):
        # Canonicalize only a local checkpoint path (relative -> absolute) for the
        # builder. A named model must keep the user's original spelling: a
        # resolver that silently substitutes an unknown name for the installed
        # default (GRACE) would otherwise hand the builder a known name and
        # suppress the documented "Unknown ... using default" warning.
        tags["MODEL"] = str(model_reference.value)

    structure = None
    poscar_path = os.path.join(workdir_abs, "POSCAR")
    potcar_path = os.path.join(workdir_abs, "POTCAR")
    if mlp != "DEEPMD" and os.path.exists(poscar_path):
        structure = root.read_structure(
            poscar_path,
            potcar_path if os.path.exists(potcar_path) else None,
        )
    # DeepMD's one-shot builder may infer a type map from its calculation
    # structure. A resident calculator cannot safely reuse such an inferred
    # ordering for unrelated requests, so server mode requires the explicit
    # map above and deliberately withholds the startup structure.
    with root._working_directory(base_dir):
        calculator = root._build_calculator_from_tags(tags, structure=structure)
    return calculator, tags, base_dir


# select.select() raises OverflowError past the platform time_t range (~9.2e9 s
# once CPython converts to int64 nanoseconds), so bound the readiness wait well
# below it. 1e9 s (~31 years) matches the client's _MAX_REQUEST_TIMEOUT.
_MAX_DAEMON_START_TIMEOUT = 1e9


def _daemon_start_timeout() -> float:
    """Return how long the parent waits for the daemon's readiness report."""

    raw = os.environ.get("VPMDK_DAEMON_START_TIMEOUT")
    if raw is None:
        return 600.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 600.0
    if not math.isfinite(value) or value <= 0:
        return 600.0
    # Cap it. select.select() converts its timeout to int64 nanoseconds, so a
    # value past ~9.2e9 s raises OverflowError -- an ArithmeticError, which
    # serve_cli's `except OSError` around _daemonize does NOT catch. The launcher
    # would die with a raw traceback AFTER the fork, leaving the grandchild to
    # execv into a resident server holding VRAM that the user believes failed to
    # start. Mirrors the client's own _MAX_REQUEST_TIMEOUT bound; anything above
    # the cap is "effectively forever" anyway.
    return min(value, _MAX_DAEMON_START_TIMEOUT)


def _move_fd_above_stdio(fd: int) -> int:
    """Return ``fd`` relocated above the standard descriptors when needed."""

    if fd > 2:
        return fd
    # Imported lazily: this module must stay importable on non-POSIX platforms,
    # where serve_cli reports the unsupported-platform error at runtime.
    import fcntl

    relocated = fcntl.fcntl(fd, fcntl.F_DUPFD, 3)
    os.close(fd)
    return relocated


def _reap_with_deadline(pid: int, deadline: float) -> None:
    """Reap ``pid`` without blocking past ``deadline``.

    A stalled intermediate child is left for init rather than hanging the CLI.
    """

    while True:
        try:
            reaped, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if reaped:
            return
        if time.monotonic() >= deadline:
            return
        time.sleep(0.01)


def _abspath_user(path: str) -> str:
    """Absolutize a user-supplied path, expanding ``~`` first.

    The daemon re-exec absolutizes every path because the daemon chdirs to ``/``.
    ``os.path.abspath`` alone would turn ``~/models/BCAR`` into
    ``$PWD/~/models/BCAR`` (a literal ``~`` component), which the re-executed
    child's ``expanduser`` can no longer recover — so a BCAR/dir/log path that
    works in the foreground would be "not found" only under ``--daemon``.
    """

    return os.path.abspath(os.path.expanduser(str(path)))


def _daemon_entrypoint_script() -> str:
    """Return the console entrypoint's file path for the daemon re-exec.

    Executing the script file (rather than ``-m vpmdk_entry``) puts its own
    directory on ``sys.path[0]``, so a source checkout keeps working: the
    runtime ``sys.path`` insertion that ``vpmdk.py`` performs is not inherited
    across ``execv``.
    """

    import vpmdk_entry

    script = getattr(vpmdk_entry, "__file__", None)
    if not script:  # pragma: no cover - namespace package without a file
        raise RuntimeError("Unable to locate the vpmdk_entry entrypoint script.")
    return os.path.abspath(script)


def _daemon_exec_argv(args, socket_path: str, log_file: str) -> list[str]:
    """Rebuild the ``serve`` command line for the re-executed daemon child.

    Every path is absolutized because the daemon chdirs to ``/`` so it never
    pins the (often temporary) launch directory.
    """

    argv = [
        sys.executable,
        # -u: the daemon's stdout is a dup2'd REGULAR FILE (the log), which
        # CPython block-buffers. Every print()-based startup diagnostic --
        # including the SERVER_MODE_SPEC 2.1 "BCAR not found; starting server
        # with default MLP=CHGNET." warning and the backend builders' own
        # messages -- then sat in that buffer for the life of the daemon, so
        # `grep` on the log found nothing while the server ran and the text was
        # lost entirely if it was SIGKILLed. The logger's records were unaffected
        # (a separate line-buffered handler), which is what made the gap easy to
        # miss. Unbuffered output costs nothing here: these are a handful of
        # startup lines plus per-request output that is already streamed.
        "-u",
        _daemon_entrypoint_script(),
        "serve",
        "--dir",
        _abspath_user(args.dir),
        "--socket",
        socket_path,
        "--idle-timeout",
        repr(float(args.idle_timeout)),
        "--log-file",
        log_file,
        "--daemon",
    ]
    if args.bcar:
        argv.extend(["--bcar", _abspath_user(args.bcar)])
    return argv


def _daemonize(
    log_file: str,
    exec_argv: list[str],
    *,
    refuse_symlink: bool = False,
) -> tuple[bool, int | None, str | None]:
    """Double-fork, then exec a fresh interpreter, keeping a readiness pipe.

    Importing ``vpmdk_core`` starts the ML runtimes, which spawn native worker
    threads (torch/OpenMP, JAX). ``fork()`` only clones the calling thread, so a
    child that kept running this interpreter could deadlock on a mutex another
    thread held at fork time — exactly what happens while loading a model. The
    child therefore performs only async-signal-safe syscalls before ``execv``
    replaces the address space with a pristine single-threaded interpreter.
    """

    read_fd, write_fd = os.pipe()
    # os.pipe() hands out the lowest free descriptors, so a caller that closed
    # its standard streams (supervisor, cron, `1>&- 2>&-`) can get fd 1 or 2.
    # The child's dup2 redirections below would then silently close the pipe,
    # and the daemon would be told to notify on its own stderr. Keep both ends
    # above the standard descriptors before anyone can clobber them.
    read_fd = _move_fd_above_stdio(read_fd)
    write_fd = _move_fd_above_stdio(write_fd)
    start_timeout = _daemon_start_timeout()
    first_pid = os.fork()
    if first_pid > 0:
        os.close(write_fd)
        # Bound the wait: a child that stalls before exec (the very hazard this
        # design guards against) must not hang the CLI without a diagnostic.
        deadline = time.monotonic() + start_timeout
        chunks: list[bytes] = []
        timed_out = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if not select.select([read_fd], [], [], remaining)[0]:
                timed_out = True
                break
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        os.close(read_fd)
        # The intermediate child only setsid()s and forks, but that fork runs
        # os.register_at_fork handlers installed by the ML runtimes; bound this
        # wait too so the same stall cannot hang the CLI after the select gave up.
        _reap_with_deadline(first_pid, deadline)
        if timed_out:
            return True, None, (
                f"TIMEOUT:daemon did not report readiness within "
                f"{start_timeout:g}s. It may still be loading; check with "
                f"'vpmdk status' and shut it down with 'vpmdk stop' if unwanted"
            )
        return True, None, b"".join(chunks).decode("utf-8", errors="replace").strip()

    os.close(read_fd)
    try:
        os.setsid()
        second_pid = os.fork()
        if second_pid > 0:
            os._exit(0)
        # No umask override: the log, socket, pidfile and socket directory all
        # set explicit modes, and forcing 0o077 here would leak into every
        # calculation the resident server writes (0600 OUTCAR/CONTCAR/...).
        # Open the replacements above the standard descriptors first. Otherwise
        # os.open can hand back fd 1 or 2 (when the caller closed them), making
        # the dup2 below a no-op that preserves FD_CLOEXEC — so the daemon would
        # lose that stream across execv and log nothing.
        devnull_fd = _move_fd_above_stdio(os.open(os.devnull, os.O_RDONLY))
        # O_NOFOLLOW for the DERIVED <socket>.log only: this fd is dup2'd onto 1
        # and 2, so following a pre-planted symlink at that predictable path would
        # append every log line, backend stdout line and traceback into an
        # attacker-chosen file as the server user. An explicit --log-file is the
        # user's own path and may legitimately be a symlink.
        log_open_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if refuse_symlink:
            # Check the entry BEFORE opening: a pre-planted readerless FIFO would
            # make this open() block forever, and O_NOFOLLOW alone would still
            # accept an attacker-owned REGULAR file (the 0600 mode is inert for an
            # existing inode) whose content then receives everything dup2'd onto
            # stdout/stderr.
            _assert_private_log_path(log_file)
            log_open_flags |= getattr(os, "O_NOFOLLOW", 0)
        # O_NONBLOCK: a reader-less FIFO at an explicit --log-file made this
        # open block in the kernel forever -- the child never execs, the
        # launcher waits the full daemon-start timeout (600 s) and prints
        # advice that cannot work, and a permanently stuck orphan leaks --
        # while the identical FOREGROUND invocation fails in <1 s via
        # _probe_foreground_log_file. With O_NONBLOCK the same condition is
        # an immediate ENXIO, reported through the readiness pipe. A FIFO
        # WITH a reader still opens; set_blocking(True) below restores
        # ordinary blocking writes for it (and is inert for regular files).
        log_fd = _move_fd_above_stdio(
            os.open(
                log_file,
                log_open_flags | getattr(os, "O_NONBLOCK", 0),
                0o600,
            )
        )
        os.set_blocking(log_fd, True)
        if refuse_symlink:
            # Re-check the fd actually opened, closing the lstat->open race.
            _verify_private_log_fd(log_fd, log_file)
        # Use literal descriptors: sys.stdin/stdout/stderr are None when the
        # caller launched us with a closed standard stream.
        os.dup2(devnull_fd, 0)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        # dup2 clears FD_CLOEXEC on the target, but be explicit so the standard
        # streams are guaranteed to survive execv.
        for standard_fd in (0, 1, 2):
            os.set_inheritable(standard_fd, True)
        os.close(devnull_fd)
        os.close(log_fd)
        # The readiness pipe must survive exec; os.pipe() FDs are close-on-exec.
        os.set_inheritable(write_fd, True)
        os.execv(
            exec_argv[0],
            [*exec_argv, "--daemon-notify-fd", str(write_fd)],
        )
        raise RuntimeError("execv returned unexpectedly")  # pragma: no cover
    except BaseException as exc:
        # This runs in the FORKED CHILD, which must never return into the parent's
        # CLI code: it is a fork of a multi-threaded process that already imported
        # the torch/JAX stack, so a normal interpreter shutdown here would run
        # those atexit handlers post-fork -- exactly what os._exit avoids.
        # Reporting is therefore best effort and the exit is unconditional: when
        # the launcher had already stopped waiting and closed the read end, the
        # os.write raised BrokenPipeError, which escaped _daemonize (os._exit sat
        # OUTSIDE this try) and let the child fall through to serve_cli's generic
        # `except OSError`, printing a second bogus "unable to daemonize" on the
        # user's terminal and returning 1 up through main().
        try:
            with contextlib.suppress(Exception):
                os.write(
                    write_fd,
                    _encode_utf8_lenient(f"ERROR:{type(exc).__name__}: {exc}\n"),
                )
            with contextlib.suppress(Exception):
                os.close(write_fd)
        finally:
            os._exit(1)


def _drain_stream_guarded(stream) -> None:
    """Flush one standard stream without letting a dead consumer override the
    exit code.

    ``vpmdk serve`` was the one subcommand still exposed to the R142 class:
    with the stream on a full filesystem or a pipe whose reader exited, the
    daemon-parent success print (and a clean foreground shutdown) buffered,
    serve_cli returned 0, and CPython's interpreter-exit flush failure
    overrode the status to 120 -- a launcher then read a LIVE, VRAM-holding
    resident as a failed start (never calling stop), or a clean stop as a
    crash (restarting the server the operator just stopped). Flush here,
    inside the command, and on failure point the stream at /dev/null so the
    finalization flush cannot raise either.

    ONE function taking the stream, applied to BOTH stdout and stderr: the
    first version guarded stdout only, and a dead stderr consumer (the
    foreground logger is a StreamHandler on stderr, and the ML-stack import
    warnings land there too) reproduced the identical exit-120 override --
    the half-mirror class again.
    """

    import contextlib as _contextlib

    if stream is None:
        # A launcher that CLOSED the fd ('vpmdk serve ... 1>&-' / '2>&-')
        # makes CPython set the sys stream to None; flushing None raised
        # AttributeError out of serve_cli's finally, so a successful daemon
        # start (and a clean foreground shutdown) exited 1 -- the very harm
        # this guard exists to prevent, via the one stream state its first
        # version missed.
        return
    try:
        stream.flush()
    except (OSError, ValueError):
        with _contextlib.suppress(Exception):
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, stream.fileno())
            os.close(devnull_fd)


def serve_cli(args) -> int:
    """Implement the thin ``vpmdk serve`` CLI layer."""

    try:
        return _serve_cli_inner(args)
    finally:
        # The exit CODE is the contract; stream delivery is best-effort.
        _drain_stream_guarded(sys.stdout)
        _drain_stream_guarded(sys.stderr)


def _serve_cli_inner(args) -> int:
    """Implement the thin ``vpmdk serve`` CLI layer."""

    # Bind the readiness pipe before any validation: in the re-executed daemon
    # child the parent sees nothing but this pipe, so an early failure that only
    # printed to the (redirected) stderr would surface as "readiness pipe
    # closed" with no cause.
    daemon_notify_fd: int | None = getattr(args, "daemon_notify_fd", None)
    already_daemonized = daemon_notify_fd is not None
    daemon_child = already_daemonized

    def report_error(message: str) -> int:
        nonlocal daemon_notify_fd
        if daemon_notify_fd is not None:
            try:
                os.write(daemon_notify_fd, _encode_utf8_lenient(f"ERROR:{message}\n"))
                os.close(daemon_notify_fd)
            except OSError:
                pass
            daemon_notify_fd = None
        print(f"Error: {message}", file=sys.stderr)
        return 1

    if os.name != "posix" or not hasattr(socket, "AF_UNIX"):
        return report_error("VPMDK server mode requires POSIX Unix-domain sockets.")
    if not math.isfinite(args.idle_timeout) or args.idle_timeout < 0:
        return report_error("--idle-timeout must be a finite non-negative number.")
    try:
        socket_path = resolve_socket_path(args.socket)
        # serve_cli always writes a pidfile (R136), so pidfile problems are
        # decidable here, before the model load.
        prepare_socket_path(socket_path, pidfile_expected=True)
    except ServerAlreadyRunning as exc:
        return report_error(str(exc))
    except Exception as exc:
        return report_error(str(exc))

    server_ref: list[VPMDKServer] = []

    def finish(code: int) -> int:
        # A worker abandoned by the third shutdown signal is still executing native
        # backend code as a daemon thread. Returning normally lets CPython finalize
        # the interpreter underneath it, C++ `std::terminate` fires ("terminate
        # called without an active exception") and the process dies of SIGABRT -- so
        # a textbook-correct teardown (warning logged, socket unlinked, handlers
        # restored) reported exit 134. A supervisor reads that as a crash and can
        # restart the server the operator just stopped. os._exit skips finalization,
        # which is exactly what the daemon path already did, so the two entry points
        # now agree on the exit status of the same shutdown path.
        abandoned = any(
            getattr(instance, "_worker_abandoned", False) for instance in server_ref
        )
        if daemon_child or abandoned:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(code)
        return code

    preflight_log_fd: int | None = None
    log_file = args.log_file
    if already_daemonized:
        # Re-executed by _daemonize: this interpreter is already the detached
        # session leader with stdio redirected, so it must not fork again.
        log_file = _abspath_user(log_file or default_log_path(socket_path))
        # Release the launch directory. Resident servers are routinely started
        # from a scratch directory that is deleted afterwards; holding it would
        # make every later job fail in _working_directory's os.getcwd(). All
        # paths reaching this point are already absolute.
        try:
            os.chdir("/")
        except OSError as exc:
            return report_error(f"unable to leave the launch directory: {exc}")
    elif args.daemon:
        log_file = _abspath_user(log_file or default_log_path(socket_path))
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            exec_argv = _daemon_exec_argv(args, socket_path, log_file)
        except Exception as exc:
            return report_error(f"unable to prepare the daemon log: {exc}")
        try:
            is_parent, daemon_notify_fd, message = _daemonize(
                log_file, exec_argv, refuse_symlink=args.log_file is None
            )
        except OSError as exc:
            return report_error(f"unable to daemonize: {exc}")
        if is_parent:
            if message and message.startswith("READY:"):
                # The resident is already started and healthy; a dead stdout
                # consumer must not turn that into exit 1. With an UNBUFFERED
                # stdout (PYTHONUNBUFFERED=1 / python -u, the standard
                # container/CI setup) the EPIPE surfaces inside print()
                # itself, past the flush-time _drain_stream_guarded in
                # serve_cli's finally -- a raw BrokenPipeError traceback for
                # a successful start, which a supervisor reads as "failed"
                # and orphans the model-holding resident. The client's
                # _write_line already guards its immediate-write face.
                with contextlib.suppress(OSError):
                    print(message.removeprefix("READY:"))
                return 0
            if message and message.startswith("TIMEOUT:"):
                return report_error(message.removeprefix("TIMEOUT:"))
            # Both child writers prepend "ERROR:" on the pipe; strip it like
            # the READY:/TIMEOUT: markers above, or every failed daemon start
            # printed the internal protocol marker twice ("Error: daemon
            # failed to start: ERROR:RuntimeError: ...").
            if message and message.startswith("ERROR:"):
                message = message.removeprefix("ERROR:")
            return report_error(
                f"daemon failed to start: {message or 'readiness pipe closed'}"
            )
        daemon_child = True  # pragma: no cover - _daemonize execs the child
    elif log_file is not None:
        # Foreground with an explicit --log-file: probe it BEFORE the model
        # load, mirroring the daemon path (which opens the log in _daemonize
        # before any backend work). The returned fd is HELD until the
        # server's FileHandler has opened the path (see
        # _probe_foreground_log_file) and closed right after construction.
        log_file = _abspath_user(log_file)
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            preflight_log_fd = _probe_foreground_log_file(log_file)
        except OSError as exc:
            return report_error(f"unable to open the log file: {exc}")

    pidfile_reserved = False
    try:
        # Reserve the endpoint atomically BEFORE the model load: two
        # concurrent serves targeting the same unused socket both pass
        # prepare_socket_path (a check, not a reservation), and the loser was
        # only detected at _bind -- after BOTH processes had loaded a
        # potentially GPU-sized backend. _write_pidfile opens with
        # O_CREAT|O_EXCL and raises ServerAlreadyRunning when the existing
        # record belongs to a live process, so the loser now fails here,
        # first. This runs in the FINAL server process (the daemon child
        # re-enters serve_cli already daemonized; the launcher parent returns
        # above), and VPMDKServer's own pidfile write later finds this
        # process's pid in the record and rewrites it in place.
        _write_pidfile(pidfile_path(socket_path), socket_path)
        pidfile_reserved = True
        calculator, tags, base_dir = _load_backend_for_server(args.dir, args.bcar)
        server = VPMDKServer(
            socket_path,
            calculator,
            tags,
            backend_base_dir=base_dir,
            idle_timeout=args.idle_timeout,
            # The pidfile is written for FOREGROUND serves too: it is the only
            # liveness evidence that survives a force-stop drain (listener
            # closed, worker still computing, socket file kept). Without it a
            # second `vpmdk serve` classified the deaf-but-alive foreground
            # server as a stale socket, unlinked it, and loaded a second
            # resident model beside the draining one -- the exact double-VRAM
            # scenario the daemon path already refused via its pidfile.
            pidfile=pidfile_path(socket_path),
            log_file=None if args.daemon else log_file,
            log_file_named=args.log_file is not None,
        )
        if preflight_log_fd is not None:
            # The FileHandler now holds its own writer on the log path, so a
            # FIFO's reader stays alive without the probe fd.
            with contextlib.suppress(OSError):
                os.close(preflight_log_fd)
            preflight_log_fd = None
        if not already_daemonized:
            # Release the launch directory for the foreground path too (the
            # already-daemonized child already chdir('/')ed above). A resident
            # server is routinely started from a scratch dir that is deleted
            # afterwards; pinning it would make every later job fail in
            # _working_directory's os.getcwd(). Everything the server needs is
            # already absolute (VPMDKServer.__init__ absolutizes socket_path and
            # log_file; base_dir from _load_backend_for_server is absolute), so
            # this is safe. Never abort serving over a chdir failure.
            with contextlib.suppress(OSError):
                os.chdir("/")
        server_ref.append(server)
        server.install_signal_handlers()

        def notify_ready() -> None:
            nonlocal daemon_notify_fd
            if daemon_notify_fd is None:
                return
            message = f"READY:VPMDK server ready at {socket_path} (pid {os.getpid()})\n"
            try:
                os.write(daemon_notify_fd, _encode_utf8_lenient(message))
            except OSError:
                # The launcher stopped waiting (its readiness timeout elapsed),
                # so the pipe is gone. The model is loaded and the socket is
                # bound: stay resident instead of dying on a BrokenPipeError.
                server.logger.warning(
                    "Readiness pipe closed before startup completed; "
                    "the launcher stopped waiting but the server is ready."
                )
            finally:
                with contextlib.suppress(OSError):
                    os.close(daemon_notify_fd)
                daemon_notify_fd = None

        try:
            server.serve_forever(ready_callback=notify_ready)
        finally:
            server.restore_signal_handlers()
        return finish(0)
    except ServerAlreadyRunning as exc:
        error = str(exc)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"

    if preflight_log_fd is not None:
        # The success path closes the probe fd right after VPMDKServer
        # construction; when backend loading or construction fails, only this
        # error tail runs. The CLI process exits soon after, but a long-lived
        # caller invoking serve_cli repeatedly leaked one fd per failed
        # foreground start with an explicit --log-file.
        with contextlib.suppress(OSError):
            os.close(preflight_log_fd)
        preflight_log_fd = None

    if pidfile_reserved and not any(
        getattr(candidate, "_pidfile_written", False) for candidate in server_ref
    ):
        # Release the pre-load endpoint reservation when startup failed
        # before a server took ownership of the pidfile. Ownership transfers
        # at the LAST statement of _bind (self._pidfile_written = True), not
        # at construction: server_ref is populated before serve_forever, so
        # gating on `not server_ref` alone left the reservation on disk for
        # every _bind failure (path too long, parent turned unwritable during
        # the model load, EADDRINUSE) -- _cleanup skips it for the same
        # not-yet-owned reason. Once ownership HAS transferred, the server's
        # own _cleanup removes the file and this tail stays out of the way.
        # Ownership-checked removal, so a reservation loser can never delete
        # the winner's file (pidfile_reserved is False when _write_pidfile
        # itself refused).
        with contextlib.suppress(Exception):
            _remove_owned_pidfile(
                pidfile_path(socket_path), socket_path, os.getpid()
            )

    if daemon_notify_fd is not None:
        try:
            os.write(daemon_notify_fd, _encode_utf8_lenient(f"ERROR:{error}\n"))
            os.close(daemon_notify_fd)
        except OSError:
            pass
    print(f"Error: {error}", file=sys.stderr)
    return finish(1)
