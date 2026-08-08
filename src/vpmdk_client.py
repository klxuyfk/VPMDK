"""Standard-library-only client and CLI for VPMDK server mode."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from vpmdk_protocol import (
    MAX_REQUEST_BYTES,
    PROTOCOL_VERSION,
    default_socket_path,
    resolve_socket_path,
)


# Bound for reading a stop acknowledgement when the caller asked not to wait for
# shutdown to complete (``--timeout 0``).
STOP_ACK_TIMEOUT = 60.0

# socket.settimeout() converts its argument to the platform time_t and raises
# OverflowError for values beyond ~time_t max (~9.2e9 s on 64-bit). Reject an
# absurd finite --timeout up front so it never reaches settimeout as an uncaught
# ArithmeticError (traceback + off-contract exit 1 + leaked fd). 1e9 s (~31
# years) is far below that threshold yet larger than any real request; use
# --timeout 0 to wait indefinitely.
_MAX_REQUEST_TIMEOUT = 1e9


def _current_directory_for_request() -> str:
    """Return the caller's cwd for the request, or fail with a clean message.

    ``os.getcwd()`` raises FileNotFoundError when the shell's directory has
    been deleted underneath it. Left unguarded that escaped ``client_cli`` as
    a raw traceback (exit 1 only via the interpreter's uncaught-exception
    default), while ``status``/``stop`` from the same state printed the
    one-line ``Error:`` diagnostic every other client failure emits. ValueError
    is the type ``client_cli`` maps to a clean exit 1.
    """

    try:
        return os.path.abspath(os.getcwd())
    except OSError as exc:
        raise ValueError(
            "the current working directory no longer exists; cd to an "
            "existing directory and resubmit"
        ) from exc


def _validate_request_timeout(timeout: float) -> None:
    """Reject a non-finite, negative, or absurdly large request timeout."""

    if not math.isfinite(timeout) or timeout < 0 or timeout > _MAX_REQUEST_TIMEOUT:
        raise ValueError(
            "timeout must be a finite number between 0 and "
            f"{_MAX_REQUEST_TIMEOUT:g} seconds (use 0 to wait indefinitely)"
        )


class VPMDKClientError(RuntimeError):
    """Base class for server-mode client errors."""


class ServerConnectionError(VPMDKClientError):
    """The server could not be reached or disconnected unexpectedly."""


class ClientTimeoutError(VPMDKClientError):
    """A client-side deadline expired."""


class RemoteCalculationError(VPMDKClientError):
    """The server completed a request with an execution failure."""

    def __init__(self, message: str, *, traceback: str | None = None):
        # `traceback` is a diagnostic-only field that client_cli prints with
        # .rstrip(). A malformed or version-skewed peer can send a truthy
        # NON-string (e.g. {"traceback":["frame","frame"]} or a number), which
        # would raise an uncaught AttributeError there -- a traceback plus an
        # off-contract exit instead of the documented code for the failure.
        # Accept only a real string; anything else is not a traceback, so drop it
        # (treated as absent). The authoritative signals -- the `code`-derived
        # exception class and the message -- are unaffected, so the documented
        # exit code (2 for a calculation failure, 1 for input, 5 for mismatch) is
        # preserved rather than being downgraded to a protocol/connection error.
        # Normalizing HERE covers every construction site at once.
        self.traceback = traceback if isinstance(traceback, str) else None
        super().__init__(message)


class RemoteInputError(RemoteCalculationError):
    """The remote work directory contains invalid or missing inputs."""


class RemoteBackendMismatch(RemoteCalculationError):
    """The request BCAR does not match the resident calculator."""


class ProtocolError(ServerConnectionError):
    """The peer did not implement the expected NDJSON protocol."""


class VPMDKClient:
    """Small synchronous client for one VPMDK Unix-domain socket."""

    def __init__(self, socket_path: str | None = None, *, connect_timeout: float = 2.0):
        self.socket_path = resolve_socket_path(socket_path)
        timeout_value = float(connect_timeout)
        # Validated here, like the request timeout: a negative, NaN, infinite,
        # or beyond-time_t value passes float() but makes settimeout() raise
        # ValueError/OverflowError on the FIRST request -- after the socket
        # object exists and outside the connection-error handlers, leaving the
        # descriptor to garbage collection. Zero is rejected too: settimeout(0)
        # means non-blocking, and a non-blocking connect cannot succeed here.
        if (
            not math.isfinite(timeout_value)
            or timeout_value <= 0
            or timeout_value > _MAX_REQUEST_TIMEOUT
        ):
            raise ValueError(
                "connect_timeout must be a finite number greater than 0 and "
                f"at most {_MAX_REQUEST_TIMEOUT:g} seconds"
            )
        self.connect_timeout = timeout_value

    def _connect(self, *, deadline: float | None) -> socket.socket:
        if os.name != "posix" or not hasattr(socket, "AF_UNIX"):
            raise ServerConnectionError(
                "VPMDK server mode requires POSIX Unix-domain sockets."
            )
        connect_timeout = self.connect_timeout
        deadline_limited = False
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ClientTimeoutError("VPMDK server request timed out")
            deadline_limited = remaining <= connect_timeout
            connect_timeout = min(connect_timeout, remaining)
        self._verify_socket_ownership()
        # Create the socket INSIDE the classified try: socket() and settimeout()
        # can raise OSError of their own (EMFILE/ENFILE near RLIMIT_NOFILE,
        # ENOBUFS), and outside the try that escaped every VPMDKClientError
        # mapping as an uncaught traceback with an off-contract exit code instead
        # of the documented unreachable/timeout results.
        connection: socket.socket | None = None
        try:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(connect_timeout)
            connection.connect(self.socket_path)
        except socket.timeout as exc:
            if connection is not None:
                connection.close()
            if deadline_limited:
                raise ClientTimeoutError("VPMDK server request timed out") from exc
            raise ServerConnectionError(
                f"Cannot connect to VPMDK server at {self.socket_path}: {exc}"
            ) from exc
        except OSError as exc:
            if connection is not None:
                connection.close()
            if deadline is not None and time.monotonic() >= deadline:
                raise ClientTimeoutError("VPMDK server request timed out") from exc
            raise ServerConnectionError(
                f"Cannot connect to VPMDK server at {self.socket_path}: {exc}"
            ) from exc
        return connection

    def _verify_default_socket_parent(self) -> None:
        """Refuse the DEFAULT endpoint when its parent directory is not ours.

        Checking only the socket's own owner is not enough at the default path.
        With XDG_RUNTIME_DIR unset that path is ``/tmp/vpmdk-<uid>/default.sock``,
        and /tmp is world-writable: before the victim's first server ever runs, an
        attacker can create ``/tmp/vpmdk-<uid>/`` (owning it) and plant
        ``default.sock`` as a SYMLINK to some other socket the victim owns -- a
        different resident server, ssh-agent, tmux. The owner check then passes
        (the target really is uid-owned), so the client would speak the protocol
        to an endpoint the attacker chose: at best a protocol error, at worst
        calculations silently routed to a different resident model.

        The SERVER already refuses to bind under a foreign-owned parent
        (_secure_private_socket_parent). Mirror that on the client, and only for
        the DEFAULT DIRECTORY: an explicit --socket/VPMDK_SOCKET pointing
        somewhere else is a location the user chose, and the docs put directory
        safety there on the user.

        Keyed on the PARENT DIRECTORY, exactly like the server's own gate
        (ensure_socket_directory compares os.path.dirname(default_socket_path())
        since round 112). Comparing the whole socket FILENAME here meant any
        sibling name inside the very same predictable directory -- the documented
        one-server-per-GPU layout ``--socket /tmp/vpmdk-<uid>/gpu0.sock`` -- ran
        no parent check at all, even though the server refuses to bind there. The
        squattable artifact is the directory, not the file name.
        """

        parent = os.path.dirname(self.socket_path) or "."
        default_parent = os.path.dirname(default_socket_path()) or "."
        if os.path.abspath(parent) != os.path.abspath(default_parent):
            return
        try:
            info = os.lstat(parent)
        except OSError:
            return  # absent/unreadable: connect() reports it as unreachable
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ServerConnectionError(
                f"Refusing to use the default VPMDK socket directory: {parent} "
                "is not a real directory"
            )
        expected_uid = os.geteuid()
        if info.st_uid != expected_uid:
            raise ServerConnectionError(
                f"Refusing to use the default VPMDK socket directory {parent}: "
                f"it is owned by uid {info.st_uid} (expected {expected_uid}); "
                "another user may be redirecting the endpoint"
            )
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ServerConnectionError(
                f"Refusing to use the default VPMDK socket directory {parent}: "
                "it is group/world-writable, so another user could replace the "
                "socket"
            )

    def _verify_socket_ownership(self) -> None:
        """Refuse an endpoint another user owns, before speaking the protocol.

        The default socket path is predictable and, with XDG_RUNTIME_DIR unset,
        lives under a world-writable /tmp -- so another local user can pre-bind it
        and impersonate the server. The SERVER side already treats exactly this as
        an attack (_secure_private_socket_parent refuses a foreign-owned parent),
        and examples/server_batch/run.sh warns about it, but the client trusted
        whatever answered: a forged ``{"event":"done","ok":true}`` is
        indistinguishable from a real calculation, and the documented readiness
        gate (``until vpmdk status --socket ...``) succeeds on its FIRST iteration
        against the impostor, so its dead-server check never runs.

        A MISSING socket is deliberately left to connect(), so the unreachable
        message and exit code stay exactly as before. This cannot break a working
        setup: the server chmods its socket 0600, so a socket owned by another
        user is unusable anyway -- it can only be an impostor.

        The check FOLLOWS symlinks (os.stat, not os.lstat) and validates what they
        point AT. A stable alias like ``ln -s /run/user/1000/vpmdk-1000/gpu0.sock
        ~/vpmdk.sock`` is a legitimate setup that connect() handles fine, and
        lstat'ing the final component rejected it as a "non-socket path" before
        even trying. Following the link loses nothing: a link planted by an
        attacker resolves to THEIR socket, whose owner check then fails, and a
        link to a non-socket is still refused.
        """

        self._verify_default_socket_parent()
        try:
            info = os.stat(self.socket_path)
        except OSError:
            return
        if not stat.S_ISSOCK(info.st_mode):
            raise ServerConnectionError(
                "Refusing to use a non-socket path as a VPMDK server endpoint: "
                f"{self.socket_path}"
            )
        expected_uid = os.geteuid()
        if info.st_uid != expected_uid:
            raise ServerConnectionError(
                f"Refusing to trust VPMDK socket owned by uid {info.st_uid} "
                f"(expected {expected_uid}); another user may be impersonating "
                f"the server at {self.socket_path}"
            )

    @staticmethod
    def _event_stream(
        connection: socket.socket,
        *,
        deadline: float | None,
    ) -> Iterator[dict[str, Any]]:
        buffer = bytearray()
        while True:
            if deadline is None:
                connection.settimeout(None)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ClientTimeoutError("VPMDK server request timed out")
                connection.settimeout(remaining)
            try:
                chunk = connection.recv(65536)
            except socket.timeout as exc:
                raise ClientTimeoutError("VPMDK server request timed out") from exc
            except OSError as exc:
                raise ServerConnectionError(
                    f"Connection to VPMDK server was lost: {exc}"
                ) from exc
            if not chunk:
                if buffer:
                    raise ProtocolError("Server closed with an incomplete JSON event")
                return
            buffer.extend(chunk)
            while b"\n" in buffer:
                raw_line, _, remainder = buffer.partition(b"\n")
                buffer = bytearray(remainder)
                if not raw_line:
                    continue
                if len(raw_line) > MAX_REQUEST_BYTES:
                    raise ProtocolError("Server event exceeds size limit")
                try:
                    event = json.loads(raw_line.decode("utf-8"))
                except (ValueError, RecursionError) as exc:
                    # Every way decode()/json.loads can reject a malformed peer
                    # event is caught and mapped to ProtocolError (exit 3), like
                    # the server's _read_request mirror, instead of leaking an
                    # uncaught exception (exit 1). ValueError is the base of
                    # UnicodeDecodeError, json.JSONDecodeError AND the plain
                    # ValueError json.loads raises for an over-4300-digit integer
                    # literal (Python 3.11+ int-string limit); catching the base
                    # avoids missing future ValueError subclasses. RecursionError
                    # (a RuntimeError) covers a deeply-nested line.
                    raise ProtocolError(f"Invalid server JSON event: {exc}") from exc
                if not isinstance(event, dict):
                    raise ProtocolError("Server event must be a JSON object")
                yield event
            # Check the unterminated leftover AFTER splitting, so an oversized
            # remainder that trailed a valid newline-terminated event is caught
            # here instead of blocking the next (possibly no-timeout) recv().
            if len(buffer) > MAX_REQUEST_BYTES:
                raise ProtocolError("Server event exceeds size limit")

    def _request(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float,
    ) -> Iterator[dict[str, Any]]:
        _validate_request_timeout(timeout)
        deadline = time.monotonic() + timeout if timeout > 0 else None
        payload = json.dumps(dict(request), separators=(",", ":")) + "\n"
        connection = self._connect(deadline=deadline)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                connection.close()
                raise ClientTimeoutError("VPMDK server request timed out")
            connection.settimeout(remaining)
        try:
            connection.sendall(payload.encode("utf-8"))
        except socket.timeout as exc:
            connection.close()
            if deadline is not None:
                raise ClientTimeoutError("VPMDK server request timed out") from exc
            raise ServerConnectionError(f"Unable to send server request: {exc}") from exc
        except OSError as exc:
            connection.close()
            if deadline is not None and time.monotonic() >= deadline:
                raise ClientTimeoutError("VPMDK server request timed out") from exc
            raise ServerConnectionError(f"Unable to send server request: {exc}") from exc

        def events() -> Iterator[dict[str, Any]]:
            try:
                yield from self._event_stream(connection, deadline=deadline)
            finally:
                connection.close()

        return events()

    @staticmethod
    def _raise_event_error(event: Mapping[str, Any]) -> None:
        message = str(event.get("error") or "VPMDK server request failed")
        code = event.get("code")
        traceback = event.get("traceback")
        if code == "backend_mismatch":
            raise RemoteBackendMismatch(message, traceback=traceback)
        if code == "input_error":
            raise RemoteInputError(message, traceback=traceback)
        if code == "calculation_error":
            raise RemoteCalculationError(message, traceback=traceback)
        if code == "server_stopping":
            raise ServerConnectionError(message)
        if code == "protocol_error":
            # A protocol-level rejection (version skew, malformed/oversized frame)
            # is a connection/protocol failure, not a calculation failure: surface
            # it as ProtocolError (a ServerConnectionError -> exit 3), matching the
            # client's own malformed-JSON handling and the status 0/3 contract.
            raise ProtocolError(message)
        if code == "internal_error":
            # A server-side defect, not a connectivity problem: surface it as a
            # failed request with the server traceback rather than exit 3.
            raise RemoteCalculationError(message, traceback=traceback)
        # A failed terminal event whose ``code`` is absent (the server protocol contract
        # documents ``{"event":"done","ok":false,"error":...}`` with no code) or
        # unrecognized (a newer server code) is still a failed *request*, not a
        # lost connection: report it as a calculation failure (exit 2) rather than
        # a ProtocolError/ServerConnectionError (exit 3).
        raise RemoteCalculationError(message, traceback=traceback)

    def run(
        self,
        workdir: str = ".",
        *,
        timeout: float = 0.0,
        log_callback: Callable[[str], None] | None = None,
        stderr_log_callback: Callable[[str], None] | None = None,
        event_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Submit one work directory and block until its terminal event."""

        # Resolve the cwd BEFORE the workdir: os.path.abspath on a RELATIVE
        # workdir consults getcwd too, and this order guarantees the deleted-cwd
        # condition surfaces through the guarded helper's clean diagnostic.
        caller_cwd = _current_directory_for_request()
        request = {
            "op": "run",
            "version": PROTOCOL_VERSION,
            # Absolutize ONLY -- deliberately no expanduser. the CLI compatibility contract
            # requires `run --dir D` to mean exactly what one-shot `vpmdk --dir D`
            # means, and run_workdir resolves with a bare os.path.abspath. An
            # unexpanded literal '~' (from a quoted argument or a shell variable)
            # therefore has to resolve the same way here: expanding it sent the
            # request to $HOME/calc while one-shot read ./~/calc, so the identical
            # invocation could compute a different structure, write its outputs to
            # a different directory, or exit 0 through the server and 1 one-shot.
            # A shell-expanded `--dir ~/calc` never reaches argv with a tilde, so
            # the ordinary case is unaffected; a literal tilde now fails
            # identically (POSCAR not found -> input_error) on both paths.
            # The one-shot compatibility contract forbids "fixing" this there.
            "workdir": os.path.abspath(workdir),
            "caller_cwd": caller_cwd,
            # The client's umask, so the server creates output artifacts with
            # the modes the byte-identical one-shot run would (a umask-077
            # client silently got world-readable outputs from a umask-022
            # resident, and the reverse broke group pipelines with 0600 files).
            # os.umask can only be read by writing; restore immediately.
            "umask": _read_current_umask(),
        }
        pending_log_parts: list[str] = []
        pending_log_stream = "stdout"

        def deliver_log(stream: str, line: str) -> None:
            # Server-side stderr (third-party warnings; see the server's
            # stderr relay) arrives as log events tagged stream="stderr" so
            # it can go back on the stream the one-shot CLI would have used.
            # Without a dedicated callback it falls back to log_callback, the
            # exact pre-tag behavior, so library callers see nothing new.
            if stream == "stderr" and stderr_log_callback is not None:
                stderr_log_callback(line)
            elif log_callback is not None:
                log_callback(line)

        for event in self._request(request, timeout=timeout):
            if event_callback is not None:
                event_callback(event)
            event_type = event.get("event")
            if event_type == "log":
                stream = "stderr" if event.get("stream") == "stderr" else "stdout"
                if pending_log_parts and stream != pending_log_stream:
                    # A chunked line only ever continues on its own stream; a
                    # stream switch completes the buffered line.
                    deliver_log(pending_log_stream, "".join(pending_log_parts))
                    pending_log_parts.clear()
                pending_log_stream = stream
                pending_log_parts.append(str(event.get("line", "")))
                if event.get("continued") is True:
                    continue
                deliver_log(stream, "".join(pending_log_parts))
                pending_log_parts.clear()
                continue
            if event_type == "done":
                if event.get("ok") is True:
                    return event
                self._raise_event_error(event)
            if event_type == "error":
                self._raise_event_error(event)
        raise ServerConnectionError("Server disconnected before completing the calculation")

    def status(self, *, timeout: float = 2.0) -> dict[str, Any]:
        request = {"op": "status", "version": PROTOCOL_VERSION}
        for event in self._request(request, timeout=timeout):
            if event.get("event") == "status":
                # Reject a structurally malformed status as a ProtocolError (-> exit
                # 3) rather than letting the formatter raise an uncaught exception
                # that escapes client_cli's excepts as an off-contract traceback
                # (the status exit-code contract: status is only exit 0 or 3). Guard every
                # field the formatter does more than str-interpolate:
                #   * `backend` is indexed with .get -> must be a JSON object (or
                #     omitted); a non-object (e.g. [1]) would AttributeError.
                #   * `uptime_s` is coerced with float() -> must be a JSON number
                #     (or omitted); a present null/string (e.g. null, "abc") would
                #     TypeError/ValueError. bool is a JSON bool, not a number.
                backend = event.get("backend")
                if backend is not None and not isinstance(backend, Mapping):
                    raise ProtocolError(
                        "Server status payload has a non-object 'backend' field"
                    )
                uptime = event.get("uptime_s")
                if uptime is not None and (
                    isinstance(uptime, bool) or not isinstance(uptime, (int, float))
                ):
                    raise ProtocolError(
                        "Server status payload has a non-numeric 'uptime_s' field"
                    )
                return event
            if event.get("event") == "error":
                self._raise_event_error(event)
        raise ServerConnectionError("Server disconnected without a status response")

    def stop(self, *, force: bool = False, timeout: float = 60.0) -> dict[str, Any]:
        """Request shutdown and, unless timeout is zero, wait for socket removal."""

        if not isinstance(force, bool):
            raise TypeError("force must be a boolean")
        _validate_request_timeout(timeout)
        started_at = time.monotonic()
        response: dict[str, Any] | None = None
        request = {"op": "stop", "version": PROTOCOL_VERSION, "force": force}
        # ``timeout=0`` means "do not wait for shutdown to finish", not "block
        # forever on the acknowledgement": _request treats 0 as no deadline, so
        # give the acknowledgement its own bound.
        request_timeout = timeout if timeout > 0 else STOP_ACK_TIMEOUT
        for event in self._request(request, timeout=request_timeout):
            if event.get("event") == "done" and event.get("ok") is True:
                response = event
                break
            if event.get("event") == "error" or event.get("ok") is False:
                self._raise_event_error(event)
        if response is None:
            raise ServerConnectionError("Server disconnected without accepting shutdown")

        if timeout == 0:
            return response

        # os.path.exists (FOLLOWS symlinks), not lexists: the endpoint may be a
        # symlink alias -- a setup _verify_socket_ownership deliberately supports.
        # The server unlinks its own bound path, which leaves the alias behind as
        # a DANGLING link; lexists stays true for that, so a clean shutdown was
        # never observed and `stop` through an alias always ended in a client
        # timeout (exit 4) instead of exit 0. Following the link asks the question
        # that actually matters: is the socket still there?
        while os.path.exists(self.socket_path):
            if time.monotonic() - started_at >= timeout:
                raise ClientTimeoutError("Timed out waiting for VPMDK server shutdown")
            time.sleep(0.05)
        return response


# The os.umask read-back dance is process-global: two concurrent run() calls
# (even through separate VPMDKClient instances) could interleave so one thread
# observed the other's temporary mask 0, or restored 0 as the process mask --
# after which every later file the process created was world-writable. The
# lock serializes VPMDK's own readers on the mutating fallback; on Linux the
# /proc read avoids mutating the mask at all.
_UMASK_LOCK = threading.Lock()


def _read_current_umask() -> int:
    try:
        with open("/proc/self/status", encoding="ascii", errors="replace") as status:
            for line in status:
                if line.startswith("Umask:"):
                    return int(line.split(":", 1)[1].strip(), 8)
    except (OSError, ValueError):
        pass
    with _UMASK_LOCK:
        current = os.umask(0)
        os.umask(current)
        return current


def _write_line(text: str, *, stream: Any = None) -> None:
    """Write one line of server-supplied text without ever raising on encoding.

    Paths and log lines can legitimately carry surrogate-escaped bytes: the OS
    layer decodes non-UTF-8 filesystem names with ``errors="surrogateescape"``,
    and the protocol transports them intact (the server's serializer falls back
    to ``ensure_ascii=True``). Writing such a string to a strict UTF-8 stdout
    raises UnicodeEncodeError -- which is a ValueError, so client_cli's trailing
    ``except ValueError`` would turn a SUCCESSFUL calculation into exit 1 and
    swallow the ``Calculation completed.`` marker that the streaming-output contract
    requires. An output-encoding detail must never change the exit contract, so
    fall back to writing the original bytes (surrogateescape round-trips them to
    exactly the bytes the filesystem gave us) and, failing that, to a lossy but
    infallible rendering.
    """

    if stream is None:
        stream = sys.stdout
    try:
        print(text, file=stream, flush=True)
        return
    except UnicodeEncodeError:
        pass
    except OSError:
        # The consumer of this stream is gone (`vpmdk run | head -1` after
        # head exits, a dead log collector): BrokenPipeError escaped
        # client_cli's except chain, so a calculation the server COMPLETED
        # (outputs written, marker sent) was reported as a raw traceback with
        # exit 120 (CPython's finalization-flush override) -- outside the
        # documented 0-5 table, defeating the retryable/non-retryable
        # distinction. Output delivery is best-effort; the EXIT CODE is the
        # contract. Silence this stream for the rest of the process so the
        # interpreter-exit flush cannot re-raise either.
        _silence_broken_stream(stream)
        return
    encoding = getattr(stream, "encoding", None) or "utf-8"
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        try:
            buffer.write(text.encode(encoding, "surrogateescape") + b"\n")
            buffer.flush()
            return
        except (UnicodeEncodeError, LookupError, OSError, ValueError):
            pass
    # Last resort: escape what cannot be encoded rather than fail the command.
    try:
        print(
            text.encode(encoding, "backslashreplace").decode(encoding, "replace"),
            file=stream,
            flush=True,
        )
    except OSError:
        _silence_broken_stream(stream)


def _silence_broken_stream(stream: Any) -> None:
    """Point a broken pipe at /dev/null so no later write can raise again."""

    import contextlib

    with contextlib.suppress(Exception):
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stream.fileno())
        os.close(devnull_fd)


def _format_status(status: Mapping[str, Any]) -> str:
    # Coerce a missing OR non-Mapping backend to {} up front so every backend.get
    # below is safe: VPMDKClient.status() already rejects a non-object backend as a
    # ProtocolError, but keep the formatter self-contained for any other caller.
    raw_backend = status.get("backend")
    backend = raw_backend if isinstance(raw_backend, Mapping) else {}
    model = backend.get("model")
    # VPMDKClient.status() already rejects a non-numeric uptime_s, but coerce
    # defensively so this formatter never raises for any other caller either
    # (every other field below is only str-interpolated, so this is the last
    # coercion that could throw).
    try:
        uptime_s = float(status.get("uptime_s", 0.0))
    except (TypeError, ValueError, OverflowError):
        # OverflowError is an ArithmeticError, NOT a ValueError: a JSON integer
        # past ~1.8e308 (but under the 4300-digit literal limit that json.loads
        # itself rejects) passes status()'s isinstance(int) check and then makes
        # float() raise it. Uncaught, it escapes every client_cli handler as a
        # traceback plus an undocumented exit code, while `status --json` renders
        # the byte-identical payload as exit 0. Tolerate it here so both
        # renderings agree and status stays 0/3 (the status exit-code contract).
        uptime_s = 0.0
    lines = [
        f"VPMDK server: {status.get('state', 'unknown')}",
        (
            "Backend: "
            f"MLP={backend.get('mlp')} MODEL={model if model is not None else '<default>'} "
            f"DEVICE={backend.get('device')}"
        ),
        f"PID: {status.get('pid')}  Uptime: {uptime_s:.1f} s",
        (
            f"Jobs: completed={status.get('jobs_completed', 0)} "
            f"failed={status.get('jobs_failed', 0)} queued={status.get('queue_length', 0)}"
        ),
        f"Protocol: {status.get('protocol')}  VPMDK: {status.get('vpmdk_version')}",
    ]
    if status.get("current_workdir"):
        lines.append(f"Current workdir: {status['current_workdir']}")
    options = backend.get("options") if isinstance(backend, Mapping) else None
    if isinstance(options, Mapping) and options:
        formatted_options = " ".join(
            f"{key}={value}" for key, value in sorted(options.items())
        )
        lines.append(f"Backend options: {formatted_options}")
    return "\n".join(lines)


def add_client_subcommands(subparsers: Any) -> None:
    """Add run/status/stop parsers to an argparse subparser collection.

    These must stay in sync with ``CLIENT_SUBCOMMANDS`` (the dispatch source of
    truth); ``test_client_subcommands_match_dispatch_constant`` enforces it.
    """

    run = subparsers.add_parser("run", help="submit one calculation to a server")
    run.add_argument("--dir", default=".", help="input directory")
    run.add_argument("--socket", help="Unix socket path")
    run.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        metavar="SEC",
        help="client deadline; 0 waits indefinitely (default: 0)",
    )

    status = subparsers.add_parser("status", help="show server status")
    status.add_argument("--socket", help="Unix socket path")
    status.add_argument("--json", action="store_true", help="print machine-readable JSON")

    stop = subparsers.add_parser("stop", help="stop a server")
    stop.add_argument("--socket", help="Unix socket path")
    stop.add_argument(
        "--force",
        action="store_true",
        help="request immediate shutdown and reject queued jobs",
    )
    stop.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="SEC",
        help="shutdown wait; 0 does not wait (default: 60)",
    )


def client_cli(args: argparse.Namespace) -> int:
    """Implement ``run``, ``status``, and ``stop`` exit-code contracts."""

    try:
        # Inside the mapped try: resolving a RELATIVE socket path consults
        # os.getcwd(), which raises from a deleted working directory --
        # constructed outside, that escaped every handler as a raw traceback.
        client = VPMDKClient(args.socket)
        if args.command == "run":
            saw_completion_marker = False

            def print_log(line: str) -> None:
                nonlocal saw_completion_marker
                _write_line(line)
                if line == "Calculation completed.":
                    saw_completion_marker = True

            def print_stderr_log(line: str) -> None:
                if sys.stderr is None:
                    # fd 2 closed (`vpmdk run 2>&-`): CPython sets sys.stderr
                    # to None and one-shot drops warnings entirely (the
                    # warnings machinery no-ops). _write_line treats a None
                    # stream as stdout, which would inject warning text into
                    # the stream scripts parse -- a None stream is part of
                    # this path's state space.
                    return
                _write_line(line, stream=sys.stderr)

            client.run(
                args.dir,
                timeout=args.timeout,
                log_callback=print_log,
                stderr_log_callback=print_stderr_log,
            )
            if not saw_completion_marker:
                _write_line("Calculation completed.")
            return 0
        if args.command == "status":
            try:
                status = client.status()
            except VPMDKClientError as exc:
                # the status exit-code contract allows only exit 0 (alive) or exit 3
                # (unreachable) for `status`. ANY failure to obtain a status --
                # an internal deadline timeout, a server-side internal_error, a
                # protocol_error, or a lost connection -- means "not a usable
                # status" and must map to exit 3, never the run --timeout code 4
                # or the calculation-failure code 2.
                raise ServerConnectionError(str(exc)) from exc
            if args.json:
                json_text = json.dumps(status, ensure_ascii=False, sort_keys=True)
                try:
                    json_text.encode("utf-8")
                except UnicodeEncodeError:
                    # A surrogate-escaped byte (non-UTF-8 filesystem name in a
                    # workdir or model path) survives ensure_ascii=False, and
                    # _write_line's surrogateescape fallback would then emit
                    # raw non-UTF-8 bytes -- machine-unparseable JSON with
                    # exit 0, DEGRADING the server's conforming wire frame
                    # (whose _serialize_event already falls back to
                    # ensure_ascii=True for exactly this case). Mirror that
                    # fallback so --json stdout always matches the wire bytes
                    # and stays parseable. The human-readable branch below
                    # keeps byte round-tripping.
                    json_text = json.dumps(status, ensure_ascii=True, sort_keys=True)
                _write_line(json_text)
            else:
                _write_line(_format_status(status))
            return 0
        if args.command == "stop":
            try:
                client.stop(force=args.force, timeout=args.timeout)
            except ClientTimeoutError:
                # Client-side timeout waiting for shutdown -> exit 4 (stop contract).
                raise
            except VPMDKClientError as exc:
                # the stop exit-code contract allows only exit 0 (stopped) / 3
                # (unreachable) / 4 (timeout) for `stop`. A non-timeout failure
                # (server internal_error, protocol_error, connection loss) is
                # "unreachable" -> exit 3, not a calculation failure (exit 2).
                raise ServerConnectionError(str(exc)) from exc
            if args.timeout == 0:
                # timeout=0 returns after acknowledgement without waiting for
                # shutdown, and the server may still be draining a calculation,
                # so do not claim it has already stopped.
                _write_line(f"VPMDK server stop requested: {client.socket_path}")
            else:
                _write_line(f"VPMDK server stopped: {client.socket_path}")
            return 0
        raise ValueError(f"Unknown client command: {args.command}")
    # Every diagnostic below can embed server-supplied text (an error message or
    # a remote traceback naming a surrogate-escaped path), so each goes through
    # _write_line: a stderr encoding failure must not replace the exit code the
    # branch just decided on.
    except RemoteBackendMismatch as exc:
        _write_line(f"Error: {exc}", stream=sys.stderr)
        return 5
    except RemoteInputError as exc:
        _write_line(f"Error: {exc}", stream=sys.stderr)
        return 1
    except RemoteCalculationError as exc:
        _write_line(f"Error: {exc}", stream=sys.stderr)
        if exc.traceback:
            _write_line(exc.traceback.rstrip(), stream=sys.stderr)
        return 2
    except ClientTimeoutError as exc:
        _write_line(f"Error: {exc}", stream=sys.stderr)
        return 4
    except ServerConnectionError as exc:
        _write_line(f"Error: {exc}", stream=sys.stderr)
        return 3
    except ValueError as exc:
        _write_line(f"Error: {exc}", stream=sys.stderr)
        return 1


def _client_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vpmdk",
        description="Submit to or manage a resident VPMDK server",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_client_subcommands(subparsers)
    return parser


def parse_client_args(parser: argparse.ArgumentParser, arguments: Sequence[str]):
    """Parse client arguments, mapping argparse's usage exit to the spec's code.

    argparse calls ``sys.exit(2)`` for a usage error (unknown flag, ``--timeout
    abc``, a missing value), which happens BEFORE client_cli's carefully mapped
    0/1/2/3/4/5 returns can run. But the server-mode exit-code contract reserves exit 2 for a
    RETRYABLE server-side calculation failure, so a permanently malformed
    invocation was reported to a retry driver as worth retrying. A bad command
    line is invalid input -> exit 1, matching the other client-side input errors.
    ``--help``/``--version`` exit 0 and are re-raised untouched.
    """

    try:
        return parser.parse_args(list(arguments))
    except SystemExit as exc:
        if exc.code == 2:
            raise SystemExit(1) from exc
        raise


def client_main(argv: Sequence[str] | None = None) -> int:
    """Parse and execute a lightweight client subcommand."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    return client_cli(parse_client_args(_client_parser(), arguments))


if __name__ == "__main__":
    raise SystemExit(client_main())
