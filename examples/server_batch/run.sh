#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Model loading can take minutes for large checkpoints, so the readiness budget
# is generous and overridable rather than a fixed one-minute poll.
READY_TIMEOUT=${VPMDK_READY_TIMEOUT:-600}

# Keep the socket inside a private directory. A predictable name directly under
# a world-writable /tmp lets another user pre-bind that path on a shared
# machine; the readiness probe would then talk to their impersonating server and
# every job would be sent there. mktemp -d creates the directory 0700-owned.
SOCKET_DIR=
if [[ -n "${VPMDK_SOCKET:-}" ]]; then
    SOCKET_PATH=${VPMDK_SOCKET}
else
    SOCKET_DIR=$(mktemp -d "${TMPDIR:-/tmp}/vpmdk-server-batch-XXXXXX")
    SOCKET_PATH="${SOCKET_DIR}/server.sock"
fi

SERVER_PID=

# Return the PID the server at SOCKET_PATH reports, or nothing. Always succeeds:
# under `set -euo pipefail` a failing probe inside a command substitution would
# otherwise abort the script before it could print its own diagnostic.
served_pid() {
    local status_output pid
    status_output=$(vpmdk status --socket "${SOCKET_PATH}" 2>/dev/null) || return 0
    pid=$(printf '%s\n' "${status_output}" |
        sed -n 's/^PID:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1) || pid=
    printf '%s' "${pid}"
    return 0
}

terminate_server() {
    # Only ever touch a server this script started. Adopting a pre-existing
    # daemon and stopping it on exit would tear down someone else's resident
    # model (and every other client queued on it).
    [[ -n "${SERVER_PID}" ]] || return 0
    # If something else now answers this socket, never issue `vpmdk stop`
    # against it; only signal our own process.
    if [[ "$(served_pid)" != "${SERVER_PID}" ]]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=
        return 0
    fi
    # `vpmdk stop` fails whenever the socket was never created (for example the
    # model aborted during initialisation). An unguarded `wait` would then block
    # until the server's idle timeout, or forever, so always fall back to
    # signalling the process directly.
    if ! vpmdk stop --socket "${SOCKET_PATH}" --timeout 60 >/dev/null 2>&1; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
    fi
    for _ in $(seq 1 100); do
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=
    return 0
}

cleanup() {
    terminate_server
    if [[ -n "${SOCKET_DIR}" ]]; then
        rm -rf "${SOCKET_DIR}"
        SOCKET_DIR=
    fi
    return 0
}

# A trapped signal is not fatal to bash: without an explicit exit the script
# would resume the calculation loop and submit jobs to the socket cleanup just
# deleted. Exit with the conventional 128+signal status instead.
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# Refuse to adopt a server this script did not start.
if vpmdk status --socket "${SOCKET_PATH}" >/dev/null 2>&1; then
    echo "A VPMDK server is already running at ${SOCKET_PATH}." >&2
    echo "This example only manages a server it starts itself; stop that" >&2
    echo "server first, or point VPMDK_SOCKET at an unused path." >&2
    exit 1
fi

vpmdk serve \
    --dir "${SCRIPT_DIR}/model_config" \
    --socket "${SOCKET_PATH}" \
    --idle-timeout 3600 &
SERVER_PID=$!

deadline=$((SECONDS + READY_TIMEOUT))
while true; do
    # Check our own server first. Probing the socket first would silently adopt
    # a foreign server if ours lost a race for the path, and the loop's liveness
    # guard would never run because the probe already succeeded.
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "VPMDK server exited before becoming ready." >&2
        exit 1
    fi
    if vpmdk status --socket "${SOCKET_PATH}" >/dev/null 2>&1; then
        break
    fi
    if (( SECONDS >= deadline )); then
        echo "VPMDK server did not become ready within ${READY_TIMEOUT}s." >&2
        exit 1
    fi
    sleep 0.1
done

# The socket answers and our process is alive; confirm they are the same server
# before submitting anything to it.
responding_pid=$(served_pid)
if [[ "${responding_pid}" != "${SERVER_PID}" ]]; then
    echo "Socket ${SOCKET_PATH} is served by PID ${responding_pid:-unknown}," >&2
    echo "not the server this script started (${SERVER_PID}); refusing to submit." >&2
    exit 1
fi

for calculation in "${SCRIPT_DIR}"/calculations/*; do
    [[ -d "${calculation}" ]] || continue
    vpmdk run --socket "${SOCKET_PATH}" --dir "${calculation}"
done

terminate_server
