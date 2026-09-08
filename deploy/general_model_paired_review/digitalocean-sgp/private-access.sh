#!/usr/bin/env bash
# Reach the authoritative remote review service through a local SSH listener.
set -euo pipefail
umask 077

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
ROOT=$(cd "$(dirname "$SCRIPT")/../../.." && pwd)
REVIEW_DIR="$ROOT/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02"
RUNTIME="$REVIEW_DIR/runtime/digitalocean-sgp-private-access"
SOCKET=hsd-general-model-paired-review-sgp-private
SESSION=paired-review-private-access
LOCAL_PORT=8772
REMOTE_PORT=8772
SSH_HOST=digitalocean-sgp

tmux_private() {
    tmux -L "$SOCKET" -f /dev/null "$@"
}

health() {
    curl --noproxy 127.0.0.1 --fail --silent --show-error --max-time 2 \
        "http://127.0.0.1:$LOCAL_PORT/api/health"
}

ensure_free_port() {
    python3 -B -S - "$LOCAL_PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit("Port 8772 is occupied. Inspect its owner before starting private access; no process was stopped.")
PY
}

run_forward() {
    local child_pid= exit_code=
    mkdir -p "$RUNTIME"
    exec 9>"$RUNTIME/tunnel.lock"
    if ! flock -n 9; then
        printf 'The dedicated private-access supervisor is already running.\n' >&2
        return 1
    fi
    trap 'trap - HUP INT TERM; if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; fi; exit 0' HUP INT TERM
    while true; do
        # A disconnected tunnel must not compete with a new owner of the port.
        ensure_free_port
        printf '%s starting private SSH access through %s\n' "$(date -u +%FT%TZ)" "$SSH_HOST"
        /usr/bin/ssh -nNT \
            -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no \
            -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -o ControlMaster=no -o ControlPath=none -o GatewayPorts=no \
            -o ForkAfterAuthentication=no -o PermitLocalCommand=no \
            -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$SSH_HOST" &
        child_pid=$!
        if wait "$child_pid"; then exit_code=0; else exit_code=$?; fi
        child_pid=
        printf '%s SSH exited (%s); retry in 3 seconds\n' "$(date -u +%FT%TZ)" "$exit_code"
        sleep 3 &
        child_pid=$!
        wait "$child_pid" || true
        child_pid=
    done
}

case "${1:-status}" in
    start)
        mkdir -p "$RUNTIME"
        exec 8>"$RUNTIME/start.lock"
        if ! flock -n 8; then
            printf 'Another private-access start is in progress.\n' >&2
            exit 1
        fi
        if tmux_private has-session -t "$SESSION" 2>/dev/null; then
            printf 'The dedicated private-access session is already running.\n'
            exec /bin/bash "$SCRIPT" status
        fi
        if ! flock -n "$RUNTIME/tunnel.lock" true; then
            printf 'A private-access foreground supervisor is running; inspect it before starting another.\n' >&2
            exit 1
        fi
        ensure_free_port
        printf -v command 'exec /bin/bash %q run >>%q 2>&1' "$SCRIPT" "$RUNTIME/tunnel.log"
        tmux_private new-session -d -s "$SESSION" -n tunnel "$command" 8>&-
        for attempt in {1..20}; do
            if ! tmux_private has-session -t "$SESSION" 2>/dev/null; then
                printf 'The private-access supervisor exited. Inspect %s/tunnel.log\n' "$RUNTIME" >&2
                exit 1
            fi
            if health >"$RUNTIME/health.json" 2>/dev/null; then
                printf 'Private access is ready: http://127.0.0.1:%s/\n' "$LOCAL_PORT"
                printf 'Authoritative service: %s 127.0.0.1:%s\nLogs: %s/tunnel.log\n' "$SSH_HOST" "$REMOTE_PORT" "$RUNTIME"
                exit 0
            fi
            sleep 0.5
        done
        printf 'The supervisor is running and will reconnect, but remote health is not ready. Inspect %s/tunnel.log\n' "$RUNTIME" >&2
        exit 1
        ;;
    status)
        if ! tmux_private has-session -t "$SESSION" 2>/dev/null; then
            printf 'The dedicated private-access tmux session is not running.\nLogs: %s/tunnel.log\n' "$RUNTIME"
            exit 1
        fi
        tmux_private list-panes -t "$SESSION" -F '#{session_name}:#{window_name} pid=#{pane_pid} command=#{pane_current_command}'
        if health; then
            printf '\nPrivate access: http://127.0.0.1:%s/ -> %s 127.0.0.1:%s\n' "$LOCAL_PORT" "$SSH_HOST" "$REMOTE_PORT"
        else
            printf '\nPrivate-access supervisor exists; the remote service is currently unreachable.\n' >&2
            printf 'Logs: %s/tunnel.log\n' "$RUNTIME" >&2
            exit 1
        fi
        printf 'Logs: %s/tunnel.log\n' "$RUNTIME"
        ;;
    stop)
        if tmux_private has-session -t "$SESSION" 2>/dev/null; then
            tmux_private kill-session -t "$SESSION"
        fi
        printf 'The dedicated private-access tmux session is stopped. The remote service and review records are retained.\n'
        ;;
    run)
        # Foreground mode, also used by the dedicated tmux session; Ctrl-C stops it.
        run_forward
        ;;
    *)
        printf 'Usage: bash %s {start|status|stop|run}\n' "$SCRIPT" >&2
        exit 2
        ;;
esac
