#!/usr/bin/env bash
# Keep the existing local review session reachable through an aliyun SSH tunnel.
set -euo pipefail
umask 077

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
ROOT=$(cd "$(dirname "$SCRIPT")/../.." && pwd)
REVIEW_DIR="$ROOT/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/reviews/paired-cases-02"
RUNTIME="$REVIEW_DIR/runtime"
SOCKET=hsd-general-model-paired-review
SESSION=paired-review
PUBLIC_ORIGIN=https://hsd.fenglin.pro
LOCAL_PORT=8772
REMOTE_PORT=18772

ensure_local_writer_allowed() {
    local marker="$REVIEW_DIR/session.json.remote-authority.json"
    if [[ -e "$marker" || -L "$marker" ]]; then
        printf 'Review records have moved to digitalocean-sgp; this old local writer will not start.\n' >&2
        printf 'Use: bash %s/deploy/general_model_paired_review/digitalocean-sgp/private-access.sh start\n' "$ROOT" >&2
        printf 'Migration marker: %s\n' "$marker" >&2
        return 1
    fi
}

tmux_review() {
    tmux -L "$SOCKET" -f /dev/null "$@"
}

health() {
    curl --noproxy 127.0.0.1 --fail --silent --show-error --max-time 2 \
        "http://127.0.0.1:$LOCAL_PORT/api/health"
}

repeat_command() {
    local component=$1 child_pid= exit_code=
    shift
    mkdir -p "$RUNTIME"
    exec >>"$RUNTIME/$component.log" 2>&1
    trap 'trap - HUP INT TERM; if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; fi; exit 0' HUP INT TERM
    while true; do
        printf '%s starting %s\n' "$(date -u +%FT%TZ)" "$component"
        "$@" &
        child_pid=$!
        if wait "$child_pid"; then exit_code=0; else exit_code=$?; fi
        child_pid=
        printf '%s %s exited (%s); retry in 3 seconds\n' "$(date -u +%FT%TZ)" "$component" "$exit_code"
        sleep 3 &
        child_pid=$!
        wait "$child_pid" || true
        child_pid=
    done
}

case "${1:-status}" in
    start)
        ensure_local_writer_allowed
        if tmux_review has-session -t "$SESSION" 2>/dev/null; then
            printf 'The paired-review session is already running.\n'
            exec /bin/bash "$SCRIPT" status
        fi
        # Refuse to compete with a manually started service or another task.
        /usr/bin/python - "$LOCAL_PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit("Port 8772 is occupied. Inspect its owner before starting; no process was stopped.")
PY
        mkdir -p "$RUNTIME"
        printf -v web_command 'exec /bin/bash %q run-web' "$SCRIPT"
        printf -v tunnel_command 'exec /bin/bash %q run-tunnel' "$SCRIPT"
        tmux_review new-session -d -s "$SESSION" -n web "$web_command"
        if ! tmux_review new-window -t "$SESSION" -n tunnel "$tunnel_command"; then
            tmux_review kill-session -t "$SESSION"
            exit 1
        fi
        for attempt in {1..20}; do
            if health >"$RUNTIME/health.json" 2>/dev/null; then
                printf 'Local review is ready. Browser entry after private forwarding: http://127.0.0.1:%s/\n' "$LOCAL_PORT"
                printf 'Client instructions: %s/deploy/general_model_paired_review/README.md\nLogs: %s\n' "$ROOT" "$RUNTIME"
                exit 0
            fi
            sleep 0.5
        done
        printf 'The supervisor started, but local health is not ready. Inspect %s/web.log\n' "$RUNTIME" >&2
        exit 1
        ;;
    status)
        tmux_review list-panes -a -F '#{session_name}:#{window_name} pid=#{pane_pid} command=#{pane_current_command}'
        health
        printf '\nForward: aliyun 127.0.0.1:%s -> local 127.0.0.1:%s\nLogs: %s\n' "$REMOTE_PORT" "$LOCAL_PORT" "$RUNTIME"
        ;;
    stop)
        if tmux_review has-session -t "$SESSION" 2>/dev/null; then
            tmux_review kill-session -t "$SESSION"
        fi
        printf 'The dedicated paired-review session is stopped; review records are retained.\n'
        ;;
    run-web)
        ensure_local_writer_allowed
        cd "$ROOT"
        repeat_command web env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
            /usr/bin/python scripts/stage1/general_model_paired_review.py \
            --reviewer-id liaozijie --session-file "$REVIEW_DIR/session.json" \
            --host 127.0.0.1 --port "$LOCAL_PORT" --public-origin "$PUBLIC_ORIGIN"
        ;;
    run-tunnel)
        repeat_command tunnel /usr/bin/ssh -NT \
            -o BatchMode=yes -o StrictHostKeyChecking=yes \
            -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -o ControlMaster=no -o ControlPath=none \
            -R "127.0.0.1:$REMOTE_PORT:127.0.0.1:$LOCAL_PORT" aliyun
        ;;
    *)
        printf 'Usage: bash %s {start|status|stop}\n' "$SCRIPT" >&2
        exit 2
        ;;
esac
