#!/usr/bin/env bash
# Copy this file together with dual_tunnel.py to the new Linux container.
set -euo pipefail
if ! command -v python3 >/dev/null 2>&1; then
    echo '缺少 python3。Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y python3' >&2
    exit 1
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/dual_tunnel.py" "$@"
