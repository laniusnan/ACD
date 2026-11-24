#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$PACKAGE/../.." && pwd)"
cd "$PROJECT"
mkdir -p "$PACKAGE/artifacts"
exec >>"$PACKAGE/artifacts/launcher.log" 2>&1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
printf 'started_at=%s launcher_pid=%s\n' "$(date -u +%FT%TZ)" "$$"
"$PROJECT/.venv/bin/python" "$PACKAGE/prepare.py"
"$PROJECT/.venv/bin/python" -m unittest discover -s "$PACKAGE" -p 'test_*.py' -v
if [ ! -f "$PACKAGE/artifacts/seal.json" ]; then
  "$PROJECT/.venv/bin/python" -u "$PACKAGE/run.py" preflight
fi
exec "$PROJECT/.venv/bin/python" -u "$PACKAGE/run.py" dispatch
