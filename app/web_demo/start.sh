#!/usr/bin/env bash
# 2026-09-23：以单进程 Waitress 启动网页和单 GPU 队列。
set -euo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${CROWN_PYTHON:-python}" -u app.py --host "${CROWN_HOST:-127.0.0.1}" --port "${CROWN_PORT:-8090}"
