#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
[[ ! -f env.local ]] || source env.local
export CROWN_PYTHON="${CROWN_PYTHON:-python}"
: "${CROWN_VOXEL_PYTHON:?Set CROWN_VOXEL_PYTHON in env.local to the dedicated Open3D interpreter}"
export CROWN_VOXEL_PYTHON
case "${1:-local}" in
 local) exec ./app/web_demo/start.sh ;;
 public) exec ./app/web_demo/start_public.sh ;;
 *) echo 'Usage: ./start.sh [local|public]' >&2; exit 2 ;;
esac
