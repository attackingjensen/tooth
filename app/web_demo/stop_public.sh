#!/usr/bin/env bash
# 2026-09-23：停止本项目记录的网页和隧道进程。
set -euo pipefail
cd -- "$(dirname -- "$0")"
for name in tunnel server; do
  pid_file="$PWD/private/$name.pid"
  if [[ -s "$pid_file" ]]; then
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
    fi
  fi
done
