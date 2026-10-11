#!/usr/bin/env bash
# 2026-09-23：启动本机网页与 Cloudflare 临时 HTTPS 隧道，并记录公网地址。
set -euo pipefail
cd -- "$(dirname -- "$0")"

private_dir="$PWD/private"
mkdir -p "$private_dir"
chmod 700 "$private_dir"

if [[ -s "$private_dir/server.pid" ]] && kill -0 "$(<"$private_dir/server.pid")" 2>/dev/null; then
  echo "Web service is already running (PID $(<"$private_dir/server.pid"))."
else
  # setsid keeps the service alive when the launching shell/session is reclaimed.
  CROWN_HOST=127.0.0.1 CROWN_PORT=8090 CROWN_GPUS="${CROWN_GPUS:-auto}" CROWN_MAX_WORKERS="${CROWN_MAX_WORKERS:-2}" CROWN_HTTPS=1 \
    setsid -f "$PWD/start.sh" >> "$private_dir/service.log" 2>&1 < /dev/null
  for _ in $(seq 1 30); do
    [[ -s "$private_dir/server.pid" ]] && kill -0 "$(<"$private_dir/server.pid")" 2>/dev/null && break
    sleep 1
  done
fi

if [[ -s "$private_dir/tunnel.pid" ]] && kill -0 "$(<"$private_dir/tunnel.pid")" 2>/dev/null; then
  echo "Tunnel is already running (PID $(<"$private_dir/tunnel.pid"))."
else
  : > "$private_dir/tunnel.log"
  nohup "$PWD/bin/cloudflared" tunnel --no-autoupdate --url http://127.0.0.1:8090 \
    >> "$private_dir/tunnel.log" 2>&1 < /dev/null &
  echo "$!" > "$private_dir/tunnel.pid"
fi

public_url=""
for _ in $(seq 1 45); do
  public_url="$(sed -nE 's|.*(https://[-a-z0-9]+\.trycloudflare\.com).*|\1|p' "$private_dir/tunnel.log" | tail -1)"
  [[ -n "$public_url" ]] && break
  sleep 1
done
if [[ -z "$public_url" ]]; then
  echo "Tunnel did not publish a URL. See $private_dir/tunnel.log" >&2
  exit 1
fi
printf '%s\n' "$public_url" > "$private_dir/public_url.txt"
chmod 600 "$private_dir/public_url.txt"
echo "$public_url"
