#!/usr/bin/env bash
set -euo pipefail

# ==== adjust if needed ====
DPU_IF="tmfifo_net0"
DPU_IP="fd00:1::2"
HOST_GW="fd00:1::1"
DNS1="2001:4860:4860::8888"
DNS2="2001:4860:4860::8844"
# ==========================

echo "[1/4] Assign IPv6 to DPU interface (idempotent)"
ip -6 addr show dev "$DPU_IF" | grep -q "$DPU_IP" || \
  ip -6 addr add "${DPU_IP}/64" dev "$DPU_IF"

echo "[2/4] Make host the preferred default route"
ip -6 route replace default via "$HOST_GW" dev "$DPU_IF" metric 1

echo "[3/4] (Optional) stop stray RAs from adding other defaults"
for IF in $(ls /sys/class/net); do
  [[ "$IF" == "$DPU_IF" ]] && continue
  sysctl -w "net.ipv6.conf.${IF}.accept_ra=0" >/dev/null || true
done

echo "[4/4] Configure IPv6 DNS now (simple resolv.conf write)"
# If systemd-resolved manages DNS, consider 'resolvectl dns tmfifo_net0 ...'
cat >/etc/resolv.conf <<EOF
nameserver ${DNS1}
nameserver ${DNS2}
EOF

echo "Done. Quick tests:"
echo "  ping -6 -c 3 ${HOST_GW}"
echo "  ping -6 -c 3 2001:4860:4860::8888"
echo "  ping -6 -c 3 ipv6.google.com"
