#!/usr/bin/env bash
set -euo pipefail

# ==== adjust if needed ====
WAN_IF="enp1s0"          # host's WAN uplink
DPU_IF="tmfifo_net0"     # host<->DPU link (host side)
ULA_SUBNET="fd00:1::/64"
HOST_INSIDE_IP="fd00:1::1"
# ==========================

echo "[1/5] Enable IPv6 forwarding"
sysctl -w net.ipv6.conf.all.forwarding=1 >/dev/null

echo "[2/5] Add host inside IPv6 (idempotent)"
ip -6 addr show dev "$DPU_IF" | grep -q "$HOST_INSIDE_IP" || \
  ip -6 addr add "${HOST_INSIDE_IP}/64" dev "$DPU_IF"

echo "[3/5] Allow forwarding (idempotent)"
ip6tables -C FORWARD -i "$DPU_IF" -o "$WAN_IF" -j ACCEPT 2>/dev/null || \
  ip6tables -A FORWARD -i "$DPU_IF" -o "$WAN_IF" -j ACCEPT

ip6tables -C FORWARD -i "$WAN_IF" -o "$DPU_IF" -m state --state ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || \
  ip6tables -A FORWARD -i "$WAN_IF" -o "$DPU_IF" -m state --state ESTABLISHED,RELATED -j ACCEPT

# ICMPv6 (ND/PMTUD) – keep open
ip6tables -C INPUT -p ipv6-icmp -j ACCEPT 2>/dev/null || \
  ip6tables -A INPUT -p ipv6-icmp -j ACCEPT
ip6tables -C FORWARD -p ipv6-icmp -j ACCEPT 2>/dev/null || \
  ip6tables -A FORWARD -p ipv6-icmp -j ACCEPT

echo "[4/5] Enable NAT66 (masquerade ULA out WAN) – idempotent"
modprobe ip6table_nat || true
modprobe nf_nat || true
ip6tables -t nat -C POSTROUTING -s "$ULA_SUBNET" -o "$WAN_IF" -j MASQUERADE 2>/dev/null || \
  ip6tables -t nat -A POSTROUTING -s "$ULA_SUBNET" -o "$WAN_IF" -j MASQUERADE

echo "[5/5] Show summary"
ip -6 addr show dev "$DPU_IF"
echo
ip6tables -nvL
echo
ip6tables -t nat -nvL