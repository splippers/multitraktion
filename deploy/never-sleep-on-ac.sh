#!/usr/bin/env bash
# Keeps a laptop render node awake with the lid closed while on mains power.
# On battery the lid still suspends. Run as root: sudo bash never-sleep-on-ac.sh
set -euo pipefail
install -d /etc/systemd/logind.conf.d
cat > /etc/systemd/logind.conf.d/10-traktion-lid.conf <<'CONF'
[Login]
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
CONF
systemctl restart systemd-logind
systemd-analyze cat-config systemd/logind.conf | grep -E "HandleLidSwitch(ExternalPower|Docked)"
