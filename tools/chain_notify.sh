#!/bin/bash
# Exit (and so notify the session that started it in the background) when a chain milestone appears in the chain log
# after this script starts: Drive mounted, smoke ok / SMOKE FAILED, an arm done or FAILED, chain DONE.
# Usage: tools/chain_notify.sh [log]   - restart it after each notification to wait for the next milestone.
L=${1:-$HOME/PycharmProjects/wundernn-connectome/logs/overnight_final.log}
n=$(wc -l < "$L")
while sleep 30; do
  hit=$(tail -n +$((n + 1)) "$L" | grep -m1 -E "Drive mounted|smoke ok|SMOKE FAILED| done: best|FAILED|DONE")
  [ -n "$hit" ] && { echo "$hit"; exit 0; }
done
