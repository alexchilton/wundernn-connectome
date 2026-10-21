#!/bin/bash
# Colab watchdog: every 5 min, check that each named session is still listed. Exits (so the caller is notified)
# the first time one disappears. Log: ~/PycharmProjects/wundernn-connectome/logs/colab_watchdog.log
L=~/PycharmProjects/wundernn-connectome/logs/colab_watchdog.log
while true; do
  live=$(timeout 60 ~/.local/bin/colab sessions < /dev/null 2>&1 | grep -o '^\[[a-z0-9]*\]' | tr -d '[]' | tr '\n' ' ')
  echo "$(date +%H:%M) live: $live" >> $L
  for s in "$@"; do
    case " $live " in *" $s "*) ;; *) echo "$(date +%H:%M) SESSION GONE: $s (live: $live)" | tee -a $L; exit 1;; esac
  done
  sleep 300
done
