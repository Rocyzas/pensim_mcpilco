#!/bin/bash
# Runs commands from python_command_runner.txt, 2 at a time, always keeping 2 running.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcpilco

max_jobs=2

while IFS= read -r cmd; do
  [[ -z "${cmd// }" ]] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
    sleep 1
  done
  echo "[$(date +%T)] START: $cmd"
  ( eval "$cmd"; echo "[$(date +%T)] DONE:  $cmd" ) &
done < python_command_runner.txt

wait
