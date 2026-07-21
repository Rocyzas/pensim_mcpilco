#!/usr/bin/env bash

#SBATCH --job-name=mcpilco_07_full
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

ARGS=(--seed 20 --num_trials 7 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/seed20_0)

MARKER="experiments/02_mcpilco_single_phase.py"

PROJECT=""
for base in "${PENSIM_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD" \
            "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
  [[ -z "$base" ]] && continue
  for sub in "." "pensim_mcpilco"; do
    if [[ -f "$base/$sub/$MARKER" ]]; then
      PROJECT="$(cd "$base/$sub" && pwd)"; break 2
    fi
  done
done
[[ -n "$PROJECT" ]] || { echo "Can't find $MARKER. Set PENSIM_ROOT." >&2; exit 1; }

PY=""
for cand in "${VENV_DIR:-}" "$PROJECT/.venv" "$PROJECT/../.venv"; do
  [[ -n "$cand" && -x "$cand/bin/python" ]] && { PY="$(cd "$(dirname "$cand")" && pwd)/$(basename "$cand")/bin/python"; break; }
done
[[ -n "$PY" ]] || { echo "No venv found. Set VENV_DIR." >&2; exit 1; }

OUTER="$(dirname "$PROJECT")"
[[ -d "$OUTER/PenSimPy/pensimpy" ]] || { echo "PenSimPy not found at $OUTER/PenSimPy." >&2; exit 1; }
export PYTHONPATH="$OUTER:$OUTER/PenSimPy${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
exec "$PY" -u experiments/02_mcpilco_single_phase.py "${ARGS[@]}" "$@"