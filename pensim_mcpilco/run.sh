#!/usr/bin/env bash

#SBATCH --job-name=mcpilco_FULL
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

set -euo pipefail

# default args, used only when none are passed on the command line
DEFAULT_ARGS=(--num_trials 12)

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

ARGS=("$@")
[[ ${#ARGS[@]} -gt 0 ]] || ARGS=("${DEFAULT_ARGS[@]}")

cd "$PROJECT"
echo "running: $MARKER ${ARGS[*]}"
exec "$PY" -u "$MARKER" "${ARGS[@]}"
