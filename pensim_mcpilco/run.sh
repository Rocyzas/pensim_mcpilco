#!/usr/bin/env bash
set -euo pipefail

ARGS=(--seed 0 --num_trials 7)

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
