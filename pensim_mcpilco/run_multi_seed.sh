#!/usr/bin/env bash

#SBATCH --job-name=mcpilco_SWEEP
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Runs the 5-variant cost-shaping sweep for ONE seed, sequentially, in a single job.
#   sbatch run_multi_seed.sh 11
#   sbatch run_multi_seed.sh --seed 11                            # same thing
#   sbatch run_multi_seed.sh --seed 11 --out_root /path/to/full   # override out root
# Any other flag is appended to every run in the sweep, e.g.
#   sbatch run_multi_seed.sh --seed 11 --num_trials 2 --fast

set -uo pipefail

usage() { echo "usage: $0 (<seed> | --seed <seed>) [--out_root <dir>] [extra args...]" >&2; exit 1; }

SEED=""
OUT_ROOT=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)       SEED="${2:-}"; shift 2 || usage ;;
    --seed=*)     SEED="${1#*=}"; shift ;;
    --out_root)   OUT_ROOT="${2:-}"; shift 2 || usage ;;
    --out_root=*) OUT_ROOT="${1#*=}"; shift ;;
    --out_dir|--out_dir=*)
      echo "--out_dir is set per variant by this script; use --out_root instead." >&2; exit 1 ;;
    # first unrecognised flag ends our own parsing: it and everything after it belongs to the
    # training script, values included (otherwise "--num_trials 2" would leak a bare 2 below)
    -*)           EXTRA_ARGS+=("$@"); break ;;
    # bare values before any flag: first is the seed, second the out root (positional form)
    *)            if [[ -z "$SEED" ]]; then SEED="$1"; elif [[ -z "$OUT_ROOT" ]]; then OUT_ROOT="$1"
                  else EXTRA_ARGS+=("$@"); break; fi; shift ;;
  esac
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer, got '${SEED}'" >&2; usage; }

# the sweep: one entry per run, index becomes the out_dir suffix (seed<N>_<i>)
VARIANTS=(
  "--visc_penalty 0.0 --no_harvest_reward --risk_weight 0.0"
  "--visc_penalty 0.5 --no_harvest_reward --risk_weight 0.0"
  "--visc_penalty 0.0 --risk_weight 0.0"
  "--visc_penalty 0.5 --risk_weight 0.0"
  "--visc_penalty 0.5 --risk_weight 0.01"
)
COMMON_ARGS=(--num_trials 11)

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

[[ -n "$OUT_ROOT" ]] || OUT_ROOT="$PROJECT/results/cluster/full"

cd "$PROJECT"
FAILED=()
for i in "${!VARIANTS[@]}"; do
  read -r -a variant <<< "${VARIANTS[$i]}"
  out_dir="$OUT_ROOT/seed${SEED}_${i}"
  args=(--seed "$SEED" --out_dir "$out_dir" "${COMMON_ARGS[@]}" "${variant[@]}" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"})

  echo "=== [$((i + 1))/${#VARIANTS[@]}] $MARKER ${args[*]}"
  # no set -e: one bad variant shouldn't cost us the other four runs in this job
  if ! "$PY" -u "$MARKER" "${args[@]}"; then
    echo "!!! variant $i (seed $SEED) failed" >&2
    FAILED+=("$i")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "seed $SEED: ${#FAILED[@]}/${#VARIANTS[@]} variants failed: ${FAILED[*]}" >&2
  exit 1
fi
echo "seed $SEED: all ${#VARIANTS[@]} variants finished"
