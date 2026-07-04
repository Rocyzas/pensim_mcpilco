#!/usr/bin/env bash
#
# Run the single-phase MC-PILCO baseline, then its analysis/plots.
# Self-contained: creates a Python venv (if missing), installs deps into it, and
# runs everything with THAT venv's python -- so it does not matter what `python`
# is on PATH (system python, no conda, etc.).
#
#   1) experiments/02_mcpilco_single_phase.py   -> trains, writes results/single_phase/seed{SEED}/
#   2) experiments/analyze_single_phase.py      -> reads that dir, writes the summary figure
#
# analyze runs only if training succeeds. SEED is shared so analyze reads the
# batch that was just produced (--seeds SEED, --recipe_batch SEED).
#
# Usage:
#   ./run_single_phase.sh                 # seed=1, num_trials=10 (creates venv first time)
#   ./run_single_phase.sh --seed 2 --num_trials 5 --fast
#   ./run_single_phase.sh --setup-only    # just build the venv, don't run
#   ./run_single_phase.sh --recreate      # delete & rebuild the venv, then run
#
# Env overrides:
#   VENV_DIR=/path/to/venv     where the venv lives        (default: <project>/.venv)
#   BASE_PYTHON=python3.11     interpreter used to BUILD the venv (default: python3)
#   PENSIM_ROOT=/path          the pensim_mcpilco dir (auto-detected otherwise)
#
set -euo pipefail

# --- config ---
SEED="${SEED:-1}"
NUM_TRIALS="${NUM_TRIALS:-10}"
FAST="${FAST:-0}"
SETUP_ONLY=0
RECREATE=0
BASE_PYTHON="${BASE_PYTHON:-python3}"

# --- parse flags ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)        SEED="$2"; shift 2 ;;
    --num_trials)  NUM_TRIALS="$2"; shift 2 ;;
    --fast)        FAST=1; shift ;;
    --setup-only)  SETUP_ONLY=1; shift ;;
    --recreate)    RECREATE=1; shift ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --- locate the pensim_mcpilco dir (contains experiments/). Prefer $PENSIM_ROOT,
#     then $PWD, then this script's dir. BASH_SOURCE is unreliable under SLURM
#     (sbatch runs a spooled copy), so it is the last resort. ---
_marker="experiments/02_mcpilco_single_phase.py"
ROOT_DIR=""
for _cand in "${PENSIM_ROOT:-}" "$PWD" "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)/.."; do
  if [[ -n "$_cand" && -f "$_cand/$_marker" ]]; then
    ROOT_DIR="$(cd "$_cand" && pwd)"; break
  fi
done
if [[ -z "$ROOT_DIR" ]]; then
  echo "Can't find the pensim_mcpilco dir (no $_marker). cd into it or set PENSIM_ROOT=/path/to/pensim_mcpilco." >&2
  exit 1
fi
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PY="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.deps_installed"   # marker so we don't reinstall every run

# --- build the venv + install deps (idempotent) ---
if [[ "$RECREATE" == "1" ]]; then
  echo ">>> --recreate: removing $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -x "$PY" ]]; then
  echo ">>> Creating venv at $VENV_DIR (base: $BASE_PYTHON)"
  command -v "$BASE_PYTHON" >/dev/null || { echo "BASE_PYTHON '$BASE_PYTHON' not found on PATH." >&2; exit 1; }
  "$BASE_PYTHON" -m venv "$VENV_DIR"
fi

if [[ ! -f "$STAMP" ]]; then
  echo ">>> Installing dependencies into the venv (first run only)"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install \
    numpy pandas scipy matplotlib \
    torch gpytorch scikit-learn scikit-optimize
  # fastodeint: imported by peni_env_setup.py but its integrate() is monkey-patched
  # away by utils/ode_patch.py (pure-SciPy LSODA), so a stub module is enough.
  SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  printf 'def integrate(*a, **k):\n    raise RuntimeError("fastodeint stub: should be patched by utils.ode_patch")\n' \
    > "$SITE/fastodeint.py"
  touch "$STAMP"
  echo ">>> Dependencies installed."
fi

if [[ "$SETUP_ONLY" == "1" ]]; then
  echo ">>> --setup-only: venv ready at $VENV_DIR"; exit 0
fi

FAST_FLAG=()
[[ "$FAST" == "1" ]] && FAST_FLAG=(--fast)

echo "=========================================================="
echo " venv:       $VENV_DIR"
echo " python:     $PY"
echo " cwd:        $ROOT_DIR"
echo " seed:       $SEED"
echo " num_trials: $NUM_TRIALS"
echo " fast:       $FAST"
echo "=========================================================="

echo
echo ">>> [1/2] Training: 02_mcpilco_single_phase.py"
"$PY" experiments/02_mcpilco_single_phase.py \
  --seed "$SEED" --num_trials "$NUM_TRIALS" "${FAST_FLAG[@]}"

echo
echo ">>> [2/2] Analysis: analyze_single_phase.py"
"$PY" experiments/analyze_single_phase.py \
  --seeds "$SEED" --recipe_batch "$SEED"

echo
echo ">>> Done. See results/single_phase/seed${SEED}/ and results/single_phase/aggregate/"
