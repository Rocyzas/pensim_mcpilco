#!/usr/bin/env bash

#SBATCH --job-name=mcpilco_SWEEP
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Runs a sweep of FULL command lines, sequentially, in a single job. Each entry in VARIANTS
# below is pasted exactly as you'd type it yourself, e.g.
#   "python -m experiments.02_mcpilco_single_phase_baseline --seed {SEED} --num_trials 11 --fast"
#   "python -m experiments.03_mcpilco_dual_phase_baseline_time --seed {SEED} --num_trials 11 --t_sampling 2"
# A leading "python"/"python3[.x]" (and a following -u) is swapped for the resolved venv
# interpreter automatically -- everything else in the line passes through untouched, quoting
# included. --out_dir is auto-appended ($OUT_ROOT/seed<SEED>_<index>, or just run<index> if
# --seed wasn't given) UNLESS the line already has its own --out_dir.
#
# Bare number arguments select WHICH variants to run, 1-INDEXED into VARIANTS below (not a
# seed!) -- e.g. `sbatch run_multi_seed.sh 1 2 3` runs only VARIANTS[0..2]. Omit them to run
# every variant in the list.
#   sbatch run_multi_seed.sh              # run ALL variants
#   sbatch run_multi_seed.sh 1 2 3        # run only the 1st, 2nd, 3rd variants
#   sbatch run_multi_seed.sh --seed 11 1 2 3            # same, with {SEED} substituted as 11
#   sbatch run_multi_seed.sh --seed 11 --out_root /path/to/full 2
# Any other flag is appended to EVERY selected variant, e.g.
#   sbatch run_multi_seed.sh --fast 1 2

set -uo pipefail

usage() { echo "usage: $0 [--seed <seed>] [--out_root <dir>] [variant# ...] [extra args...]" >&2; exit 1; }

SEED=""
OUT_ROOT=""
EXTRA_ARGS=()
SELECT=()   # 1-indexed variant selectors, e.g. (1 2 3) -- filled in below, resolved after
            # VARIANTS is defined (bare numbers select WHICH variants to run, NOT a seed)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)       SEED="${2:-}"; shift 2 || usage ;;
    --seed=*)     SEED="${1#*=}"; shift ;;
    --out_root)   OUT_ROOT="${2:-}"; shift 2 || usage ;;
    --out_root=*) OUT_ROOT="${1#*=}"; shift ;;
    --out_dir|--out_dir=*)
      echo "--out_dir is set per variant by this script; use --out_root instead." >&2; exit 1 ;;
    # first unrecognised (non-numeric) flag ends our own parsing: it and everything after it
    # belongs to the training script, values included (otherwise "--num_trials 2" would leak a
    # bare 2 into SELECT below)
    -*)           EXTRA_ARGS+=("$@"); break ;;
    # bare numbers: which variants to run (1-indexed into VARIANTS, validated once VARIANTS is
    # defined below, since we don't know its length yet here)
    *)            SELECT+=("$1"); shift ;;
  esac
done
[[ -z "$SEED" || "$SEED" =~ ^[0-9]+$ ]] || { echo "--seed must be an integer, got '${SEED}'" >&2; usage; }

# the sweep: one FULL command line per run (see the header comment above), index becomes the
# out_dir suffix (seed<N>_<i>, or run<i> if --seed wasn't given). Example:
#   VARIANTS=(
#     "python -m experiments.02_mcpilco_single_phase_baseline --seed {SEED} --num_trials 11"
#     "python -m experiments.03_mcpilco_dual_phase_baseline --seed {SEED} --num_trials 11 --pivot_hours 80"
#   )
VARIANTS=(
  # "python experiments/02_mcpilco_single_phase_baseline.py --seed {SEED} --num_trials 10 --cost_function PeniConcentrationDenseCost --t_sampling 5 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/ConcCost/single_phase/No_time/seed{SEED}_0"
  # "python experiments/02_mcpilco_single_phase_baseline_time.py --seed {SEED} --num_trials 10 --cost_function PeniConcentrationDenseCost --t_sampling 5 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/ConcCost/single_phase/Added_time/seed{SEED}_1"
  "python experiments/03_mcpilco_dual_phase_baseline.py --seed {SEED} --num_trials 10 --num_explorations 10 --cost_function PeniConcentrationDenseCost --t_sampling 5 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/ConcCost/dual_phase/No_time/10expl/seed{SEED}_2"
  "python experiments/03_mcpilco_dual_phase_baseline_priors.py --seed {SEED} --num_trials 10 --cost_function PeniConcentrationDenseCost --t_sampling 5 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/ConcCost/dual_phase/No_time/priors/seed{SEED}_2"
  "python experiments/03_mcpilco_dual_phase_baseline.py --seed {SEED} --num_trials 10 --cost_function PeniConcentrationDenseCost --num_high_feed_probes --num_explorations 1 --t_sampling 5 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/ConcCost/dual_phase/No_time/probes/seed{SEED}_2"
  # "python experiments/03_mcpilco_dual_phase_baseline_time.py --seed {SEED} --num_trials 10 --cost_function PeniConcentrationDenseCost --t_sampling 5 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/ConcCost/dual_phase/Added_time/seed{SEED}_3"
)

# Appended to EVERY variant above, after its own args -- optional, for flags you don't want to
# repeat in every pasted line (most people will just put everything in VARIANTS directly and
# leave this empty).
COMMON_ARGS=()

# Resolve SELECT (1-indexed, from the CLI) into INDICES (0-indexed, into VARIANTS) now that we
# know how long VARIANTS actually is. No selectors given -> run everything, same as before.
INDICES=()
if [[ ${#SELECT[@]} -eq 0 ]]; then
  INDICES=("${!VARIANTS[@]}")
else
  for n in "${SELECT[@]}"; do
    if ! [[ "$n" =~ ^[0-9]+$ ]] || (( n < 1 || n > ${#VARIANTS[@]} )); then
      echo "invalid variant selector '$n': must be an integer between 1 and ${#VARIANTS[@]}" >&2
      exit 1
    fi
    INDICES+=($((n - 1)))
  done
fi

# Only used to LOCATE the project root below (does a candidate path look like pensim_mcpilco?)
# -- unrelated to which experiment script(s) VARIANTS actually runs. Points at a core module
# rather than any one experiments/*.py driver, so it stays valid regardless of which variants
# (single/dual-phase, baseline, _time, ...) you're running, and won't break if a specific
# experiment script is ever renamed or removed.
MARKER="mcpilco/pensim_wrapper.py"

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
for i in "${INDICES[@]}"; do
  variant="${VARIANTS[$i]}"
  [[ -z "${variant//[[:space:]]/}" ]] && continue   # skip blank entries

  if [[ "$variant" == *"{SEED}"* && -z "$SEED" ]]; then
    echo "!!! variant $((i + 1)) uses {SEED} but no --seed was given -- skipping" >&2
    FAILED+=("$i")
    continue
  fi
  variant="${variant//\{SEED\}/$SEED}"

  # Tokenise the pasted line with bash's own quote/word-splitting rules (via array-literal
  # assignment, not a plain `read`, so quoted values with spaces survive intact), same as if
  # you'd typed it at a prompt.
  eval "variant_words=($variant)"

  # Drop a leading python/python3/python3.x (and a following -u): we run everything through OUR
  # resolved venv interpreter below instead, so a literal "python" token here would otherwise be
  # passed as a bogus first ARGUMENT to that interpreter rather than naming one of its own.
  if [[ "${variant_words[0]:-}" =~ ^python[0-9.]*$ ]]; then
    variant_words=("${variant_words[@]:1}")
  fi
  if [[ "${variant_words[0]:-}" == "-u" ]]; then
    variant_words=("${variant_words[@]:1}")
  fi

  args=("${variant_words[@]}" ${COMMON_ARGS+"${COMMON_ARGS[@]}"})
  # only inject our own out_dir if this line didn't already bring one
  if [[ "$variant" != *"--out_dir"* ]]; then
    if [[ -n "$SEED" ]]; then suffix="seed${SEED}_${i}"; else suffix="run${i}"; fi
    args+=(--out_dir "$OUT_ROOT/$suffix")
  fi
  args+=(${EXTRA_ARGS+"${EXTRA_ARGS[@]}"})

  echo "=== [variant $((i + 1))/${#VARIANTS[@]}] $PY -u ${args[*]}"
  # no set -e: one bad variant shouldn't cost us the other runs in this job
  if ! "$PY" -u "${args[@]}"; then
    echo "!!! variant $((i + 1)) failed" >&2
    FAILED+=("$i")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "${#FAILED[@]}/${#INDICES[@]} selected variant(s) failed: ${FAILED[*]}" >&2
  exit 1
fi
echo "all ${#INDICES[@]} selected variant(s) finished"
