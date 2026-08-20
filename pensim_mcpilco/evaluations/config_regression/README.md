# Config regression check against stored runs

Regenerates each stored run's config from the parameters recorded in its own `note.txt`
and diffs it against the config that `note.txt` actually recorded. Catches the case where
a code change silently alters what an already-published experiment would produce.

Compared fields: `state_dim`, `active_dims`, `initial_state`, `std_meas_noise`
(+ `pivot_step` for dual-phase).

`initial_state` is the load-bearing one: `_measure_init_state_stats()` MEASURES it by
rolling 6 pure-recipe batches through PenSimPy up to K_WARM, so matching it to 1e-7
exercises the simulator, the recipe, STATE_RANGES and the encoding — not just constants.

    PYTHONPATH=.. python evaluations/config_regression/check_single_phase_notes.py
    PYTHONPATH=.. python evaluations/config_regression/check_dual_phase_notes.py

Last run: 2026-08-08, all 24 runs under results/full/ (ConcCost + MassCost, seeds 4/5/6,
single- and multi-phase, No_time and Added_time) MATCH, with the CER state-channel change
(pensim_wrapper.set_state_names + STATE_RANGES["CER"]) applied.

CAVEAT: this verifies CONFIG-level agreement, not that a full training run reproduces its
stored yields end to end. That would take hours per run and would be testing every code
change since 2026-08-05, not just the one under review.
