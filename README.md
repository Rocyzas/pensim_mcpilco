TODO:

Decide on:
1. Observation, State space
- State: a reduced (to make FP tractable) and physically-motivated (biomass, substrate, dissolved oxygen, pH, viscosity, penicillin conc, plus phase indicator), according to the Goldrick's mechanistic models (IndPenSim)
- Answers: "What dimensionality reduction tradeoffs (in terms of total yield or cost) in GP compared to DRL models"

2. Action space
- what's actually controllable on a real plant at the timescale Im modeling (pH and temperature regulated by PIDs)
- probably will need: substrate feed rate and aeration/agitation.

3. For multi-phase use same observational state but control differnt actions (see and justify which actions are more important during each phase).
- not confirmed, but: 
    - Growth (biomass accumulation) - substrate feed rate, aeration, and temperature/pH setpoints 
    - Production (additional levers like precursor (e.g., phenylacetic acid) feed rate), and substrate feed rate is not as important (not confirmed)

GENERALISABILITY
4. Decide on what is structural vs paramteric
- Structural: phase detector, per phase dynamics GP models, policy(s)
- Parametric (organism specific): bounds, different variables, initial conditions, reward weighting

*genelaise how to obtain the parameters and not the parameters themselves* 

- find a second simple organism to show generalisability
Question - which actions/states to chose?


PHASE SWITCH
1. unsupervised change-point detection on the dynamics



# Variables
Raman enables (3 per paper): PAA (phenylacetic acid), X, P concentration.
    + Substrate (S) and Viscosity (vis).

1. PID controlled: pH, Temperature, PAA concentration (disabling and controlling as RL agent action)
    PAA concentration is on only when Raman is on, so I am disabling the `bypass_paa_pid` flag for now and controlling it manually.

2. Observable without Raman: 
- ONLINE: T, ph, DO2, O2, CO2outgas, Wt (vessel wright), pressure, agitator RPM, all flow rates
- OFFLINE: P, X, PAA, NH3, Viscosity (without raman observed with Lab samples only every 12h.)

3. Actions
- Fs (sugar feed), Foil, Fg (aeration feed), head pressure, F_discharge, Fw (water for dillution), Fpaa (PAA feed), NH3_shots (ammonia shots)

# Decision on variables with justification
ACTIONS:
1. BO baseline uses 6: DISCHARGE, FS, FOIL, FG, PRES, WATER, FPAA (excluded)



<!-- COMPARISON  -  SEEDS -->
Comparison is made in the @03_compare ipynb notebook

1. Agent RNG seed (torch/numpy, global RNG) - controls learning algorithm's randomness, which is policy weight initialisation, particle sampling, dropout
2. Batch seed (PenSimEnv random_seed_ref) - physical fermentation realisation, which is initial conditions, kinetics, disturbances.



FLAGS TO REMOVE:
1. Fixed seed on the rollout()
2. T_SAMPLING 5.0
3. 


THINGS IMPLEMENTED:
1. Initial explorations ignore the failed batches. Although failed batches due to control are okay for GP inputs, the ones that failed due to physics should be discarded (such as Vis>100 etc). Some failures are detected in real time using multivariate statistical process control (as per paper).
2. Included ONLY states that are affected by my Fs control: X, P, Wt, (S is not available). And added time.
3. X, P, Wt are clamped, log-encoded and then normalised.



sbatch run.sh --visc_penalty 0.0 --no_harvest_reward --risk_weight 0.0 --seed 11 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/seed11_0 --num_trials 11

sbatch run.sh --visc_penalty 0.5 --no_harvest_reward --risk_weight 0.0 --seed 11 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/seed11_1 --num_trials 11

sbatch run.sh --visc_penalty 0.0 --risk_weight 0.0 --seed 11 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/seed11_2 --num_trials 11

sbatch run.sh --visc_penalty 0.5 --risk_weight 0.0 --seed 11 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/seed11_3 --num_trials 11

sbatch run.sh --visc_penalty 0.5 --risk_weight 0.01 --seed 11 --out_dir /home/s2889898/Diss/pensim_mcpilco/pensim_mcpilco/results/cluster/full/seed11_4 --num_trials 11


seed3_5 - rbf of V,X,P
seed2_32 - rbf on none
seed3_6 - rbf on none, num_min_diff_cost=25
seed3_7 - rbf on V,X,P, num_min_diff_cost=25 - very good yield, only one -600, --risk_weight 0.01
seed3_8 - rbf on V,X,P(fix), num_min_diff_cost=25 + --num_high_feed_probes 3 --risk_weight 0.01 - decreased yield
finding out what decreased yield running, 

seed3_12 - rbf on V,X,P(fix), num_min_diff_cost=25 --risk_weight 0.01
seed2_33 (same as above but different seed)
seed2_33 (same as above but different cost function of PeniMassChangeCost)

seed3_13 - rbf on V,X,P(fix), num_min_diff_cost=25 --risk_weight 0.01 --num_high_feed_probes 3 cancelled



Added DO2
seed3_9 - rbf on V,X,P(fix), num_min_diff_cost=25 + --num_high_feed_probes 3


PLAN
- Train seed3_12, and observer result (only difference is probes). 
    If if keeps the yield as good as seed3_7, REMOVE num_high_feed_probes as they are not needed for good yield
    If yield decreases it means that P(fix) did not help.
        Then try without P(fix) and with num_high_feed_probes