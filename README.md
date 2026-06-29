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