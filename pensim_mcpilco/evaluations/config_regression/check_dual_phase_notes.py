import re, sys, importlib, numpy as np
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'..')
RES = Path("/Users/rokaspranevicius/Documents/Aca/UniversityOfEdinburgh/MSc/pensimpy_mcpilco/pensim_mcpilco/results/full")

def grab(t, key):
    m = re.search(re.escape(key)+r":\s*array\(\[([^\]]*)\]", t)
    return np.array([float(x) for x in m.group(1).replace("\n"," ").split(",")]) if m else None

rows=[]
for note in sorted(RES.rglob("note.txt")):
    if "multi-phase" not in str(note): continue
    t = note.read_text()
    p = dict(re.findall(r"^([\w_]+) = (.*)$", t.split("== resolved config ==")[0], re.M))
    mod = "mcpilco.config_dual_phase_baseline_time" if "Added_time" in str(note) \
          else "mcpilco.config_dual_phase_baseline"
    get_config = importlib.import_module(mod).get_config
    kw = dict(seed=int(p["seed"]), num_trials=int(p["num_trials"]), fast=False,
        risk_weight=float(p["risk_weight"]), visc_penalty=float(p["visc_penalty"]),
        constraint_strength=float(p["constraint_strength"]),
        harvest_reward=p["harvest_reward"]=="True",
        pms_visc_delay=p["pms_visc_delay"]=="True",
        use_offline_measurements=p["use_offline_measurements"]=="True",
        cost_function=p["cost_function"], num_explorations=int(p["num_explorations"]),
        pivot_hours=float(p["pivot_hours"]),
        blend_half_width_hours=float(p["blend_half_width_hours"]))
    cfg = get_config(**kw); m=cfg["mc_pilco_init"]; r=cfg["reinforce_par"]
    mlp = m["model_learning_par"]
    sd = int(re.search(r"'state_dim': (\d+)", t).group(1))
    ps = int(re.search(r"'pivot_step': (\d+)", t).group(1))
    ad = re.search(r"'active_dims': array\(\[([^\]]*)\]", t)
    ad = np.array([int(x) for x in ad.group(1).split(",")]) if ad else None
    checks = {
      "state_dim":     m["state_dim"]==sd,
      "pivot_step":    mlp["pivot_step"]==ps,
      "active_dims":   ad is not None and np.array_equal(
                         np.asarray(mlp["phase1_par"]["init_dict_list"][0]["active_dims"]), ad),
      "initial_state": np.allclose(r["initial_state"], grab(t,"'initial_state'"), atol=1e-7),
      "std_meas_noise":np.allclose(m["std_meas_noise"], grab(t,"'std_meas_noise'"), atol=1e-9),
    }
    bad=[k for k,v in checks.items() if not v]
    rows.append((str(note.parent.relative_to(RES)), "MATCH" if not bad else "MISMATCH: "+",".join(bad),
                 f"dim={sd} pivot_step={ps} cost={p['cost_function']}"))
for n,s,e in rows: print(f"  {s:24s} {n:48s} {e}")
print(f"\n{sum(1 for r in rows if r[1]=='MATCH')} matched, "
      f"{sum(1 for r in rows if r[1].startswith('MISMATCH'))} mismatched, of {len(rows)}")
