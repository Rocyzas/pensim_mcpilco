import re, sys, ast, numpy as np
from pathlib import Path
sys.path.insert(0,'pensim_mcpilco'); sys.path.insert(0,'.')
RES = Path("/Users/rokaspranevicius/Documents/Aca/UniversityOfEdinburgh/MSc/pensimpy_mcpilco/pensim_mcpilco/results/full")

def parse(note):
    t = note.read_text()
    p = dict(re.findall(r"^(\w+) = (.*)$", t.split("== resolved config ==")[0], re.M))
    def grab(key):
        m = re.search(re.escape(key)+r":\s*array\(\[([^\]]*)\]", t)
        return np.array([float(x) for x in m.group(1).replace("\n"," ").split(",")]) if m else None
    ad = re.search(r"'active_dims': array\(\[([^\]]*)\]", t)
    return p, (np.array([int(x) for x in ad.group(1).split(",")]) if ad else None), \
           grab("'initial_state'"), grab("'std_meas_noise'"), \
           int(re.search(r"'state_dim': (\d+)", t).group(1))

import mcpilco.pensim_wrapper as pw
rows=[]
for note in sorted(RES.rglob("note.txt")):
    p, ad, x0, smn, sd = parse(note)
    single = "single-phase" in str(note)
    if not single:      # dual-phase driver uses a different config module
        rows.append((str(note.parent.relative_to(RES)), "skipped (multi-phase config)", ""))
        continue
    mod = "mcpilco.config_single_phase_baseline_time" if "Added_time" in str(note) \
          else "mcpilco.config_single_phase_baseline"
    import importlib
    get_config = importlib.import_module(mod).get_config
    cfg = get_config(seed=int(p["seed"]), num_trials=int(p["num_trials"]), fast=False,
        optim_horizon_steps=None, num_anchor_batches=0, num_anchors=12, anchor_var=0.01,
        risk_weight=float(p["risk_weight"]), visc_penalty=float(p["visc_penalty"]),
        constraint_strength=float(p["constraint_strength"]),
        harvest_reward=p["harvest_reward"]=="True", num_high_feed_probes=0,
        pms_visc_delay=p["pms_visc_delay"]=="True",
        use_offline_measurements=p["use_offline_measurements"]=="True",
        cost_function=p["cost_function"], num_explorations=int(p["num_explorations"]))
    m=cfg["mc_pilco_init"]; r=cfg["reinforce_par"]
    ok=[]
    ok.append(("state_dim", m["state_dim"]==sd))
    ok.append(("active_dims", ad is not None and np.array_equal(
        np.asarray(m["model_learning_par"]["init_dict_list"][0]["active_dims"]), ad)))
    ok.append(("initial_state", x0 is not None and np.allclose(r["initial_state"], x0, atol=1e-7)))
    ok.append(("std_meas_noise", smn is not None and np.allclose(m["std_meas_noise"], smn, atol=1e-9)))
    bad=[k for k,v in ok if not v]
    rows.append((str(note.parent.relative_to(RES)), "MATCH" if not bad else "MISMATCH: "+",".join(bad),
                 f"dim={sd} cost={p['cost_function']}"))
for n,s,e in rows: print(f"  {s:32s} {n:52s} {e}")
print()
mm=[r for r in rows if r[1].startswith("MISMATCH")]
print(f"{sum(1 for r in rows if r[1]=='MATCH')} matched, {len(mm)} mismatched, "
      f"{sum(1 for r in rows if 'skipped' in r[1])} skipped")
