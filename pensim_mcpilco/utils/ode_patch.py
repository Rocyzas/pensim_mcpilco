import math
from scipy.integrate import solve_ivp
from PenSimPy.pensimpy.ode.indpensim_ode_py import indpensim_ode_py

_CLIP = 0.001  # matches PenSimPy's own >0 clipping (peni_env_setup line 219)

def _lsoda_integrate(y0, par, t_start, t_end, h):
    # PenSimPy calls integrate with t_end = nominal_end + h
    t_stop = t_end - h
    if t_stop <= t_start + 1e-12:
        return list(y0)

    y0 = [v if (math.isfinite(v) and v > 0) else _CLIP for v in y0]
    last = [0.0] * len(y0)

    def f(t, y):
        nonlocal last
        ys = [v if (math.isfinite(v) and v > 0) else _CLIP for v in y]
        try:
            last = list(indpensim_ode_py(t, ys, par))
            return last
        except (OverflowError, ValueError, ZeroDivisionError):
            return last  # coast on the last valid derivative, don't crash or freeze

    sol = solve_ivp(f, (t_start, t_stop), y0, method="LSODA",
                    max_step=h, rtol=1e-6, atol=1e-8)
    return sol.y[:, -1].tolist() if sol.success else list(y0)

def patch_fastodeint():
    """Replace compiled fastodeint with the LSODA integrator. Idempotent."""
    import fastodeint
    if getattr(fastodeint, "_patched_lsoda", False):
        return
    fastodeint.integrate = _lsoda_integrate
    fastodeint._patched_lsoda = True
