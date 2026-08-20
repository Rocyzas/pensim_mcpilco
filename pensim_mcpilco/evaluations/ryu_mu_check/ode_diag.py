"""Instrumented copy of indpensim_ode_py that returns the internal kinetic terms.

Mirrors PenSimPy/pensimpy/ode/indpensim_ode_py.py exactly (same constants, same
ordering) but returns a dict of the intermediate quantities instead of dy.
Only the terms needed for the Ryu / mu-threshold question are returned.
"""
import math


def indpensim_ode_diag(t, y, par):
    mu_p = par[0]
    mux_max = par[1]

    ratio_mu_e_mu_b = 0.4
    P_std_dev = 0.0015
    mean_P = 0.002
    mu_v = 1.71e-4
    mu_a = 3.5e-3
    mu_diff = 5.36e-3
    beta_1 = 0.006
    K_b = 0.05
    K_diff = 0.75
    K_diff_L = 0.09
    K_e = 0.009
    K_v = 0.05
    delta_r = 0.75e-4
    k_v = 3.22e-5
    rho_a0 = 0.35
    rho_d = 0.18
    r_0 = 1.5e-4

    Y_sX = 1.85
    Y_sP = 0.9
    m_s = 0.029
    c_oil = 1000
    c_s = 600

    a = 0.38
    Henrys_c = 0.0251
    r = 2.1
    epsilon = 0.1
    R = 8.314
    X_crit_DO2 = 0.1
    P_crit_DO2 = 0.3
    A_inhib = 1
    Eg = 14880
    Ed = 173250
    k_g = 450
    k_d = 2.5e+29
    abc = 0.033
    K1 = 1e-5
    K2 = 2.5e-8
    N_conc_paa = par[3]
    X_crit_N = 150
    PAA_c = par[4]
    X_crit_PAA = 2400
    P_crit_PAA = 200
    B_1 = -64.29
    B_2 = -1.825
    B_3 = 0.3649
    B_4 = 0.1280
    B_5 = -4.9496e-04
    X_crit_CO2 = 7570
    O_2_in = 0.21

    Fs = par[6]
    Fw = par[14]
    Fw = 0 if Fw < 0 else Fw
    pressure = par[15]
    viscosity = y[9] if par[30] == 0 else par[16]
    F_discharge = par[17]
    Fpaa = par[18]
    Foil = par[19]
    Fb = par[11]
    Fa = par[12]
    dist_flag = par[21]
    distMuP = par[22]
    distMuX = par[23]
    distsc = par[24]
    distcoil = par[25]
    distPAA = par[27]
    distO_2in = par[29]
    alpha_evp = 5.2400e-4
    Tv = 373
    T0 = 273

    pho_b = (1100 + y[3] + y[11] + y[12] + y[13] + y[14])

    if dist_flag == 1:
        mu_p += distMuP
        mux_max += distMuX
        c_s = c_s + distsc
        c_oil += distcoil
        PAA_c += distPAA
        O_2_in += distO_2in

    A_t1 = (y[10]) / (y[11] + y[12] + y[13] + y[14])

    s = y[0]
    a_1 = y[12]
    a_0 = y[11]
    a_3 = y[13]
    total_X = y[11] + y[12] + y[13] + y[14]

    h_b = (y[4] / 1000) / (math.pi * r ** 2)
    h_b = h_b * (1 - epsilon)
    pressure_bottom = 1 + pressure + pho_b * h_b * 9.81e-5
    pressure_top = 1 + pressure
    total_pressure = (pressure_bottom - pressure_top) / (math.log(pressure_bottom / pressure_top))

    viscosity = 1 if viscosity < 4 else viscosity
    DOstar_tp = total_pressure * O_2_in / Henrys_c

    # the unconditional block (lines 216-225 of the original) -- inhib_flag branches above it
    # are dead code, these always overwrite.
    pH_inhib = 1 / (1 + (y[6] / K1) + (K2 / y[6]))
    NH3_inhib = 0.5 * (1 - math.tanh(A_inhib * (X_crit_N - y[30])))
    T_inhib = k_g * math.exp(-(Eg / (R * y[7]))) - k_d * math.exp(-(Ed / (R * y[7])))
    CO2_inhib = 0.5 * (1 + math.tanh(A_inhib * (X_crit_CO2 - y[28] * 1000)))
    DO_2_inhib_X = 0.5 * (1 - math.tanh(A_inhib * (X_crit_DO2 * DOstar_tp - y[1])))
    DO_2_inhib_P = 0.5 * (1 - math.tanh(A_inhib * (P_crit_DO2 * DOstar_tp - y[1])))
    PAA_inhib_X = 0.5 * (1 + (math.tanh((X_crit_PAA - y[29]))))
    PAA_inhib_P = 0.5 * (1 + (math.tanh((-P_crit_PAA + y[29]))))
    pH = -math.log10(y[6])
    mu_h = math.exp((B_1 + B_2 * pH + B_3 * y[7] + B_4 * (pH ** 2)) + B_5 * (y[7] ** 2))

    P_inhib = 2.5 * P_std_dev * ((P_std_dev * 2.5066282746310002) ** -1
                                 * math.exp(-0.5 * ((s - mean_P) / P_std_dev) ** 2))

    mu_a0 = ratio_mu_e_mu_b * mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X
    mu_e = mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X

    K_diff = 0.75 - (A_t1 * beta_1)
    if K_diff < K_diff_L:
        K_diff = K_diff_L

    r_b0 = mu_a0 * a_1 * s / (K_b + s)
    r_e1 = (mu_e * a_0 * s) / (K_e + s)
    r_d1 = mu_diff * a_0 / (K_diff + s)
    r_m0 = m_s * a_0 / (K_diff + s)

    n = 16
    phi = [0] * 10
    phi[0] = y[26]
    for k in range(2, 11):
        phi[k - 1] = 4.1887902047863905 * (1.5e-4 + (k - 2) * delta_r) ** 3 * y[n] * delta_r
        n += 1
    v_2 = sum(phi)
    rho_a1 = (a_1 / ((a_1 / rho_a0) + v_2))
    v_a1 = a_1 / (2 * rho_a1) - v_2

    r_p_gross = mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P
    r_p_degrad = mu_h * y[3]
    r_p = r_p_gross - r_p_degrad

    r_m1 = (m_s * rho_a0 * v_a1 * s) / (K_v + s)
    r_d4 = mu_a * a_3

    F_evp = y[4] * alpha_evp * (math.exp(2.5 * (y[7] - T0) / (Tv - T0)) - 1)
    dilution = Fs + Fb + Fa + Fw - F_evp + Fpaa

    r_k = r_0 + 8 * delta_r
    r_m = (r_0 + 10 * delta_r)
    n_k = y[24]
    vac_flux = (math.pi * ((r_k + r_m) ** 3) / 6) * rho_d * k_v * n_k

    da_0_dt = r_b0 - r_d1 - y[11] * dilution / y[4]
    da_1_dt = r_e1 - r_b0 + r_d1 - vac_flux - y[12] * dilution / y[4]
    da_3_dt = vac_flux - r_d4 - y[13] * dilution / y[4]
    da_4_dt = r_d4 - y[14] * dilution / y[4]
    X_1 = da_0_dt + da_1_dt + da_3_dt + da_4_dt
    dP_dt = r_p - y[3] * dilution / y[4]

    ds_dt = (-r_e1 * Y_sX - r_b0 * Y_sX - r_m0 - r_m1
             - (Y_sP * mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P)
             + Fs * c_s / y[4] + Foil * c_oil / y[4] - y[0] * dilution / y[4])

    return dict(
        t=t,
        s=s,
        a0=a_0, a1=a_1, a3=a_3, a4=y[14], X=total_X, P=y[3], V=y[4],
        DO2=y[1], DOstar=DOstar_tp, DO2_pct_sat=100.0 * y[1] / DOstar_tp,
        NH3=y[30], PAA=y[29], CO2_d_mgL=y[28] * 1000.0, pH=pH, T=y[7],
        viscosity=viscosity,
        mux_max=mux_max, mu_p_par=mu_p,
        # inhibition switches
        pH_inhib=pH_inhib, NH3_inhib=NH3_inhib, T_inhib=T_inhib, CO2_inhib=CO2_inhib,
        DO_2_inhib_X=DO_2_inhib_X, DO_2_inhib_P=DO_2_inhib_P,
        PAA_inhib_X=PAA_inhib_X, PAA_inhib_P=PAA_inhib_P,
        P_inhib=P_inhib, mu_h=mu_h,
        # exposed "mu" channels
        mu_e=mu_e, mu_a0=mu_a0,
        # true realised specific rates
        r_b0=r_b0, r_e1=r_e1, r_d1=r_d1, r_d4=r_d4, vac_flux=vac_flux,
        mu_X_true=r_e1 / total_X,                     # gross specific growth rate, 1/h
        mu_X_net=X_1 / total_X,                       # incl. dilution, 1/h
        monod_e=s / (K_e + s), monod_b=s / (K_b + s),
        a0_frac=a_0 / total_X,
        v_a1=v_a1,
        # production
        r_p=r_p, r_p_gross=r_p_gross, r_p_degrad=r_p_degrad, dP_dt=dP_dt,
        qP=r_p_gross / total_X,                       # specific production rate, g/g/h
        dX_dt=X_1, ds_dt=ds_dt, dilution=dilution,
    )
