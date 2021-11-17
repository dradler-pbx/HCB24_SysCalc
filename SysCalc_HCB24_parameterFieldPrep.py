import ComponentsLibrary as cmp
from scipy.optimize import fsolve
import numpy as np
from CoolProp.CoolProp import PropsSI as CPPSI
from tqdm import tqdm
import pandas as pd

params = dict()
params['TBox'] = 4.  # degC
params['TAi'] = 30.  # degC

params['RPM'] = 1800  # 1/min
params['Refrigerant'] = "R290"
params['SecLiquid'] = "INCOMP::MEG[0.5]"
params['kEVAP'] = 420.  # W/m2/K
params['AEVAP'] = 1.  # m2
params['dTSH'] = 4.  # K
params['kCOND'] = 45.  # W/m2/K
params['ACOND'] = 10.  # m2
params['mACOND'] = 0.56  # kg/s
params['dTSC'] = 0.1  # K
params['kHEX'] = 500  # W/m2/K
params['AHEX'] = 1.  # m2
params['mABox'] = 0.5  # kg/s
params['mSL'] = 0.25  # kg/s
params['eta_V'] = 0.9  # -
params['eta_S'] = 0.65  # -
params['stroke'] = 33e-6  # m3
params['kA_IHX'] = 2.3  # W/K
params['tolerance'] = 1e-4  # various

RPM_var = list(reversed([2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4400, 4800, 5200, 5600, 6000, 6800, 7600, 8400]))
mSL_var = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
mACOND_var = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
T_glyc_in_var = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
T_condair_in_var = [15, 20, 25, 30, 35, 40, 45, 50]

params_update_list = []
flip_flag1 = False
flip_flag2 = False
flip_flag3 = False
flip_flag4 = False

for T_glyc in T_glyc_in_var:
    mSL_var = list(reversed(mSL_var))
    for msl in mSL_var:
        mACOND_var = list(reversed(mACOND_var))
        for macond in mACOND_var:
            RPM_var = list(reversed(RPM_var))
            for rpm in RPM_var:
                T_condair_in_var = list(reversed(T_condair_in_var))
                for T_cond in T_condair_in_var:
                    if T_cond < T_glyc:
                        continue
                    params_update_list.append({'RPM': rpm, 'mSL': msl, 'mACOND': macond, 'TSL1': T_glyc + 273.15, 'TAi': T_cond + 273.15})


'''Simulation Loop'''


def simulation(params, initial_guess:None):
    TBox = params['TBox']
    TAi = params['TAi']
    RPM = params['RPM']
    Refrigerant = params['Refrigerant']
    SecLiquid = params['SecLiquid']
    kEVAP = params['kEVAP']
    AEVAP = params['AEVAP']
    dTSH = params['dTSH']
    kCOND = params['kCOND']
    ACOND = params['ACOND']
    mACOND = params['mACOND']
    dTSC = params['dTSC']
    kHEX = params['kHEX']
    AHEX = params['AHEX']
    mABox = params['mABox']
    mSL = params['mSL']
    eta_V = params['eta_V']
    eta_S = params['eta_S']
    stroke = params['stroke']
    kA_IHX = params['kA_IHX']
    tolerance = params['tolerance']

    TSL1 = params['TSL1']

    if not initial_guess:
        TBox = TBox
        TAi = TAi
        T0 = TSL1 - 5
        T7 = T0 + dTSH
        TC = TAi + 10
        (TAoDesuperheat, TAoCondenser, TAoSubcool) = ((TC + TAi) / 2, (TC + TAi) / 2, (TC + TAi) / 2)
        T1 = T7 + 1.
        T5 = TC - 1.

        p0 = CPPSI("P", "T", T0, "Q", 1, Refrigerant)
        pC = CPPSI("P", "T", TC, "Q", 0, Refrigerant)
        TSL2 = TSL1 - 3
        TSLmid = TSL2 + 2.5
        (xC1, xC2, xC3) = (0.05, 0.8, 0.15)
        (xE1, xE2) = (0.8, 0.2)
        TAHEX = TBox - 2

    else:
        T0 = initial_guess['T0']
        T7 = initial_guess['T7']
        TC = initial_guess['TC']
        TAoDesuperheat = initial_guess['TAoDesuperheat']
        TAoCondenser = initial_guess['TAoCondenser']
        TAoSubcool = initial_guess['TAoSubcool']
        T1 = initial_guess['T1']
        T5 = initial_guess['T5']
        p0 = CPPSI("P", "T", T0, "Q", 1, Refrigerant)
        pC = CPPSI("P", "T", TC, "Q", 0, Refrigerant)
        TSL2 = initial_guess['TSL2']
        TSLmid = initial_guess['TSLmid']
        (xC1, xC2, xC3) = (initial_guess['xC1'], initial_guess['xC2'], initial_guess['xC3'])
        (xE1, xE2) = (initial_guess['xE1'], initial_guess['xE1'])
    h5 = CPPSI('H', 'T', T5, 'P', pC, Refrigerant)
    h3 = CPPSI("H", "P", pC, "Q", 0, Refrigerant)
    T4 = TC

    try:
        for iteration in range(1, 1000):
            # print(iteration)
            p0prev = p0
            pCprev = pC
            TSL2prev = TSL2
            T1prev = T1
            # print(iteration)
            # (mR, Pel, h2, T2) = cmp.Poly_CPR_varRPM(p0, pC, RPM)
            (mR, Pel, h2, T2) = cmp.CPR_efficiency_model(pin=p0, Tin=T1, pC=pC, RPM=RPM, eff_S=eta_S, eff_V=eta_V,
                                                         dV=stroke)
            # print(p0, pC, RPM)
            # print(cmp.Poly_CPR_varRPM(p0, pC, RPM))
            # print([TC, TAoDesuperheat, TAoCondenser, TAoSubcool, xC1, xC2, xC3])
            # print([mR, mACOND, ACOND, T2, TAi, dTSC, kCOND])
            (TC, TAoDesuperheat, TAoCondenser, TAoSubcool, xC1, xC2, xC3) = fsolve(cmp.COND3Zone,
                                                                                   [TC, TAoDesuperheat, TAoCondenser,
                                                                                    TAoSubcool, xC1, xC2, xC3],
                                                                                   [mR, mACOND, ACOND, T2, TAi, 0, kCOND])
            pC = CPPSI("P", "T", TC, "Q", 0, Refrigerant)
            h3 = CPPSI("H", "P", pC, "Q", 0, Refrigerant)
            T4 = TC
            (T5, T1) = fsolve(cmp.IHX, [T5, T1], [kA_IHX, T7, mR, mR, Refrigerant, T4, pC, p0])
            h5 = CPPSI('H', 'T', T5, 'P', pC, Refrigerant)
            ([T0, TSL2, TSLmid, xE1, xE2], evap_infodict, evap_int, evap_str) = fsolve(cmp.EVAP, [T0, TSL2, TSLmid, xE1, xE2],
                                                  [mR, h5, mSL, TSL1, kEVAP, AEVAP, dTSH, SecLiquid, False], full_output=True)
            T7 = T0 + dTSH
            (T1, T5) = fsolve(cmp.IHX, [T1, T5], [kA_IHX, T4, mR, mR, Refrigerant, T7, p0, pC])
            p0 = CPPSI("P", "T", T0, "Q", 1, Refrigerant)


            if np.abs(p0 - p0prev) < tolerance and np.abs(pC - pCprev) < tolerance and np.abs(T1 - T1prev) < tolerance:
                break
        converged = True
    except:
        converged = False
    h7 = CPPSI('H', 'T', T7, 'P', p0, Refrigerant)
    return_dict = {'converged': converged,
                   'RPM': RPM,
                   'TAi': TAi,
                   'mACOND': mACOND,
                   'TSL1': TSL1,
                   'mSL': mSL,
                   'QEvap': mR * (h7 - h5),
                   'mR': mR,
                   'Pel': Pel,
                   'h2': h2,
                   'T2': T2,
                   'TC': TC,
                   'TAoDesuperheat': TAoDesuperheat,
                   'TAoCondenser': TAoCondenser,
                   'TAoSubcool': TAoSubcool,
                   'xC1': xC1,
                   'xC2': xC2,
                   'xC3': xC3,
                   'pC': pC,
                   'h3': h3,
                   'T4': T4,
                   'T5': T5,
                   'h5': h5,
                   'T0': T0,
                   'TSL2': TSL2,
                   'TSLmid': TSLmid,
                   'xE1': xE1,
                   'xE2': xE2,
                   'p0': p0,
                   'T7': T7,
                   'T1': T1
                   }
    return converged, return_dict


result_list = list()
i = 0
for param_update in tqdm(params_update_list):
    params.update(param_update)

    if i == 0:
        result = dict()
        result['T0'] = params['TSL1'] - 2
        result['T7'] = result['T0'] + params['dTSH']
        result['TC'] = params['TAi'] + 4
        result['TAoDesuperheat'] = (result['TC'] + params['TAi']) / 2
        result['TAoCondenser'] = (result['TC'] + params['TAi']) / 2
        result['TAoSubcool'] = (result['TC'] + params['TAi']) / 2
        result['T1'] = result['T7'] + 1
        result['T5'] = result['TC'] - 1
        result['TSL2'] = params['TSL1'] - 1
        result['TSLmid'] = params['TSL1'] - 0.5
        (result['xC1'], result['xC2'], result['xC3']) = (0.05, 0.8, 0.15)
        (result['xE1'], result['xE1']) = (0.8, 0.2)

    converged, result = simulation(params, result)
    if converged:
        result_list.append(result)
    else:
        converged, result = simulation(params, initial_guess=None)
        result_list.append(result)


    if i == 3000:
        break

    i += 1

df = pd.DataFrame(result_list)
