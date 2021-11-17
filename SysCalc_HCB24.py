import ComponentsLibrary as cmp
from scipy.optimize import fsolve
import numpy as np
from CoolProp.CoolProp import PropsSI as CPPSI
from openpyxl import workbook, load_workbook
from tqdm import tqdm
TBox = 4.  # degC
TAi = 30.  # degC

RPM = 1800  # 1/min
Refrigerant = "R290"
SecLiquid = "INCOMP::MEG[0.5]"
kEVAP = 420.  # W/m2/K
AEVAP = 1.  # m2
dTSH = 4.  # K
kCOND = 45.  # W/m2/K
ACOND = 10.  # m2
mACOND = 0.56  # kg/s
dTSC = 0.1  # K
kHEX = 500  # W/m2/K
AHEX = 1.  # m2
mABox = 0.5  # kg/s

mSL = 0.25  # kg/s
eta_V = 0.9  # -
eta_S = 0.65  # -
stroke = 33e-6  # m3
kA_IHX = 2.3  # W/K
tolerance = 1e-4  # various


TBox = TBox + 273.15
TAi = TAi + 273.15
T0 = TBox - 10
T7 = T0 + dTSH
TC = TAi +10
(TAoDesuperheat, TAoCondenser, TAoSubcool) = ((TC+TAi)/2, (TC+TAi)/2, (TC+TAi)/2)
T1 = T7 + 1.
T5 = TC - 1.

p0 = CPPSI("P","T",T0,"Q",1,Refrigerant)
pC = CPPSI("P","T",TC,"Q",0,Refrigerant)
TSL1 = TBox -2
TSL2 = TSL1 - 3
TSLmid = TSL2 + 2.5
(xC1,xC2,xC3) = (0.05,0.8,0.15)
(xE1,xE2) = (0.8,0.2)
TAHEX = TBox - 2

'''Simulation Loop'''
for iteration in range(1,1000):
    # print(iteration)
    p0prev = p0
    pCprev = pC
    TSL1prev = TSL1
    T1prev = T1
    # print(iteration)
    # (mR, Pel, h2, T2) = cmp.Poly_CPR_varRPM(p0, pC, RPM)
    (mR, Pel, h2, T2) = cmp.CPR_efficiency_model(pin=p0, Tin=T1, pC=pC, RPM=RPM, eff_S=eta_S, eff_V=eta_V, dV=stroke)
    # print(p0, pC, RPM)
    # print(cmp.Poly_CPR_varRPM(p0, pC, RPM))
    # print([TC, TAoDesuperheat, TAoCondenser, TAoSubcool, xC1, xC2, xC3])
    # print([mR, mACOND, ACOND, T2, TAi, dTSC, kCOND])
    (TC, TAoDesuperheat, TAoCondenser, TAoSubcool, xC1, xC2, xC3) = fsolve(cmp.COND3Zone,[TC, TAoDesuperheat, TAoCondenser, TAoSubcool, xC1, xC2, xC3], [mR, mACOND, ACOND, T2, TAi, 0, kCOND])
    pC = CPPSI("P","T",TC,"Q",0,Refrigerant)
    h3 = CPPSI("H","P",pC,"Q",0,Refrigerant)
    T4 = TC
    (T5, T1) = fsolve(cmp.IHX, [T5, T1], [kA_IHX, T7, mR, mR, Refrigerant, T4, pC, p0])
    h5 = CPPSI('H', 'T', T5, 'P', pC, Refrigerant)
    (T0, TSL2, TSLmid, xE1, xE2) = fsolve(cmp.EVAP,[T0, TSL2, TSLmid, xE1, xE2], [mR, h5, mSL, TSL1, kEVAP, AEVAP, dTSH, SecLiquid, False])
    T7 = T0 + dTSH
    (T1, T5) = fsolve(cmp.IHX, [T1, T5], [kA_IHX, T4, mR, mR, Refrigerant, T7, p0, pC])
    p0 = CPPSI("P","T", T0, "Q", 1, Refrigerant)
    (TSL1, TAHEX)=fsolve(cmp.HeatEx, [TSL1, TAHEX], [mSL, TSL2, mABox, TBox, kHEX, AHEX, SecLiquid])

    if np.abs(p0-p0prev) < tolerance and np.abs(pC-pCprev) < tolerance and np.abs(TSL1-TSL1prev) < tolerance and np.abs(T1-T1prev) < tolerance:
        break

# calculate all necessary thermodynamic data for visualization
T3 = TC
h1 = CPPSI("H", "T", T1, "P", p0, Refrigerant)
s1 = CPPSI("S", "T", T1, "P", p0, Refrigerant)
s2 = CPPSI("S", "T", T2, "P", pC, Refrigerant)
s3 = CPPSI("S", "T", T3, "Q", 0, Refrigerant)
s4 = CPPSI("S", "T", T4, "Q", 0, Refrigerant)
pSL = 1e5
hSL1 = CPPSI("H", "T", TSL1, "P", pSL, SecLiquid)
sSL1 = CPPSI("S", "T", TSL1, "P", pSL, SecLiquid)
hSL2 = CPPSI("H", "T", TSL2, "P", pSL, SecLiquid)
sSL2 = CPPSI("S", "T", TSL2, "P", pSL, SecLiquid)
(p5, p6, p7) = (p0, p0, p0)
T6 = T0
s5 = CPPSI("S", "P", p5, "H", h5, Refrigerant)
h6 = h5
s6 = CPPSI("S", "P", p6, "H", h6, Refrigerant)
h7 = CPPSI("S", "T", T7, "P", p0, Refrigerant)
s7 = CPPSI("S", "P", p7, "H", h7, Refrigerant)

# generate the data container
chariot = np.zeros([12,4])
chariot[0] = [Pel, mR, mABox, mACOND]
chariot[1] = [T1, p0, h1, s1]
chariot[2] = [T2, pC, h2, s2]
chariot[3] = [TC - dTSC, pC, h3, s3]
chariot[4] = [T4, p0, h3, s4]
chariot[5] = [TSL1, pSL, hSL1, sSL1]
chariot[6] = [TSL2, pSL, hSL2, sSL2]
chariot[7] = [RPM, mSL, TC, TBox]
chariot[8] = [TAHEX, TAi, TAoCondenser, 0]
chariot[9] = [T5, p5, h5, s5]
chariot[10] = [T6, p6, h6, s6]
chariot[11] = [T7, p7, h7, s7]

cmp.RefCyclePlot(chariot)



