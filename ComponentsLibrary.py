# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 13:13:55 2018

@author: dominik
"""

from CoolProp.CoolProp import PropsSI as CPPSI
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread



def dhCOND(x):
    #Enthalpy of condensation as a function of condensing temperature for R290
    y = -10.185*x**2 + 4282.3*x - 35495
    return y

def H2EVAP(x):
    #Enthalpy of saturated R290 gas at a given temperature (x)
    if x<200 or x>320:
        return "ERROR in H2EVAP. Temperature exceeded limits!"
    else:
        return -1.7605*x**2 + 2045.4*x + 147383

def LMTD_calc(Thi,Tho,Tci,Tco):
    #calculating the logaritmic mean temperature of two fluids with defined "hot" and "cold" fluid
    dT1 = Thi-Tco
    dT2 = Tho-Tci
    
    if dT1 == dT2:
        LMTD = dT1
    else:
        LMTD = (dT1-dT2)/np.log(dT1/dT2)
    if dT1 < 0:
        return 0.0
    # prevent NaN values:
    if np.isnan(LMTD):
        LMTD = 1e-6
    return LMTD

def COND3Zone(x, args):
    # Calculating a three-zone condenser model based on LMTD model
    # Further infos in documentation
    # (TC, TAoDesuperheat, TAoCondenser, TAoSubcool, xC1, xC2, xC3)

    # Necessary model parameters
    mR = args[0]
    mA = args[1]
    Atot = args[2]
    TRin = args[3]
    TAi = args[4]
    dTSC = args[5]
    k = args[6] * np.ones(3)

    # Boundary for fsolve calculation to not cause the logaritmic mean temperature to generate NaN values (neg. logarithm):
    # The outlet air temperature of the superheat section must not exceed the refrigerant inlet temperature.
    if x[1] > TRin:
        x[1] = TRin - 1e-4
    if x[0]-dTSC < TAi:
        x[0] = TAi + dTSC + 1e-4

    #calculate material parameters
    cpR = np.zeros(2)
    cpR[0] = CPPSI("C", "T", x[0], "Q", 1, "R290")
    cpR[1] = CPPSI("C", "T", x[0], "Q", 0, "R290")
    cpA = CPPSI("C", "T", TAi, "P", 1.0e5, "AIR")

    #Calculate the mean logaritmic temperature value for all three sections of the condenser
    LMTD = np.zeros(3)
    LMTD[0] = LMTD_calc(TRin, x[0], TAi, x[1])
    LMTD[1] = LMTD_calc(x[0], x[0], TAi, x[2])
    LMTD[2] = LMTD_calc(x[0], x[0] - dTSC, TAi, x[3])

    # Formulation of the equation system as according to fsolve documentation ( 0 = ... ).
    # The equation set  and model definition is documented in the model description.
    f = np.zeros(7)
    f[0] = mR * cpR[0] * (TRin - x[0]) - mA * cpA * x[4] * (x[1] - TAi)
    f[1] = mR * cpR[0] * (TRin - x[0]) - k[0] * x[4] * Atot * LMTD[0]
    f[2] = mR * dhCOND(x[0]) - k[1] * x[5] * Atot * LMTD[1]
    f[3] = mR * dhCOND(x[0]) - mA * cpA * x[5] * (x[2] - TAi)
    f[4] = mR * cpR[1] * dTSC - mA * cpA * x[6] * (x[3] - TAi)
    f[5] = mR * cpR[1] * dTSC - k[2] * x[6] * LMTD[2]
    f[6] = 1 - x[4] - x[5] - x[6]

    return f

def Poly_CPR(p0,pC,RPM):
    # Calculating the mass flow, electrical power consumption and refrigerant outlet conditions
    # of an Embraco VNEU217 compressor with pressure and RPM input parameters

    # Calculate the referencing evaporation/condensing temperatures to the given pressure,
    # because the polynomial data is given depending on temperatures
    T0 = CPPSI("T","P",p0,"Q",1,"R290")-273.15
    TC = CPPSI("T","P",pC,"Q",0,"R290")-273.15

    # three different rpms are possible (2000,3000,4500)
    if RPM==4500:
        aQ=[0.017150000000,0.000317300000,-0.000457700000,0.000003729000,0.000000754800,0.000010360000,-0.000000004374,-0.000000044700,-0.000000040250,-0.000000085950]
        aP=[4.66664E+02,1.26299E+01,3.28778E+01,-5.87948E-02,2.09458E-02,-5.41316E-01,-1.85488E-03,-9.23727E-04,1.65486E-03,4.09168E-03]
    elif RPM==3000:
        aQ=[0.01037000000,0.00019940000,-0.00020150000,0.00000278300,0.00000268300,0.00000518900,0.00000001782,0.00000001620,-0.00000002457,-0.00000004455]
        aP=[-365.423,-5.95874,56.7609,-0.23166,0.367893,-1.01831,-0.0019421,0.00236174,-0.000607773,0.007214]
    elif RPM==2000:
        aQ=[0.006223000000,0.000243300000,0.000011780000,0.000001910000,-0.000003135000,-0.000001614000,-0.000000008229,-0.000000034240,0.000000010630,0.000000014810]
        aP=[-1074.780,-37.994,73.1891,-1.34793,0.0706826,-1.56424,-0.0137568,0.00129106,0.0014607,0.0120691]
    else:
        print("Wrong RPM!")
        return
    #calculate mass flow
    mdot=aQ[0]+aQ[1]*T0+aQ[2]*TC+aQ[3]*T0**2+aQ[4]*T0*TC+aQ[5]*TC**2+aQ[6]*T0**3+aQ[7]*T0**2*TC+aQ[8]*T0*TC*TC+aQ[9]*TC**3
    #calculate the electr. power consumption
    P=aP[0]+aP[1]*T0+aP[2]*TC+aP[3]*T0**2+aP[4]*T0*TC+aP[5]*TC**2+aP[6]*T0**3+aP[7]*T0**2*TC+aP[8]*T0*TC*TC+aP[9]*TC**3
    #at this point all relevant data is obtained, the rest of the parameters are just derived from them:

    #calculating the outlet enthalpy
    hin = CPPSI("H","T",T0+273.15+7,"P",p0,"R290")
    hout = P/mdot + hin

    #calculating the outlet temperature
    Tout = CPPSI("T","H",hout,"P",pC,"R290")
    
    return (mdot,P,hout,Tout)

def Poly_CPR_varRPM(p0, pC, RPM):
    # Calculating the mass flow, electrical power consumption and refrigerant outlet conditions
    # of an Embraco VNEU217 compressor with pressure and RPM input parameters
    # with linear interpolation between 2000 and 4500RPM

    # Calculate the referencing evaporation/condensing temperatures to the given pressure,
    # because the polynomial data is given depending on temperatures
    A = np.zeros(4)
    B = np.zeros(4)
    sol = np.zeros(4)
    T0 = CPPSI("T","P",p0,"Q",1,"R290")-273.15
    hin = CPPSI("H","T",T0+273.15+7,"P",p0,"R290")

    if RPM <= 3000:
        B = np.asarray(Poly_CPR(p0,pC,3000))
        sol = A + RPM * (B-A)/3000
        sol[2] = sol[1]/sol[0] + hin
        sol[3] = CPPSI("T","H",sol[2],"P",pC,"R290")
    if RPM > 3000 and RPM <= 4500:
        A = np.asarray(Poly_CPR(p0,pC,3000))
        B = np.asarray(Poly_CPR(p0,pC,4500))
        sol = A + (RPM - 3000) * (B - A) / (4500-3000)
        sol[2] = sol[1]/sol[0] + hin
        sol[3] = CPPSI("T","H",sol[2],"P",pC,"R290")

    return sol

def CPR_efficiency_model(pin, Tin, pC, RPM, eff_S, eff_V, dV):
    #compressor model based on efficiency parameters: eps_S...isentropic / eps_V...volumetric
    rho = CPPSI("D", "P", pin, "T", Tin, "R290")
    mdot = RPM/60 * dV * eff_V * rho  # mass flow
    hin = CPPSI("H","T",Tin,"P",pin,"R290")  # inlet enthalpy
    sin = CPPSI("S","T",Tin,"P",pin,"R290")  # inlet entropy
    houtS = CPPSI("H", "S", sin, "P", pC, "R290")  # enthalpy at outlet under isentropic conditions
    P = mdot * (houtS-hin)/eff_S  # power input
    hout = P/mdot + hin  # real outlet enthalpy
    Tout = CPPSI("T","P",pC,"H",hout,"R290")  # outlet temperature


    return (mdot,P,hout,Tout)

def EVAP(x,args):
    # Calculating a two-zone evaporator model based on LMTD model
    # Further infos in documentation
    # x = [T0,TSLout,TSLmid,x_evaporation,x_superheat]

    # Necessary model parameters
    mR = args[0]
    hRi  = args[1]
    mSL   = args[2]
    TSLi  = args[3]
    kGiven = args[4]
    Atot = args[5]
    dTSH = args[6]
    SecLiquid = args[7]
    BoundarySwitch = args[8]
    SecLiquid = 'INCOMP::MEG[0.5]'
    f = np.zeros(5)
    # print(BoundarySwitch)

    # Boundaries for fsolve calculation to not cause the logaritmic mean temperature to generate NaN values (neg. logarithm)
    # The refrigerants oulet temperature must not be higher than the coolants inlet temperature:
    if BoundarySwitch:
        # print('Boundary on')
        if x[0] + dTSH > TSLi:
            x[0] = TSLi - dTSH
            # print("BING1")        #Visualize, if the code runs in this boundary
        # # The evaporation temperature must not be higher than the coolants outlet temperature
        if x[0] > x[1]:
            x[0] = x[1]-1e-6
        #     print("BING2")        #Visualize, if the code runs in this boundary
        if x[1] > x[2]:
            x[1] = x[2] + 1e-3
        #     print("BING3")        #Visualize, if the code runs in this boundary

    # For the two different zones of heat transfer, different heat transfer coefficients can be defined.
    # If only one is given, it shall be used for all zones.
    if isinstance(kGiven,float) or isinstance(kGiven,int):
        k = np.ones(2)
        k = k * kGiven
    elif len(kGiven) == 2:
        k = kGiven
    else:
        raise ValueError('wrong dimension of k-value')
        return


    # calculate material parameters
    cpR = CPPSI('C','T',x[0],'Q',1,'R290')  # heat capacity of fully evaporated refrigerant
    cpSL = CPPSI('C','T',(TSLi+x[1])/2,'P',1e5,SecLiquid)  # heat capacity of secondary liquid
    hRGas = CPPSI("H","T",x[0],"Q",1,"R290")  # enthalpy of fully evaporated refrigerant
    # p0 = CPPSI("P","T",x[0],"Q",1,"R290")
    # hRSuperheated = CPPSI("H","T",x[0]+dTSH,"P",p0,"R290")


    # Calculate the mean logarithmic temperature value for all two sections of the condenser
    LMTD = np.zeros(2)
    LMTD[0] = LMTD_calc(x[2], x[1], x[0], x[0])
    LMTD[1] = LMTD_calc(TSLi, x[2], x[0], dTSH+x[0])
    # Formulation of the equation system as according to fsolve documentation ( 0 = ... ).
    # The equation set  and model definition is documented in the model description.
    # x = [T0,TSLout,TSLmid,x_evaporation,x_superheat]

    # energy balance evaporating zone between refrigerant and sec. liquid
    f[0] = mR * (hRGas - hRi) - mSL * cpSL * (x[2] - x[1])

    # energy balance evaporating zone between refrigerant and LMTD model
    f[1] = mR * (hRGas - hRi) - k[0] * x[3] * Atot * LMTD[0]

    # energy balance superheating zone between refrigerant and sec. liquid
    # f[ 2 ] = mR * (hRSuperheated - hRGas) - mSL * cpSL * (TSLi - x[ 2 ])
    f[2] = mR * cpR * dTSH - mSL * cpSL * (TSLi - x[ 2 ])

    # energy balance superheating zone between refrigerant and LMTD model
    # f[3] = mR * (hRSuperheated-hRGas) - k[1] * x[4]/100 * Atot * LMTD[1]
    f[3] = mR * cpR * dTSH - k[ 1 ] * x[ 4 ] * Atot * LMTD[ 1 ]

    # area fraction balance (1 = x_evaporating + x_superheating)
    f[4] = 1 - x[3] - x[4]

    return f

def EVAPspecialSH(x,args):
    # Calculating a two-zone evaporator model based on LMTD model
    # Further infos in documentation
    # x = [T0,TSLout,x_evaporation,x_superheat]
    # this model does not consider any glycol temperature raise due to superheat

    # Necessary model parameters
    mR = args[0]
    hRi  = args[1]
    mSL   = args[2]
    TSLi  = args[3]
    kGiven = args[4]
    Atot = args[5]
    dTSH = args[6]
    SecLiquid = args[7]
    SecLiquid = 'INCOMP::MEG[0.5]'
    f = np.zeros(4)

    # Boundaries for fsolve calculation to not cause the logaritmic mean temperature to generate NaN values (neg. logarithm)
    # The refrigerants oulet temperature must not be higher than the coolants inlet temperature:
    if x[0] + dTSH > TSLi:
        x[0] = TSLi - dTSH
    #     print("BING1")        #Visualize, if the code runs in this boundary
    # # The evaporation temperature must not be higher than the coolants outlet temperature
    if x[0] > x[1]:
        x[0] = x[1]-1.
        # print("BING2")        #Visualize, if the code runs in this boundary
    # if x[1] > x[2]:
    #     x[1] = x[2] + 1e-3
    # #     print("BING3")        #Visualize, if the code runs in this boundary

    # For the two different zones of heat transfer, different heat transfer coefficients can be defined.
    # If only one is given, it shall be used for all zones.
    if isinstance(kGiven,float) or isinstance(kGiven,int):
        k = np.ones(2)
        k = k * kGiven
    elif len(kGiven) == 2:
        k = kGiven
    else:
        raise ValueError('wrong dimension of k-value')
        return

    # calculate material parameters
    cpR = CPPSI('C','T',x[0],'Q',1,'R290')  # heat capacity of fully evaporated refrigerant
    cpSL = CPPSI('C','T',(TSLi+x[1])/2,'P',1e5,SecLiquid)  # heat capacity of secondary liquid
    hRGas = CPPSI("H","T",x[0],"Q",1,"R290")  # enthalpy of fully evaporated refrigerant
    # p0 = CPPSI("P","T",x[0],"Q",1,"R290")
    # hRSuperheated = CPPSI("H","T",x[0]+dTSH,"P",p0,"R290")

    # Calculate the mean logarithmic temperature value for all two sections of the condenser
    LMTD = np.zeros(2)
    LMTD[0] = LMTD_calc(TSLi, x[1], x[0], x[0])
    LMTD[1] = LMTD_calc(TSLi, TSLi, x[0], dTSH+x[0])

    # Formulation of the equation system as according to fsolve documentation ( 0 = ... ).
    # The equation set  and model definition is documented in the model description.
    # x = [T0,TSLout,x_evaporation,x_superheat]
    # energy balance evaporating zone between refrigerant and sec. liquid
    f[0] = mR * (hRGas - hRi) - mSL * cpSL * (TSLi - x[1])

    # energy balance evaporating zone between refrigerant and LMTD model
    f[1] = mR * (hRGas - hRi) - k[0] * x[2] * Atot * LMTD[0]

    return f

def EVAPsinglezone(x,args):
    # Calculating a two-zone evaporator model based on LMTD model
    # Further infos in documentation
    # x = [T0,TSLout]

    # Necessary model parameters
    mR = args[0]
    hRi  = args[1]
    mSL   = args[2]
    TSLi  = args[3]
    kGiven = args[4]
    Atot = args[5]
    dTSH = args[6]
    SecLiquid = args[7]
    # SecLiquid = 'INCOMP::MEG[0.5]'

    # Boundaries for fsolve calculation to not cause the logaritmic mean temperature to generate NaN values (neg. logarithm)
    # The refrigerants oulet temperature must not be higher than the coolants inlet temperature:
    # if x[0] + dTSH > TSLi:
    #     x[0] = TSLi - dTSH
    #     print("BING1")        #Visualize, if the code runs in this boundary
    # # The evaporation temperature must not be higher than the coolants outlet temperature
    # if x[0] > x[1]:
    #     x[0] = x[1]-1e-6
    #     print("BING2")        #Visualize, if the code runs in this boundary
    # if x[1] > x[2]:
    #     x[1] = x[2] + 1e-3
    #     print("BING3")        #Visualize, if the code runs in this boundary

    # For the two different zones of heat transfer, different heat transfer coefficients can be defined.
    # If only one is given, it shall be used for all zones.
    # if isinstance(kGiven,float) or isinstance(kGiven,int):
    #     k = np.ones(2)
    #     k = k * kGiven
    # elif len(kGiven) == 2:
    #     k = kGiven
    # else:
    #     raise ValueError('wrong dimension of k-value')
    #     return


    # calculate material parameters
    # cpR = CPPSI('C','T',x[0],'Q',1,'R290')  # heat capacity of fully evaporated refrigerant
    cpSL = CPPSI('C','T',(TSLi+x[1])/2,'P',1e5,SecLiquid)  # heat capacity of secondary liquid
    hRGas = CPPSI("H","T",x[0],"Q",1,"R290")  # enthalpy of fully evaporated refrigerant
    # p0 = CPPSI("P","T",x[0],"Q",1,"R290")
    # hRSuperheated = CPPSI("H","T",x[0]+dTSH,"P",p0,"R290")

    # Calculate the mean logarithmic temperature value for all two sections of the condenser
    LMTD = LMTD_calc(TSLi, x[1], x[0], x[0])

    # Formulation of the equation system as according to fsolve documentation ( 0 = ... ).
    # The equation set  and model definition is documented in the model description.
    # x = [T0,TSLout,TSLmid,x_evaporation,x_superheat]
    f = np.zeros(2)

    # energy balance evaporating zone between refrigerant and sec. liquid
    f[0] = mR * (hRGas - hRi) - mSL * cpSL * (TSLi - x[1])

    # energy balance evaporating zone between refrigerant and LMTD model
    f[1] = mR * (hRGas - hRi) - kGiven * Atot * LMTD

    # energy balance superheating zone between refrigerant and sec. liquid
    # f[2] = mR * (hRSuperheated-hRGas) - mSL * cpSL * (TSLi - x[2])

    # energy balance superheating zone between refrigerant and LMTD model
    # f[3] = mR * (hRSuperheated-hRGas) - k[1] * x[4]/100 * Atot * LMTD[1]

    # area fraction balance (1 = x_evaporating + x_superheating)
    # f[4] = 100 - x[3] - x[4]

    return f

def HeatEx(x,args):
    #x = [TSLo,TAo]
    [mSL, TSLi, mA, TAi, kHEX, AHEX, SecLiquid] = args
    cpSL = CPPSI("C","T",(TSLi+x[0])/2,"P",1e5,SecLiquid)
    cpA = CPPSI("C","T",(TAi+x[1])/2,"P",1e5,"Air")
    LMTD = LMTD_calc(TAi,x[1],TSLi,x[0])

    f = np.zeros(2)
    f[0] = mSL * cpSL * (x[0]-TSLi) - mA * cpA * (TAi - x[1])
    f[1] = mSL * cpSL * (x[0]-TSLi) - kHEX * AHEX * LMTD
    return f

def RefCyclePlot(data):
    #Plot function, that works with a data set of style:
    # chariot[0] = [Pel, mR, mABox, mACOND]
    # chariot[1] = [T1, p0, h1, s1]
    # chariot[2] = [T2, pC, h2, s2]
    # chariot[3] = [T3, pC, h3, s3]
    # chariot[4] = [T4, p0, h3, s4]
    # chariot[5] = [TSL1, pSL, hSL1, sSL1]
    # chariot[6] = [TSL2, pSL, hSL2, sSL2]
    # chariot[7] = [RPM, mSL, TC, TBox]
    # chariot[8] = [TAHEX, TAi, TACo, 0]
    # chariot[9] = [T5, p5, h5, s5]
    # chariot[10] = [T6, p6, h6, s6]
    # chariot[11] = [T7, p7, h7, s7]
    #----------------
    #A plot will be generated that shows the system flow diagram on the left side with the state parameters
    # and a log(p),h - diagram and a T,s - diagram on the right side.

    TGlycolIn = data[5][0]
    TGlycolOut = data[6][0]
    x_evap_in = CPPSI('Q', 'P', data[1,1], 'H', data[10,2], 'R290')


    '''Data generation'''
    QEVAP = data[0,1]*(data[4,2]-data[1,2])
    ref = "R290"

    newPlot = plt.figure(figsize=(16,9), dpi=80)

    plt.subplot(121)

    cycleimage = imread('media/200428_SystemDrawing_v10.jpg')
    plt.imshow(cycleimage, interpolation='bilinear')
    plt.axis('off')

    # Ref general text
    plt.text(440, 250, 'REF-GENERAL:\nPel = ' + str(round(data[0][0], 2)) + 'W\nCOP = '+ str(round(-QEVAP/data[0][0],2)) + '\n n = ' + str(data[7,0]) + 'RPM\nmdot_ref = ' + str(
        round(data[0,1] * 1000, 2)) + 'g/s\nmdot_Air,cond = ' + str(round(data[0][3], 2)) + 'kg/s', fontsize=8, ha='center', va='top')
    # T0, TC
    plt.text(440, 450,'TC = ' + str(round(data[7][2] - 273.15, 2)) + '°C\nT0 = ' + str(round(data[10][0] - 273.15, 2)) + '°C', fontsize=8, ha='center', va='top')

    # SL general text
    plt.text(440, 790, 'SL-GENERAL:\nQ = ' + str(round(QEVAP, 2)) + 'W\nmdot_SL = ' + str(
        round(data[7,1], 2)) + 'kg/s\nmdot_Air,cargo = ' + str(round(data[0,2], 2)) + ' kg/s', fontsize=8, ha='center', va='top')

    # Compressor outlet
    plt.text(730, 150, 'T = ' + str(round(data[2,0] - 273.15, 2)) + '°C\np = ' + str(round(data[2,1] * 1e-5, 2)) + 'bar', fontsize=8, ha='left', va='top')

    # Condenser outlet
    plt.text(160, 120, 'T = ' + str(round(data[3,0] - 273.15, 2)) + '°C\np = ' + str(round(data[3,1] * 1e-5, 2)) + 'bar', fontsize=8, ha='left', va='bottom')

    # Receiver outlet
    plt.text(160, 355, 'T = ' + str(round(data[4,0] - 273.15, 2)) + '°C\np = ' + str(round(data[4,1] * 1e-5, 2)) + 'bar', fontsize=8, ha='left', va='bottom')

    # TXV inlet
    plt.text(110, 420, 'T = ' + str(round(data[9,0] - 273.15, 2)) + '°C\np = ' + str(round(data[9,1] * 1e-5, 2)) + 'bar', fontsize=8, ha='right', va='top')

    # TXV outlet
    plt.text(110, 570, 'T = ' + str(round(data[10,0] - 273.15, 2)) + '°C\np = ' + str(round(data[10,1] * 1e-5, 2)) + 'bar\nx ='+str(round(x_evap_in, 3)), fontsize=8, ha='right', va='top')

    # Evaporator outlet
    plt.text(730, 500, 'T = ' + str(round(data[11,0] - 273.15, 2)) + '°C\np = ' + str(round(data[11,1] * 1e-5, 2)) + 'bar', fontsize=8, ha='left', va='top')

    # Compressor inlet
    plt.text(730, 310, 'T = ' + str(round(data[1,0] - 273.15, 2)) + '°C\np = ' + str(round(data[1,1] * 1e-5, 2)) + 'bar', fontsize=8, ha='left', va='top')

    # glycol temperatures
    plt.text(160, 720, 'T = ' + str(round(TGlycolOut - 273.15, 2)) + '°C', ha='left', va='bottom', fontsize=8)
    plt.text(160, 1000, 'T = ' + str(round(TGlycolOut - 273.15, 2)) + '°C', ha='left', va='bottom', fontsize=8)
    plt.text(560, 1000, 'T = ' + str(round(TGlycolIn - 273.15, 2)) + '°C', ha='left', va='bottom', fontsize=8)
    plt.text(560, 720, 'T = ' + str(round(TGlycolIn - 273.15, 2)) + '°C', ha='left', va='bottom', fontsize=8)

    # air temperatures
    plt.text(440, 160, 'T_Cond,out = ' + str(round(data[8,2] - 273.15, 2)) + '°C', ha='center', va='top', fontsize=8)
    plt.text(440, 90, 'T_amb = ' + str(round(data[8,1] - 273.15, 2)) + '°C', ha='center', va='bottom', fontsize=8)
    plt.text(440, 980, 'T_box = ' + str(round(data[7,3] - 273.15, 2)) + '°C', ha='center', va='bottom', fontsize=8)
    plt.text(440, 1080, 'T_HEX,out = ' + str(round(data[8,0] - 273.15, 2)) + '°C', ha='center', va='top', fontsize=8)

    plt.subplot(222)
    hL = np.zeros(100)
    hG = np.zeros(100)
    P = np.linspace(1e5, CPPSI("PCRIT", "R290"), 100) * 1e-5

    for i in range(100):
        hL[i] = CPPSI("H", "P", P[i] * 1e5, "Q", 0, ref)
        hG[i] = CPPSI("H", "P", P[i] * 1e5, "Q", 1, ref)

    plt.semilogy(hL, P, 'grey')
    plt.semilogy(hG, P, 'grey')
    plt.semilogy([data[1][2], data[2][2]], [data[1][1] * 1e-5, data[2][1]* 1e-5], 'red')
    plt.semilogy([data[2][2], data[3][2]], [data[2][1] * 1e-5, data[3][1]* 1e-5], 'red')
    plt.semilogy([data[3][2], data[4][2]], [data[3][1] * 1e-5, data[4][1]* 1e-5], 'r')
    plt.semilogy([data[4][2], data[1][2]], [data[4][1] * 1e-5, data[1][1]* 1e-5], 'r')
    plt.xlabel('Enthalpy')
    plt.ylabel('log(p)')
    plt.grid()

    plt.subplot(224)

    sL = np.zeros(100)
    sG = np.zeros(100)
    T = np.linspace(200, CPPSI("TCRIT", "R290"), 100) - 273.15

    for i in range(100):
        sL[i] = CPPSI("S", "T", T[i] + 273.15, "Q", 0, ref)
        sG[i] = CPPSI("S", "T", T[i] + 273.15, "Q", 1, ref)
    s1 = data[1][3]
    s2 = data[2][3]
    s2l = CPPSI("S", "P", data[2][1], "Q", 1, ref)
    s3 = data[3][3]
    s4 = data[4][3]
    s7l = CPPSI("S", "P", data[1][1], "Q", 1, ref)
    s5 = data[9][3]
    s6 = data[10][3]
    s7 = data[11][3]
    # s7l = CPPSI("S", "P", data[1][1], "")

    plt.plot(sL, T, 'grey')
    plt.plot(sG, T, 'grey')
    plt.plot([s1, s2], [data[1][0] - 273.15, data[2][0] - 273.15], 'red')
    plt.plot([s2, s2l], [data[2][0] - 273.15, data[3][0] - 273.15], 'red')
    plt.plot([s2l, s3], [data[3][0] - 273.15, data[3][0] - 273.15], 'red')
    plt.plot([s3, s6], [data[3][0] - 273.15, data[9][0] - 273.15], 'red')
    plt.plot([s6, s7l], [data[9][0] - 273.15, data[10][0] - 273.15], 'red')
    plt.plot([s7l, s1], [data[10][0] - 273.15, data[1][0] - 273.15], 'red')
    plt.xlabel('Entropy')
    plt.ylabel('Temperature')
    plt.grid()

    plt.show()
    # plt.hold(True)

def IHX(x, args):
    # This model calculates one side of an IHX, for the complete model, two of these models have to be used.
    # x = [T_A_out, T_B_out]
    kA = args[0]
    T_B_in = args[1]  # Inlet temperature of b side
    mdot_B = args[2]  # Inlet mass flow of b side
    mdot_A = args[3]  # Inlet mass flow of a side
    ref = args[4]  # Refrigerant string
    T_A_in = args[5]  # Inlet temperature of a side
    p_A = args[6]
    p_B = args[7]

    cp_A = CPPSI("CPMASS", "T", T_A_in+273.15, "P", p_A, ref)
    cp_B = CPPSI("CPMASS", "T", T_B_in+273.15, "P", p_B, ref)

    if T_A_in > T_B_in:
        Thi = T_A_in
        Tho = x[0]
        Tci = T_B_in
        Tco = x[1]
    else:
        Thi = T_B_in
        Tho = x[1]
        Tci = T_A_in
        Tco = x[0]

    LMTD = LMTD_calc(Thi, Tho, Tci, Tco)
    Qdot = kA * LMTD

    f = np.zeros(2)
    if T_A_in > T_B_in:
        f[0] = mdot_A * cp_A * (T_A_in - x[0]) - Qdot  # energy balance side A
        f[1] = mdot_B * cp_B * (T_B_in - x[1]) + Qdot  # energy balance side B
    else:
        f[0] = mdot_A * cp_A * (T_A_in - x[0]) + Qdot  # energy balance side A
        f[1] = mdot_B * cp_B * (T_B_in - x[1]) - Qdot  # energy balance side B

    return f

def calc_hexfan(k1_hex, k2_hex, dp_max, Vdot_max, Pel_max):
    pass