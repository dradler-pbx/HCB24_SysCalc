from scipy.interpolate import interpn
import numpy as np
import pandas as pd

def get_values_from_df(TC, T0, speed, df):
    mdot = np.zeros([len(TC), len(T0), len(speed)])
    power = np.zeros_like(mdot)
    for x, TC_val in enumerate(TC):
        df_TC = df[df['TC']==TC_val]
        for y, T0_val in enumerate(T0):
            df_T0 = df_TC[df_TC['T0']==T0_val]
            for z, speed_val in enumerate(speed):
                if speed_val in df_T0['speed'].values:
                    test = df_T0[df_T0['speed']==speed_val]['refrigerantmassflow'].values
                    mdot[x, y, z] = df_T0[df_T0['speed']==speed_val]['refrigerantmassflow'].values/3600
                    power[x, y, z] = df_T0[df_T0['speed']==speed_val]['powerconsumption'].values
                else:
                    mdot[x, y, z] = np.nan
                    power[x, y, z] = np.nan

    return mdot, power

# read the measurement data
file = "CompressorPerformanceMap.csv"
data = pd.read_csv(file, sep=';')

TC = data['TC'].drop_duplicates().values
TC.sort()
T0 = data['T0'].drop_duplicates().values
T0.sort()
speed = data['speed'].drop_duplicates().values
speed.sort()

(mdotR, power) = get_values_from_df(TC, T0, speed, data)

point = np.array([55, -15, 3600])
print(interpn(points=(TC, T0, speed), values=(mdotR), xi=point))


# example:
#arrays constituting the 3D grid
# def func_3d(X, Y, Z):
#     return 2*X+3*Y-Z
#
# x = np.linspace(0, 50, 50)
# y = np.linspace(0, 50, 50)
# z = np.linspace(0, 50, 50)
# points = (x, y, z)
# #generate a 3D grid
# X, Y, Z = np.meshgrid(x, y, z)
#
# #evaluate the function on the points of the grid
# values = func_3d(X, Y, Z)
#
# point = np.array([2.5, 3.5, 1.5])
#
# # points = the regular grid, #values =the data on the regular grid
# # point = the point that we want to evaluate in the 3D grid
# print(interpn(points, values, point))