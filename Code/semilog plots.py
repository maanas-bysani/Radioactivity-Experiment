# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:29:30 2023

@author: Maanas
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:47:58 2023

@author: Maanas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import splrep, BSpline
import scipy.stats as stats
import scipy

plt.rcParams["figure.figsize"] = (12,8)

#%%

def find_thickness(raw_measurment):
    thickness = (raw_measurment + zero_error)*least_count
    return thickness

sample_time = 5
no_of_cycles = 10

zero_error = +2
least_count = 0.01
thickness_error = 0.01

gamma_background_per_cycle = 0 #2

gamma_background_per_cycle_al = 4.2 #2
gamma_background_per_cycle_cu = 2.4 #2

background_per_cycle = 1.3

al_density = 2.70
cu_density = 8.96

#%%

al_data = pd.read_csv('al.csv')
al_count_per_cycle = al_data['count']
al_thickness_measurement = al_data['thickness']

al_thickness = find_thickness(al_thickness_measurement)
al_thickness_measurement_areal_density = al_thickness*al_density/10 


cu_data = pd.read_csv('cu.csv')
cu_count_per_cycle = cu_data['count']
cu_thickness_measurement = cu_data['thickness']

cu_thickness = find_thickness(cu_thickness_measurement)
cu_thickness_measurement_areal_density = cu_thickness*cu_density/10 

#%% aluminium final
plt.rcParams["figure.figsize"] = (14,8)

background_per_cycle = 1.4

plt.errorbar(al_thickness_measurement_areal_density, al_count_per_cycle, np.sqrt(al_count_per_cycle), thickness_error, fmt = 'none', zorder = 1, color = 'black')
plt.scatter(al_thickness_measurement_areal_density,al_count_per_cycle, zorder = 10, label = 'Data', s=30, marker='o')

plt.fill_between([0,0.2673], -999, 999999, alpha = 0.12, label = 'Region 1; Low Energy \u03b2', color='red')
plt.fill_between([0.2673,0.918], -999, 999999, alpha = 0.12, label = 'Region 2; High Energy \u03b2', color='green')
plt.fill_between([0.918,1.6], -999, 999999, alpha = 0.12, label = 'Region 3; \u03b3-ray', color='goldenrod')

plt.axvline(x=0.2673, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)
plt.axvline(x=0.918, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)

plt.axhline(y=background_per_cycle, xmin=0, xmax=1, color = 'black', linestyle = 'dashed', label='Background')


m = -8.1237469309661927
c = 9.625463475728742

m_err = 0.94
c_err = 0.44

m_max = m+m_err
m_min = m-m_err

c_max = c+c_err
c_min = c-c_err

x_dummy = np.array(np.arange(0.2673,1.1442,0.001))
y_actual = m*x_dummy + c

y_max = m_max*x_dummy+c_max
y_min = m_min*x_dummy+c_min

plt.fill_between(x_dummy, np.exp(y_min), np.exp(y_max), alpha = 0.2, label = 'Uncertainty', color='grey')

plt.plot(x_dummy, np.exp(y_actual))

plt.axvline(x=1.1442, ymin=0, ymax=1, color = 'black', linestyle = 'solid', alpha = 1, label='Range 'R'')

plt.yscale('log', base=np.exp(1))

ticks = [np.exp(i) for i in range(0,10)]
dic = {np.exp(0) : "0", np.exp(1) : "1", np.exp(2) : "2", np.exp(3) : "3", np.exp(4) : "4", np.exp(5) : "5", np.exp(6) : "6", np.exp(7) : "7", np.exp(8) : "8", np.exp(9) : "9", np.exp(10) : "10"}
labels = [dic.get(t, ticks[i]) for i,t in enumerate(ticks)]
plt.yticks(ticks, labels)

plt.ylim(0,np.exp(9))
plt.xlim(0,1.6)
plt.grid()

plt.title('Semilog Absorption Plot for Aluminium', fontsize=18)
plt.ylabel('ln(counts per cycle)', fontsize=14)
plt.xlabel('Areal Density g/cm\u00b2', fontsize=14)

plt.legend()
plt.tight_layout()
plt.savefig('aluminium.png', dpi=1200, bbox_inches='tight')
plt.show()


#%% copper final

background_per_cycle = 0.75

plt.errorbar(cu_thickness_measurement_areal_density, cu_count_per_cycle, np.sqrt(cu_count_per_cycle), thickness_error, fmt = 'none', zorder = 1, color = 'black')
plt.scatter(cu_thickness_measurement_areal_density,cu_count_per_cycle, zorder = 10, label = 'Data', s=30, marker='o')

plt.fill_between([0,0.34048], -999, 999999, alpha = 0.12, label = 'Region 1; Low Energy \u03b2', color='red')
plt.fill_between([0.34048,0.87808], -999, 999999, alpha = 0.12, label = 'Region 2; High Energy \u03b2', color='green')
plt.fill_between([0.87808,1.5], -999,999999, alpha = 0.12, label = 'Region 3; \u03b3-ray', color='goldenrod')

plt.axvline(x=0.34048, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)
plt.axvline(x=0.87808, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)

plt.axhline(y=background_per_cycle, xmin=0, xmax=1, color = 'black', linestyle = 'dashed', label='Background')

# actual
c = 9.36934
m = -8.77809

# m = -9.76765955
# c = 9.79596257

# m_err = 1.25326
# c_err = 0.77

m_max = m+m_err
m_min = m-m_err

c_max = c+c_err
c_min = c-c_err

x_dummy = np.array(np.arange(0.34048,1.1005,0.001))
y_actual = m*x_dummy + c

y_max = m_max*x_dummy+c_max
y_min = m_min*x_dummy+c_min

plt.fill_between(x_dummy, np.exp(y_min), np.exp(y_max), alpha = 0.2, label = 'Uncertainty', color='grey')

plt.plot(x_dummy, np.exp(y_actual))

plt.axvline(x=1.1005, ymin=0, ymax=1, color = 'black', linestyle = 'solid', alpha = 1, label='Range 'R'')

plt.yscale('log', base=np.exp(1))

ticks = [np.exp(i) for i in range(0,10)]
dic = {np.exp(0) : "0", np.exp(1) : "1", np.exp(2) : "2", np.exp(3) : "3", np.exp(4) : "4", np.exp(5) : "5", np.exp(6) : "6", np.exp(7) : "7", np.exp(8) : "8", np.exp(9) : "9", np.exp(10) : "10"}
labels = [dic.get(t, ticks[i]) for i,t in enumerate(ticks)]
plt.yticks(ticks, labels)


plt.ylim(0,np.exp(9))
plt.xlim(0,1.5)
plt.grid()

plt.title('Semilog Absorption Plot for Copper', fontsize=18)
plt.ylabel('ln(counts per cycle)', fontsize=14)
plt.xlabel('Areal Density g/cm\u00b2', fontsize=14)

plt.legend()
plt.tight_layout()
plt.savefig('copper.png', dpi=1200, bbox_inches='tight')
plt.show()


#%% happy with these limits - cu: 13-28

# plt.errorbar(cu_thickness_measurement_areal_density, cu_count_per_cycle, np.sqrt(cu_count_per_cycle), thickness_error, fmt = 'none', zorder = 1, color = 'black')
# plt.scatter(cu_thickness_measurement_areal_density,cu_count_per_cycle, zorder = 10, label = 'Data', s=30, marker='o')

# plt.yscale('log', base=np.exp(1))

# ticks = [np.exp(i) for i in range(0,10)]
# dic = {np.exp(0) : "0", np.exp(1) : "1", np.exp(2) : "2", np.exp(3) : "3", np.exp(4) : "4", np.exp(5) : "5", np.exp(6) : "6", np.exp(7) : "7", np.exp(8) : "8", np.exp(9) : "9", np.exp(10) : "10"}
# labels = [dic.get(t, ticks[i]) for i,t in enumerate(ticks)]
# plt.yticks(ticks, labels)

# plt.fill_between([0,0.34048], -999, np.exp(9), alpha = 0.12, label = 'Region 1; Low Energy \u03b2', color='red')
# plt.fill_between([0.34048,0.87808], -999, np.exp(9), alpha = 0.12, label = 'Region 2; High Energy \u03b2', color='green')
# plt.fill_between([0.87808,1.5], -999, np.exp(9), alpha = 0.12, label = 'Region 3; \u03b3-ray', color='goldenrod')

# plt.axvline(x=0.34048, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)
# plt.axvline(x=0.87808, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)

# plt.axhline(y=0.85, xmin=0, xmax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)


# # m = -7.415169813691641
# # c = 9.26406371297965

# # m = -10.1
# # c = 10

# # m = -9.7286
# # c = 9.79596257

# # c = 10.49596257


# m = -9.76765955
# c = 9.79596257

# # actual:

# c = 9.36934
# m = -8.77809



# m_err = 1.25326
# c_err = 0.77

# m_max = m+m_err
# m_min = m-m_err

# c_max = c+c_err
# c_min = c-c_err


# x_dummy = np.array(np.arange(0.34048,1.0302,0.01))
# y_actual = m*x_dummy + c

# y_max = m_max*x_dummy+c_max
# y_min = m_min*x_dummy+c_min

# plt.fill_between(x_dummy, np.exp(y_min), np.exp(y_max), alpha = 0.2, label = 'Error', color='grey')


# plt.plot(x_dummy, np.exp(y_actual))
# plt.ylim(0,np.exp(9))
# plt.xlim(0,1.5)
# plt.grid()

# plt.title('Semilog absorption plot for Copper')
# plt.ylabel('ln(counts per cycle)')
# plt.xlabel('Areal Density g/cm\u00b2')

# plt.legend()
# plt.tight_layout()
# plt.show()


#%% happy with these limits - al: 15-40

# background_per_cycle = 1.4

# plt.errorbar(al_thickness_measurement_areal_density, al_count_per_cycle, np.sqrt(al_count_per_cycle), thickness_error, fmt = 'none', zorder = 1, color = 'black')
# plt.scatter(al_thickness_measurement_areal_density,al_count_per_cycle, zorder = 10, label = 'Data', s=30, marker='o')
# plt.yscale('log', base=np.exp(1))

# ticks = [np.exp(i) for i in range(0,10)]
# dic = {np.exp(0) : "0", np.exp(1) : "1", np.exp(2) : "2", np.exp(3) : "3", np.exp(4) : "4", np.exp(5) : "5", np.exp(6) : "6", np.exp(7) : "7", np.exp(8) : "8", np.exp(9) : "9", np.exp(10) : "10"}
# labels = [dic.get(t, ticks[i]) for i,t in enumerate(ticks)]
# plt.yticks(ticks, labels)

# plt.fill_between([0,0.2673], -999, np.exp(9), alpha = 0.12, label = 'Region 1; Low Energy \u03b2', color='red')
# plt.fill_between([0.2673,0.918], -999, np.exp(9), alpha = 0.12, label = 'Region 2; High Energy \u03b2', color='green')
# plt.fill_between([0.918,1.6], -999, np.exp(9), alpha = 0.12, label = 'Region 3; \u03b3-ray', color='goldenrod')

# plt.axvline(x=0.2673, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)
# plt.axvline(x=0.918, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)

# plt.axhline(y=background_per_cycle, xmin=0, xmax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)

# m = -1.93553026
# c =  9.28536773

# m = -2.281903966013556

# m = -8.1
# c = 9.6

# m = -2.68840855

# m = -8.1237469309661927
# c = 9.625463475728742

# # c = 10.49596257
# # m = -2.281903966013556
# # c = 12.131927454416813

# x_dummy = np.array(np.arange(0.2673,1.1673,0.01))
# y_dummy = m*x_dummy + c

# plt.plot(x_dummy, np.exp(y_dummy))

# plt.ylim(0,np.exp(9))
# plt.xlim(0,1.6)
# plt.grid()

# plt.title('Semilog absorption plot for Aluminium')
# plt.ylabel('ln(counts per cycle)')
# plt.xlabel('Areal Density g/cm\u00b2')

# plt.legend()
# plt.tight_layout()
# plt.show()



#%%
# # plt.scatter(al_thickness_measurement_areal_density, np.log(al_count_per_cycle))

# # plt.errorbar(al_thickness_measurement_areal_density, al_count_per_cycle, np.sqrt(al_count_per_cycle), thickness_error, fmt = 'none', zorder = 1, color = 'black')


# # locs, labels = plt.yticks()

# # labels = [i for i in range(0,len(locs))]

# ## or 
# labels = [dic.get(t, ticks[i]) for i,t in enumerate(ticks)]


# plt.yticks(ticks, labels)
# # plt.yticklabels(labels)

# # plt.errorbar(xs, ys, np.array(count_error, dtype='f'), np.array(thickness_error, dtype='f'), fmt = 'none', zorder = 1, color = 'red')

# # plt.fill_between(np.arange(0, xs[best_partition_index[0]], 0.01), -999, 999, alpha = 0.1, label = 'Region/Grouping 1')

# plt.fill_between([0,0.75], -999, np.exp(9), alpha = 0.1, label = 'Region/Grouping 1')

# plt.legend()
# plt.tight_layout()
# plt.show()
#%%
