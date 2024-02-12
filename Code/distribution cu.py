# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:55:39 2023

@author: Maanas
"""

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import splrep, BSpline
import scipy.stats as stats
import scipy
import scipy.optimize as op

plt.rcParams["figure.figsize"] = (10,8)

#%%

sample_time = 5
no_of_cycles = 10

zero_error = +2
least_count = 0.01
thickness_error = 0.01

gamma_background_per_cycle = 0 #2

gamma_background_per_cycle_cu = 2.4 #2

cu_density = 8.96


#%% functions

def find_energy(range_in_mm, density):
    range_in_areal_density =  density * range_in_mm/10 
    numerator = np.sqrt(np.abs(((1 + (range_in_areal_density/0.11))**2) - 1))
    denominator = np.sqrt(22.4)
    energy = numerator/denominator
    return energy
    
def find_thickness(raw_measurment):
    thickness = (raw_measurment + zero_error)*least_count
    return thickness

def gaussian(x, amp, mu, sigma):
    return (amp * np.exp(-((x - mu) ** 2) / (2 * sigma** 2)))


#%%

def prelim_analysis_histogram(count_per_cycle, thickness_measurement, no_of_cycles, sample_time, background_per_cycle, 
                    gamma_background_per_cycle, df, startindexes, endindexes, 
                    startindex_list, endindex_list, startvals, endvals, m_list, c_list):
    
    # startindex = endindex = startvals = endvals = m_per_cycle = c_per_cycle = np.array([], dtype=int)
    # df = pd.DataFrame({'startindex':[], 'endindex':[], 'm':[], 'c':[],})

    total_count = count_per_cycle*no_of_cycles
    count_per_sec = count_per_cycle/sample_time
    
    ln_count_per_cycle = np.log(count_per_cycle)
    count_per_cycle_error = 1/np.sqrt(count_per_cycle)
    
    ln_count_per_sec = np.log(count_per_sec)
    count_per_sec_error = 1/np.sqrt(count_per_sec)
    
    ln_count_total = np.log(total_count)
    count_total_error = 1/np.sqrt(total_count)
    
    total_background_per_cycle = background_per_cycle + gamma_background_per_cycle
    ln_total_background_per_cycle = np.log(total_background_per_cycle)
    
    total_background = total_background_per_cycle*no_of_cycles
    ln_total_background = np.log(total_background)
    
    total_background_per_sec = total_background_per_cycle/sample_time
    ln_total_background_per_sec = np.log(total_background_per_sec)
    
    thickness = find_thickness(thickness_measurement)
    
    for i in range(0,len(thickness)+1):
        for j in range(0,len(thickness)+1):
            diff = i-j
            if np.absolute(diff) > 2:# (len(thickness)+1)/4:            
                startindexes = np.append(startindexes, i)
                endindexes = np.append(endindexes, j)
                
    row_count = 0

    for s in startindexes:
        for e in endindexes:
            filter_thickness = thickness[s:e]
            filter_count_per_cycle = count_per_cycle[s:e]
            
            x = filter_thickness
            y = np.log(filter_count_per_cycle)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            
            # df['startindex'][row_count] = s
            # df['endindex'][row_count] = e
            # df['m'][row_count] = m
            # df['c'][row_count] = c


            startindex_list = np.append(startindex_list, s)
            endindex_list = np.append(endindex_list,e)
            m_list = np.append(m_list,m)
            c_list = np.append(c_list,c)
            
            print(row_count)

            row_count = row_count + 1

    return startindexes, endindexes, startindex_list, endindex_list, startvals,endvals, m_list, c_list, ln_total_background_per_cycle


def prelim_energy_analysis(background_per_cycle, gamma_background_per_cycle, m_list, c_list, x_intercept, density, energy_list):         

    total_background_per_cycle = background_per_cycle #+ gamma_background_per_cycle
    ln_total_background_per_cycle = np.log(total_background_per_cycle)
    
    for i,j in zip(m_list, c_list):
        if i != 0:
            intercept = (ln_total_background_per_cycle-j)/i
            x_intercept = np.append(x_intercept, intercept)

    energy_list = np.append(energy_list,find_energy(x_intercept, density))
    
    return energy_list, x_intercept

#%%

cu_data = pd.read_csv('cu.csv')
cu_count_per_cycle = cu_data['count']
cu_thickness_measurement = cu_data['thickness']

cu_start = 10
cu_end = 30

[startindexes_cu, endindexes_cu, startindex_list_cu, endindex_list_cu, startvals_cu, endvals_cu] = [np.array([], dtype=int) for _ in range(0,6)]

[m_per_cycle_cu, c_per_cycle_cu, x_intercept_per_cycle_cu, energy_cu] = [np.array([], dtype=float) for _ in range(0,4)]

df_per_cycle_cu = pd.DataFrame({'startindex':[], 'endindex':[], 'm':[], 'c':[],})

cu_count_per_cycle = cu_count_per_cycle[cu_start:cu_end].reset_index(drop=True)
cu_thickness_measurement = cu_thickness_measurement[cu_start:cu_end].reset_index(drop=True)

#%%

background_per_cycle = 0.75

startindexes_cu, endindexes_cu, startindex_list_cu, endindex_list_cu, startvals_cu, endvals_cu, m_per_cycle_cu, c_per_cycle_cu, ln_total_background_per_cycle = prelim_analysis_histogram(cu_count_per_cycle, cu_thickness_measurement, no_of_cycles, sample_time, background_per_cycle, gamma_background_per_cycle, df_per_cycle_cu, startindexes_cu, endindexes_cu, startindex_list_cu, endindex_list_cu, startvals_cu, endvals_cu, m_per_cycle_cu, c_per_cycle_cu)

#%%

energy_cu, x_intercept_per_cycle_cu = prelim_energy_analysis(background_per_cycle,gamma_background_per_cycle_cu, m_per_cycle_cu, c_per_cycle_cu, x_intercept_per_cycle_cu, cu_density, energy_cu)

#%%

m_per_cycle_cu_edited = m_per_cycle_cu[m_per_cycle_cu < -0]
c_per_cycle_cu_edited = c_per_cycle_cu[m_per_cycle_cu < -0]

#%%

energy_cu2 = np.array([], dtype=float)
x_intercept_per_cycle_cu2 = np.array([], dtype=float)

#%%

energy_cu2, x_intercept_per_cycle_cu2 = prelim_energy_analysis(background_per_cycle,gamma_background_per_cycle_cu, m_per_cycle_cu_edited, c_per_cycle_cu_edited, x_intercept_per_cycle_cu2, cu_density, energy_cu2)

#%%

plt.hist(m_per_cycle_cu, bins=250)
plt.xlim(-20,-0.1)
# plt.ylim(0,15000)
plt.axvline(x=-0.1)
plt.show()

plt.hist(m_per_cycle_cu_edited, bins=200)
plt.xlim(-20,-0.1)
# plt.ylim(0,15000)
plt.show()

#%%

plt.hist(energy_cu, bins=200)
plt.xlim(1.5,3)
plt.axvline(x=2.27)
plt.show()

plt.hist(energy_cu2, bins=200)
plt.xlim(1.5,3)
plt.axvline(x=2.27)
plt.show()

#%%

filtered_energy_cu = energy_cu[energy_cu < 5]

plt.hist(filtered_energy_cu, bins=200)
plt.xlim(1.5,3)
plt.axvline(x=2.27)
plt.axvline(x=2.24)
plt.axvline(x=2.03)
plt.axvline(x=2.14)
plt.show()

#%%

counts_cu, bins_cu = np.histogram(filtered_energy_cu, bins=200)
bin_midpoints_cu = 0.5 * ( bins_cu[1:] + bins_cu[:-1] ) 


a_guess = np.max(counts_cu)
m_guess = np.median(bin_midpoints_cu)
sig_guess = 0.5

p0 = [a_guess,m_guess,sig_guess]

fit,cov = op.curve_fit(gaussian, bin_midpoints_cu,counts_cu, p0,maxfev = 1000000000)

print("The parameters")
print(fit)
print('--'*45)
print('The covariance matrix')
print(cov)

curve_fit=gaussian(bin_midpoints_cu, fit[0], fit[1], fit[2])

scale=1.46

plt.stairs(counts_cu, bins_cu)
plt.plot(bin_midpoints_cu,curve_fit,color='black')
plt.plot(bin_midpoints_cu,scale*curve_fit,color='purple')
plt.show()

print("The signal parameters are")
print(" Gaussian amplitude = %.5e +/- %.3e" %(fit[0],np.sqrt(cov[0,0])))
print(" Peak position (mu) = %.5f +/- %.3f"%(fit[1],np.sqrt(cov[1,1])))
print(" Gaussian width (sigma) = %.5f +/- %.3f"%(fit[2],np.sqrt(cov[2,2])))

#%%

print(np.mean(filtered_energy_cu))

#%%

np.savetxt('m_per_cycle_cu.txt', m_per_cycle_cu)
np.savetxt('c_per_cycle_cu.txt', c_per_cycle_cu)
np.savetxt('energy_cu.txt', energy_cu)
np.savetxt('filtered_energy_cu.txt', filtered_energy_cu)
np.savetxt('x_intercept_per_cycle_cu.txt', x_intercept_per_cycle_cu)

#%%

counts_al, bins_al = np.histogram(m_per_cycle_cu_edited, bins=200)
bin_midpoints_al = 0.5 * ( bins_al[1:] + bins_al[:-1] ) 


a_guess = 1552
m_guess = -10
sig_guess = 0.02

p0 = [a_guess,m_guess,sig_guess]

fit,cov = op.curve_fit(gaussian, bin_midpoints_al,counts_al, p0,maxfev = 1000000000)

print("The parameters")
print(fit)
print('--'*45)
print('The covariance matrix')
print(cov)

curve_fit=gaussian(bin_midpoints_al, fit[0], fit[1], fit[2])

scale=1.95

plt.stairs(counts_al, bins_al)
plt.plot(bin_midpoints_al,curve_fit,color='black')
plt.plot(bin_midpoints_al,scale*curve_fit,color='purple')
plt.show()

print("The signal parameters are")
print(" Gaussian amplitude = %.5e +/- %.3e" %(fit[0],np.sqrt(cov[0,0])))
print(" Peak position (mu) = %.5f +/- %.3f"%(fit[1],np.sqrt(cov[1,1])))
print(" Gaussian width (sigma) = %.5f +/- %.3f"%(fit[2],np.sqrt(cov[2,2])))

#%%


counts_al, bins_al = np.histogram(c_per_cycle_cu_edited, bins=200)
bin_midpoints_al = 0.5 * ( bins_al[1:] + bins_al[:-1] ) 

a_guess = 3000
m_guess = 8
sig_guess = 0.1

p0 = [a_guess,m_guess,sig_guess]

fit,cov = op.curve_fit(gaussian, bin_midpoints_al,counts_al, p0,maxfev = 1000000000)

print("The parameters")
print(fit)
print('--'*45)
print('The covariance matrix')
print(cov)

curve_fit=gaussian(bin_midpoints_al, fit[0], fit[1], fit[2])

scale=1.95

plt.stairs(counts_al, bins_al)
plt.plot(bin_midpoints_al,curve_fit,color='black')
plt.plot(bin_midpoints_al,scale*curve_fit,color='purple')
plt.show()

print("The signal parameters are")
print(" Gaussian amplitude = %.5e +/- %.3e" %(fit[0],np.sqrt(cov[0,0])))
print(" Peak position (mu) = %.5f +/- %.3f"%(fit[1],np.sqrt(cov[1,1])))
print(" Gaussian width (sigma) = %.5f +/- %.3f"%(fit[2],np.sqrt(cov[2,2])))