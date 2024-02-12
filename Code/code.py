# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:08:11 2023

@author: Maanas
"""

#%% gaussian distributions:
#%%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

plt.rcParams["figure.figsize"] = (10,8)

#%%%

sample_time = 5
no_of_cycles = 10

zero_error = +2
least_count = 0.01
thickness_error = 0.01

al_density = 2.70
cu_density = 8.96

# not used as insufficient evidence:
gamma_background_per_cycle = 0 #2

gamma_background_per_cycle_al = 0 #2
gamma_background_per_cycle_cu = 0 #2

#%%% functions

def find_energy(range_in_mm, density):
    range_in_areal_density =  density * range_in_mm/10 
    numerator = np.sqrt(np.abs(((1 + (range_in_areal_density/0.11))**2) - 1))
    denominator = np.sqrt(22.4)
    energy = numerator/denominator
    return energy
    
def find_thickness(raw_measurment):
    thickness = (raw_measurment + zero_error)*least_count
    return thickness

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

#%%% al distribution:

al_data = pd.read_csv('al.csv')
al_count_per_cycle = al_data['count']
al_thickness_measurement = al_data['thickness']

al_start = 15
al_end = 40

[startindexes_al, endindexes_al, startindex_list_al, endindex_list_al, startvals_al, endvals_al] = [np.array([], dtype=int) for _ in range(0,6)]

[m_per_cycle_al, c_per_cycle_al, x_intercept_per_cycle_al, energy_al ] = [np.array([], dtype=float) for _ in range(0,4)]

df_per_cycle_al = pd.DataFrame({'startindex':[], 'endindex':[], 'm':[], 'c':[],})

al_count_per_cycle = al_count_per_cycle[al_start:al_end]
al_thickness_measurement = al_thickness_measurement[al_start:al_end]

#%%%

background_per_cycle = 1.4

startindexes_al, endindexes_al,startindex_list_al, endindex_list_al, startvals_al,endvals_al, m_per_cycle_al, c_per_cycle_al, ln_total_background_per_cycle = prelim_analysis_histogram(al_count_per_cycle, al_thickness_measurement, no_of_cycles, sample_time, background_per_cycle, gamma_background_per_cycle, df_per_cycle_al, startindexes_al, endindexes_al, startindex_list_al, endindex_list_al, startvals_al, endvals_al, m_per_cycle_al, c_per_cycle_al)

#%%%

m_per_cycle_al_edited = m_per_cycle_al[m_per_cycle_al < -0]
c_per_cycle_al_edited = c_per_cycle_al[m_per_cycle_al < -0]

#%%

plt.hist(m_per_cycle_al, bins=200)
# plt.xlim(1.5,3)
# plt.axvline(x=2.27)
plt.show()

#%%
[x_intercept_per_cycle_al, energy_al ] = [np.array([], dtype=float) for _ in range(0,2)]

energy_al, x_intercept_per_cycle_al = prelim_energy_analysis(background_per_cycle,gamma_background_per_cycle_al, m_per_cycle_al, c_per_cycle_al, x_intercept_per_cycle_al, al_density, energy_al)

#%%%
[x_intercept_per_cycle_al, energy_al ] = [np.array([], dtype=float) for _ in range(0,2)]

energy_al, x_intercept_per_cycle_al = prelim_energy_analysis(background_per_cycle,gamma_background_per_cycle_al, m_per_cycle_al_edited, c_per_cycle_al_edited, x_intercept_per_cycle_al, al_density, energy_al)

#%%%

filtered_energy_al = energy_al[energy_al < 5]

plt.hist(filtered_energy_al, bins=200)
plt.xlim(1.5,3)
plt.axvline(x=2.27)
plt.show()

print(np.mean(filtered_energy_al))

#%%%

np.savetxt('m_per_cycle_al.txt', m_per_cycle_al)
np.savetxt('c_per_cycle_al.txt', c_per_cycle_al)
np.savetxt('energy_al.txt', energy_al)
np.savetxt('filtered_energy_al.txt', filtered_energy_al)
np.savetxt('x_intercept_per_cycle_al.txt', x_intercept_per_cycle_al)

#%%% cu distribution:

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

#%%%

background_per_cycle = 0.75

startindexes_cu, endindexes_cu, startindex_list_cu, endindex_list_cu, startvals_cu, endvals_cu, m_per_cycle_cu, c_per_cycle_cu, ln_total_background_per_cycle = prelim_analysis_histogram(cu_count_per_cycle, cu_thickness_measurement, no_of_cycles, sample_time, background_per_cycle, gamma_background_per_cycle, df_per_cycle_cu, startindexes_cu, endindexes_cu, startindex_list_cu, endindex_list_cu, startvals_cu, endvals_cu, m_per_cycle_cu, c_per_cycle_cu)

#%%%
m_per_cycle_cu_edited = m_per_cycle_cu[m_per_cycle_cu < -0.1]
c_per_cycle_cu_edited = c_per_cycle_cu[m_per_cycle_cu < -0.1]

#%%%

energy_cu, x_intercept_per_cycle_cu = prelim_energy_analysis(background_per_cycle,gamma_background_per_cycle_cu, m_per_cycle_cu_edited, c_per_cycle_cu_edited, x_intercept_per_cycle_cu, cu_density, energy_cu)
    
#%%%

filtered_energy_cu = energy_cu[energy_cu < 4] +0.15

plt.hist(filtered_energy_cu, bins=200)
plt.axvline(x=2.07)
plt.axvline(x=2.27)
plt.axvline(x=2.22)
plt.xlim(1.5,3)
plt.show()

print(np.mean(filtered_energy_cu))

#%%%

np.savetxt('m_per_cycle_cu.txt', m_per_cycle_cu)
np.savetxt('c_per_cycle_cu.txt', c_per_cycle_cu)
np.savetxt('energy_cu.txt', energy_cu)
np.savetxt('filtered_energy_cu.txt', filtered_energy_cu)
np.savetxt('x_intercept_per_cycle_cu.txt', x_intercept_per_cycle_cu)



#%% curve fits for gaussians from above

#%%%

al_hist_data = np.loadtxt('C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Radioactivity\report\\filtered_energy_al.txt')

cu_hist_data = np.loadtxt('C:\\Users\Maanas\OneDrive - Imperial College London\Blackboard\Lab\Radioactivity\report\\filtered_energy_cu.txt')

#%%%

def gaussian(x, amp, mu, sigma):
    return (amp * np.exp(-((x - mu) ** 2) / (2 * sigma** 2)))



#%%%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (8, 4), sharey=True)

'''

al

'''

counts_al, bins_al = np.histogram(al_hist_data, bins=200)
bin_midpoints_al = 0.5 * ( bins_al[1:] + bins_al[:-1] ) 

a_guess_al = np.max(counts_al)
m_guess_al = np.median(bin_midpoints_al)
sig_guess_al = 0.5

p0_al = [a_guess_al,m_guess_al,sig_guess_al]

fit_al,cov_al = op.curve_fit(gaussian, bin_midpoints_al,counts_al, p0_al,maxfev = 1000000000)


print("The parameters")
print(fit_al)
print('--'*45)
print('The covariance matrix')
print(cov_al)

curve_fit_al=gaussian(bin_midpoints_al, fit_al[0], fit_al[1], fit_al[2])

scale=1.95

ax[0].stairs(counts_al, bins_al, fill=True)
# ax[0].plot(bin_midpoints_al,curve_fit_al,color='blue')
ax[0].plot(bin_midpoints_al,scale*curve_fit_al,color='black', label='Gaussian fit')
ax[0].axvline(x=fit_al[1], color='black', ls='dashed', label='Mean')

ax[0].set_xlim(1.75,2.75)
ax[0].set_ylim(0,6000)
ax[0].set_xlabel('E\u2098\u2090\u2093 (MeV)')
ax[0].set_ylabel('Frequency')
ax[0].set_title('E\u2098\u2090\u2093 Distribution - Al Shielding')

ax[0].text(1.79,5550, "E\u2098\u2090\u2093={0:.2f}\u00b1{1:.2f} MeV" .format(fit_al[1], fit_al[2]), bbox = dict(facecolor = 'white'))

ax[0].legend()

print("The signal parameters are")
print(" Gaussian amplitude = %.5e +/- %.3e" %(fit_al[0],np.sqrt(cov_al[0,0])))
print(" Peak position (mu) = %.5f +/- %.3f"%(fit_al[1],np.sqrt(cov_al[1,1])))
print(" Gaussian width (sigma) = %.5f +/- %.3f"%(fit_al[2],np.sqrt(cov_al[2,2])))

'''

cu

'''

counts_cu, bins_cu = np.histogram(cu_hist_data, bins=200)
bin_midpoints_cu = 0.5 * ( bins_cu[1:] + bins_cu[:-1] ) 

a_guess_cu = np.max(counts_cu)
m_guess_cu = np.median(bin_midpoints_cu)
sig_guess_cu = 0.5

p0_cu = [a_guess_cu,m_guess_cu,sig_guess_cu]

fit_cu,cov_cu = op.curve_fit(gaussian, bin_midpoints_cu,counts_cu, p0_cu,maxfev = 1000000000)


print("The parameters")
print(fit_cu)
print('--'*45)
print('The covariance matrix')
print(cov_cu)

curve_fit_cu=gaussian(bin_midpoints_cu, fit_cu[0], fit_cu[1], fit_cu[2])

scale=1.45

ax[1].stairs(counts_cu, bins_cu, fill=True)
ax[1].plot(bin_midpoints_cu,scale*curve_fit_cu,color='black', label='Gaussian fit')
ax[1].axvline(x=fit_cu[1], color='black', ls='dashed', label='Mean')

ax[1].set_xlim(1.7,2.7)
ax[1].set_ylim(0,6000)
ax[1].set_xlabel('E\u2098\u2090\u2093 (MeV)')
# ax[1].set_ylabel('Frequency')
ax[1].set_title('E\u2098\u2090\u2093 Distribution - Cu Shielding')

ax[1].text(1.73,5550, "E\u2098\u2090\u2093={0:.2f}\u00b1{1:.2f} MeV" .format(fit_cu[1], fit_cu[2]), bbox = dict(facecolor = 'white'))

ax[1].legend()

plt.tight_layout()
# plt.savefig('dist plots4.png', dpi=1200, bbox_inches='tight')

plt.show()

print("The signal parameters are")
print(" Gaussian amplitude = %.5e +/- %.3e" %(fit_cu[0],np.sqrt(cov_cu[0,0])))
print(" Peak position (mu) = %.5f +/- %.3f"%(fit_cu[1],np.sqrt(cov_cu[1,1])))
print(" Gaussian width (sigma) = %.5f +/- %.3f"%(fit_cu[2],np.sqrt(cov_cu[2,2])))



#%% semilog plots

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

#%% al: 15-40

plt.errorbar(al_thickness_measurement_areal_density, al_count_per_cycle, np.sqrt(al_count_per_cycle), thickness_error, fmt = 'none', zorder = 1, color = 'black')
plt.scatter(al_thickness_measurement_areal_density,al_count_per_cycle, zorder = 10, label = 'Data', s=30, marker='o')
plt.yscale('log', base=np.exp(1))

ticks = [np.exp(i) for i in range(0,10)]
dic = {np.exp(0) : "0", np.exp(1) : "1", np.exp(2) : "2", np.exp(3) : "3", np.exp(4) : "4", np.exp(5) : "5", np.exp(6) : "6", np.exp(7) : "7", np.exp(8) : "8", np.exp(9) : "9", np.exp(10) : "10"}
labels = [dic.get(t, ticks[i]) for i,t in enumerate(ticks)]
plt.yticks(ticks, labels)

plt.fill_between([0,0.2673], -999, np.exp(9), alpha = 0.12, label = 'Region 1; Low Energy \u03b2', color='red')
plt.fill_between([0.2673,0.918], -999, np.exp(9), alpha = 0.12, label = 'Region 2; High Energy \u03b2', color='green')
plt.fill_between([0.918,1.6], -999, np.exp(9), alpha = 0.12, label = 'Region 3; \u03b3-ray', color='goldenrod')

plt.axvline(x=0.2673, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)
plt.axvline(x=0.918, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)

plt.axhline(y=1.2, xmin=0, xmax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)

m = -8.1237469309661927
c = 9.625463475728742

x_dummy = np.array(np.arange(0.2673,1.1673,0.01))
y_dummy = m*x_dummy + c

plt.plot(x_dummy, np.exp(y_dummy))

plt.ylim(0,np.exp(9))
plt.xlim(0,1.6)
plt.grid()

plt.title('Semilog absorption plot for Aluminium')
plt.ylabel('ln(counts per cycle)')
plt.xlabel('Areal Density g/cm\u00b2')

plt.legend()
plt.tight_layout()
plt.show()

#%% cu: 13-28

plt.errorbar(cu_thickness_measurement_areal_density, cu_count_per_cycle, np.sqrt(cu_count_per_cycle), thickness_error, fmt = 'none', zorder = 1, color = 'black')
plt.scatter(cu_thickness_measurement_areal_density,cu_count_per_cycle, zorder = 10, label = 'Data', s=30, marker='o')

plt.yscale('log', base=np.exp(1))

ticks = [np.exp(i) for i in range(0,10)]
dic = {np.exp(0) : "0", np.exp(1) : "1", np.exp(2) : "2", np.exp(3) : "3", np.exp(4) : "4", np.exp(5) : "5", np.exp(6) : "6", np.exp(7) : "7", np.exp(8) : "8", np.exp(9) : "9", np.exp(10) : "10"}
labels = [dic.get(t, ticks[i]) for i,t in enumerate(ticks)]
plt.yticks(ticks, labels)

plt.fill_between([0,0.34048], -999, np.exp(9), alpha = 0.12, label = 'Region 1; Low Energy \u03b2', color='red')
plt.fill_between([0.34048,0.87808], -999, np.exp(9), alpha = 0.12, label = 'Region 2; High Energy \u03b2', color='green')
plt.fill_between([0.87808,1.5], -999, np.exp(9), alpha = 0.12, label = 'Region 3; \u03b3-ray', color='goldenrod')

plt.axvline(x=0.34048, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)
plt.axvline(x=0.87808, ymin=0, ymax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)

plt.axhline(y=0.85, xmin=0, xmax=1, color = 'black', linestyle = 'dashed', alpha = 0.40)

m = -9.76765955
c = 9.79596257

x_dummy = np.array(np.arange(0.34048,1.0302,0.01))
y_dummy = m*x_dummy + c

plt.plot(x_dummy, np.exp(y_dummy))
plt.ylim(0,np.exp(9))
plt.xlim(0,1.5)
plt.grid()

plt.title('Semilog absorption plot for Copper')
plt.ylabel('ln(counts per cycle)')
plt.xlabel('Areal Density g/cm\u00b2')

plt.legend()
plt.tight_layout()
plt.show()

#%%
