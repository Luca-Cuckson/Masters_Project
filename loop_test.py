import matplotlib.pyplot as plt

import numpy as np
import scraps as scr
import pprint as pp
import h5py as h5
import os
import re
import pandas as pd

import looping_funcs as lf
import fitting as fg


# ======================================================================================================= 
# LOAD IN DATA
# ======================================================================================================= 

dataFile = "D:/LucaC_L4_Project/Luca E2E/Luca E2E/sweep_20261022_132646.h5"

resObjs = lf.load_resonators_from_sweep(dataFile)

# check to see if resonator object created properly
print(
    resObjs[16].I,
    resObjs[16].Q,
    resObjs[16].freq,
    resObjs[16].name,
    resObjs[16].pwr,
    resObjs[16].temp
)


df = pd.read_csv("D:/LucaC_L4_Project/Luca E2E/Luca E2E/LT402#4_2026_09_12.csv")

#print(df)

# ======================================================================================================= 
# RETURN LOOPS TO ORIGINAL (PRE-TRANSLATION AND ROTATION) POSITIONS
# =======================================================================================================

def sendResHome(Resonator_list, sweep):

    thetas = sweep['Rotation (degrees)'] * np.pi / 180 
    I_centres = sweep['I Centre']
    Q_centres = sweep['Q Centre']

    check_list = []

    for i in range(len(Resonator_list)):

        #print(Resonator_list[i].freq[0])

        I_old = Resonator_list[i].I.copy()
        Q_old = Resonator_list[i].Q.copy()

        # use rotation matrix to send IQ points home
        Resonator_list[i].I = I_old*np.cos(thetas[i]) - Q_old*np.sin(thetas[i])
        Resonator_list[i].Q = I_old*np.sin(thetas[i]) + Q_old*np.cos(thetas[i])

        # shift centres!
        Resonator_list[i].I += I_centres[i]
        Resonator_list[i].Q += Q_centres[i]

        # time to check if I've thrown loops out of whack
        r_before = np.sqrt(I_old**2 + Q_old**2)
        r_after  = np.sqrt((Resonator_list[i].I - I_centres[i])**2 + (Resonator_list[i].Q - Q_centres[i])**2)
        check_list.append(np.max(np.abs(r_after - r_before)))

    print(np.max(check_list))

    return

sendResHome(resObjs, df)

# ======================================================================================================= 
# VISUAL CHECKING
# ======================================================================================================= 

#"""
#for i in range(15, 19):
#    print('Do fit results exist for the first object? ', resObjs[i].hasFit)
#
#    fig, (ax1, ax2) = plt.subplots(2, 1)
#    ax1.plot(resObjs[i].freq, np.sqrt(resObjs[i].I**2 + resObjs[i].Q**2))
#    ax2.plot(resObjs[i].I, resObjs[i].Q, '.')
#    ax2.axis('equal')

fitresObjs = []
fitresObjs_circles = []

for i in range(13,18):
    resObjs[i].load_params(scr.cmplxIQ_params)
    resObjs[i].do_lmfit(scr.cmplxIQ_fit, qc=5000)

    fitresObjs.append(resObjs[i])
    fitresObjs_circles.append(fg.fit_circle(resObjs[i].I, resObjs[i].Q))

print(fitresObjs_circles)

# check if the original centre and resonance have been recovered correctly
for res in fitresObjs:
    print(lf.compute_loop_reference_points(res))
#"""

# Check to see if fit parameters for resonators are obtained
resObj2 = fitresObjs[-3]
print(resObj2.__dict__.keys())
#pp.pprint(resObj2.__dict__)
print('Guess = ', resObj2.fmin, ' Hz')
print('Best fit = ', resObj2.lmfit_result['default']['result'].params['f0'].value, ' Hz')

fig = scr.plotResListData(
    [resObj2],
    plot_types=['IQ']
)
#plt.show()


figA = scr.plotResListData([fitresObjs[-3]],
                          plot_types = ['LogMag', 'Phase', 'IQ'], #Make two plots
                          num_cols = 3, #Number of columns
                          fig_size = 4, #Size in inches of each subplot
                          show_colorbar = False, #Don't need a colorbar with just one trace
                          force_square = True, #If you love square plots, this is for you!
                          plot_fits = [True]*3) #Overlay the best fit, need to specify for each of the plot_types
#"""

"""
#Check to see whether a results object exists
for i in range(len(fitresObjs)):
    print('Do fit results exist for the first object? ', fitresObjs[i].hasFit)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(fitresObjs[i].freq, np.sqrt(fitresObjs[i].I**2 + fitresObjs[i].Q**2))

    ax2.plot(fitresObjs[i].I, fitresObjs[i].Q, '.')

    t = np.linspace(0, 2*np.pi, 100)
    xc, yc, r = fitresObjs_circles[i]
    x, y = fg.circle(t, xc, yc, r)

    ax2.plot(x, y)


    ax2.axvline(0, color='k')
    ax2.axhline(0, color='k')
    ax2.axis('equal')
"""
plt.show()