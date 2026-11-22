#%%

import matplotlib.pyplot as plt

import numpy as np
import scraps as scr
import pprint as pp
import h5py as h5
import os
import re
import pandas as pd

import functions.looping_funcs as lf
import functions.fitting as fg
import functions.hanger_resonator_ccw as hrc


#%% ======================================================================================================= 
# LOAD IN DATA
# =========================================================================================================

resTemp = 0.02   #-- Resonator operating temperature in Kelvin

#sweepcsv = pd.read_csv(r"D:\LucaC_L4_Project\11-11-26_Data\Plus3db\LT402#4_2026_09_12.csv")
#sweeph5 = "D:/LucaC_L4_Project/11-11-26_Data/Plus3db/sweep_20261111_150929.h5"

#sweepcsv = pd.read_csv(r"D:\LucaC_L4_Project\11-11-26_Data\Initial_Powers\LT402#4_2026_09_12.csv")
#sweeph5 = r"D:\LucaC_L4_Project\11-11-26_Data\Initial_Powers\sweep_20261111_094457.h5"

sweepcsv = pd.read_csv(r"D:\LucaC_L4_Project\11-11-26_Data\Minus3dB\LT402#4_2026_09_12.csv")
sweeph5 = r"D:\LucaC_L4_Project\11-11-26_Data\Minus3dB\sweep_20261111_162737.h5"

def load_resonators_from_sweep(h5_path: str) -> list:
    """Legacy - Loads resonators from a widesweep H5 file and loads each into a scraps
    resonator object.

    Parameters
    ----------
    h5_path : str
        Filepath of the widesweep h5 file.

    Returns
    -------
    list of scr.Resonator
        List of loaded Resonator objects.

    Raises
    ------
    FileNotFoundError
        Raised when the HDF5 file does not exist at the given filepath.
    KeyError
        Raised when the expected keys within the HDF5 file do not exist, suggesting incorrect data format.
    """

    #powers = sweep['Powers (dB)'] # Extract powers from csv to create resonator

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5.File(h5_path, 'r') as f:
        if 'Resonators' not in f:
            raise KeyError("HDF5 file does not contain required 'Resonators' dataset")
        
        data = f['Resonators']
        resonators = []
        
        for (resname, r) in data.items():
            if not 'IQ Data' in r:
                print("Skipped resonator due to missing IQ Data")
                continue

            if not 'Frequencies sampled' in r:
                print("Skipped resonator due to missing Frequencies sampled")
                continue

            IQ = r['IQ Data'][:]
            freq = r['Frequencies sampled'][:]

            #print(IQ) # was checking shape of IQ (it is ...+...j complex number)
            #print(resname) # Was looking at initial order - Done by first digit of res..., then ordered at end of this function

            resonators.append(
                scr.makeResFromData({
                'name': resname,
                'temp': 0,
                'pwr': 0,
                'freq': freq,
                'I': np.real(IQ),
                'Q': np.imag(IQ)}))
            
    # Sort resonators by name using natural sort (so 'Res10' comes after 'Res2')
    resonators.sort(key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s.name)])

    return resonators

def addPower(Resonator_list, sweep):  #-- Create new resonator objectd with correct power and temperature
    #-- Done this way due to difficulty of correctly matching powers to resIDs 
    #-- given resonator ordering in sweep h5 file
    #-- Checks later indicate precisely zero shift in resonator IQ positions

    powers = sweep['Powers (dB)']

    resonators = []

    for i in range(len(Resonator_list)):

        #print(Resonator_list[i].freq[0])
        resname = Resonator_list[i].name
        freq = Resonator_list[i].freq

        resonators.append(
            scr.makeResFromData({
            'name': resname,
            'temp': resTemp,
            'pwr': powers[i],
            'freq': freq,
            'I': Resonator_list[i].I,
            'Q': Resonator_list[i].Q}))

    return resonators


ResObjs = load_resonators_from_sweep(sweeph5)
resObjs = addPower(ResObjs, sweepcsv)

#%% =======================================================================================================
# CHECK LOOPS UNMOVED
# =========================================================================================================
#-- I have recreated the resonator list from the old one, 
#-- so want to check I haven't introduced error in loop positions

checkNum = 16
'''
plt.figure(0)
plt.plot(ResObjs[checkNum].I, ResObjs[checkNum].Q)
plt.plot(resObjs[checkNum].I, resObjs[checkNum].Q)
'''

mag_diffs = []
for i in range(len(resObjs)):
    Q_diff = ResObjs[i].Q - resObjs[i].Q
    I_diff = ResObjs[i].I - resObjs[i].I
    mag_diff = np.sqrt(Q_diff ** 2 + I_diff ** 2)
    mag_diffs.append(np.max(mag_diff))

    #print(i, 'mag_diff =', mag_diff, ', Q_diff =', Q_diff, ', I_diff =', I_diff)

print('Max point shift magnitude =', np.max(mag_diffs))


#%% =======================================================================================================
# FIT RESONATORS
# =========================================================================================================

#-- Obtain fits for resonator objects
fitresObjs = []
res_offset = 15  #-- Offset to pick 1st resonator read off
res_count = 15  #-- Number of resonators to check

for i in range(res_count):  #-- Using group github code counter-clockwise versions of fitting parameters (since our loops are opposite sweep direction)
    i += res_offset
    resObjs[i].load_params(hrc.hanger_ccw_params)
    resObjs[i].do_lmfit(hrc.hanger_ccw_fit)

    fitresObjs.append(resObjs[i])

#%% ======================================================================================================= 
# VISUAL CHECKING
# =========================================================================================================

n = 11  #-- check a single resonator

#-- Check to see if fit parameters for resonators are obtained
resObj2 = fitresObjs[n]  #-- Picking a specific resonator to observe
print('Picked resonator is', resObj2.name)


#print(resObj2.__dict__.keys())  #-- Check all resonator fit quantities obtained
#print(resObj2.lmfit_result['default']['result'].params.keys())
#print(resObj2.lmfit_result['default']['result'])


print('Guess = ', resObj2.fmin, ' Hz')  #-- Initial guess of resonant frequency
print('Best fit = ', resObj2.lmfit_result['default']['result'].params['f0'].value, ' Hz') # If significant difference observed, plot data to see why
print('Reduced chi-squared = ', resObj2.lmfit_result['default']['result'].redchi)

#-- Just to check reading out sweep csv correctly (could comment out)
print('Centre = ', sweepcsv['I Centre'] + 1j * sweepcsv['Q Centre'], 'Rotation =', sweepcsv['Rotation (degrees)'])

#-- Plot of IQ loop alone
fig = scr.plotResListData(
    [resObj2],
    plot_types=['IQ'],
    plot_fits = [False]
)
#plt.show()


#-- Plotting multiple resonator fits and data
figA = scr.plotResListData([resObj2],
                          plot_types = ['LogMag', 'Phase', 'IQ', 'rIQ'], #-- Make two plots
                          num_cols = 2, #-- Number of columns
                          fig_size = 4, #-- Size in inches of each subplot
                          show_colorbar = False, #-- Don't need a colorbar with just one trace
                          force_square = True, #-- If you love square plots, this is for you!
                          plot_fits = [True, True, True, True]) #-- Overlay the best fit, need to specify for each of the plot_types
#"""

#%% ======================================================================================================= 
# MULTI-RESONATOR INFO
# =========================================================================================================
#-- TODO: Needs a big tidy-up


print(fitresObjs[0].name)  #-- Check which resonator starting with (for comparison with csv info)


#-- Want lists for multiple resonator objects of:
f0s = []   #-- Resonant frequency
qcs = []   #-- Coupling quality factor
qis = []   #-- Internal quality factor
redChis = []   #-- Redcued chi-squared of lmfit to loop data
for i in range(len(fitresObjs)):
    f0s.append(fitresObjs[i].lmfit_result['default']['result'].params['f0'].value)
    qcs.append(fitresObjs[i].lmfit_result['default']['result'].params['qc'].value)
    qis.append(fitresObjs[i].lmfit_result['default']['result'].params['qi'].value)
    redChis.append(fitresObjs[i].lmfit_result['default']['result'].redchi)

    if fitresObjs[i].lmfit_result['default']['result'].params['qc'].value > 1e7:
        print(fitresObjs[i].name)

#-- Find the total coupling factor 
Qs = np.sqrt((np.array(qis)) ** 2 + (np.array(qcs)) ** 2)



resNum = []
for i in range(res_count):
    resNum.append(i + res_offset)


#-- Plot how fit parameters vary with resonator
fig, (ax1,ax2,ax3,ax4, ax5) = plt.subplots(5, 1, sharex=True)
ax1.plot(resNum, f0s)
ax2.plot(resNum, qcs)
#ax2.set_ylim(0, 800000)
ax3.plot(resNum, qis)
#ax3.set_ylim(0, 1e6)
ax4.plot(resNum, redChis)
#ax4.set_ylim(0, 500)
ax5.plot(resNum, Qs)
#ax5.set_ylim(0, 10 * 1e5)

#-- Quickly get some filtered data to look at majority of resonators
#-- some outliers are throwing the histogram fit way off
threshold_qc = 110000
filtered_qc = list(filter(lambda x: 20000 < x < threshold_qc, qcs))
threshold_qi = 2 * 1e6
filtered_qi = list(filter(lambda x: x < threshold_qi, qis))

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.hist(filtered_qc, 20)
#ax1.set_xlim(20000, 110000)
ax2.hist(filtered_qi, 20)
#ax2.set_xlim(0, 2 * 1e6)

plt.show()

#-- Plot both IQ loops and fit residuals across multiple resonators
for res in fitresObjs[0:5]:
    figA = scr.plotResListData([res],
                          plot_types = ['LogMag', 'Phase', 'IQ', 'rIQ'], #-- Make two plots
                          num_cols = 2, #-- Number of columns
                          fig_size = 4, #-- Size in inches of each subplot
                          show_colorbar = False, #-- Don't need a colorbar with just one trace
                          force_square = True, #-- If you love square plots, this is for you!
                          plot_fits = [True, True, True, True]) #-- Overlay the best fit, need to specify for each of the plot_types

    plt.show()
    print('Above resonator is', res.name)


#%% ======================================================================================================= 
# SEND RESONATORS TO ORIGIN
# =========================================================================================================

"""I might need to look back and double check quality of movement to origin.
   Errors in new positions/rotations likely due to poor fitting to bifurcated
   or non-circular IQ loops when overdriving"""
def sendToOrigin(Resonator_list, sweep): #-- Use fit centres & rotations from csv file to send to origin

    thetas = sweep['Rotation (degrees)'] * np.pi / 180 
    I_centres = sweep['I Centre']
    Q_centres = sweep['Q Centre']

    resonators = []

    for i in range(len(Resonator_list)):

        #-- Keep these resonator object parameters unchanged 
        resname = Resonator_list[i].name
        freq = Resonator_list[i].freq
        power = Resonator_list[i].pwr

        #-- Shift centres!
        I_centred = Resonator_list[i].I - I_centres[i]
        Q_centred = Resonator_list[i].Q - Q_centres[i]

        #-- Use rotation matrix to send IQ points home
        #-- -theta used as want to set rotation to zero point (i.e. -Q axis)
        I_new = I_centred*np.cos(-thetas[i]) - Q_centred*np.sin(-thetas[i])
        Q_new = I_centred*np.sin(-thetas[i]) + Q_centred*np.cos(-thetas[i])

        resonators.append(
            scr.makeResFromData({
            'name': resname,
            'temp': resTemp,
            'pwr': power,
            'freq': freq,
            'I': I_new,
            'Q': Q_new}))

    return resonators

originResObjs = sendToOrigin(resObjs, sweepcsv)

#%% ======================================================================================================= 
# CHECK RESONATORS AT ORIGIN AND ROTATED
# =========================================================================================================

checkRange = 10  #-- How many plots checking before satisfied something's going right
checkOffset = 15 #-- Ignore first cluster of switched-off resonators (Power = -200dB)
"""
for i in range(checkOffset, checkOffset + checkRange):
    I, Q = originResObjs[i].I, originResObjs[i].Q
    plt.figure()
    plt.plot(I, Q)
    plt.axvline(0, color='k')
    plt.axhline(0, color='k')

plt.show()
"""

#%% ======================================================================================================= 
# S_21 GRADIENT
# =========================================================================================================

grad_checkRange = 5
grad_checkOffset = 15

#-- Obtain |S_21| for each resonator myself (since scraps won't fit around origin)
S21_Mags = [] 
for res in originResObjs: #-- Output nested list of S_21 magnitudes across resonators
    Is, Qs = res.I, res.Q
    MagS_21 = np.sqrt(Is ** 2 + Qs ** 2)
    S21_Mags.append(MagS_21)

"""
#-- Visual checks
for i in range(grad_checkOffset, grad_checkOffset + grad_checkRange):
    freq, mag = originResObjs[i].freq, S21_Mags[i]
    plt.figure()
    plt.plot(freq, mag)
"""


#-- Obtain gradient of |S_21| for each resonator
S21_grads = []
midFreqs = []   #-- Want nested list of frequency sample midpoints
for i in range(0, len(originResObjs)):
    #-- Get frequency between original |S_21| points
    frequency = originResObjs[i].freq
    mid_freq = .5 * (frequency[1:] + frequency[:-1])
    midFreqs.append(mid_freq)

    #-- Get difference between neighbouring |S_21| values
    S21s = S21_Mags[i]
    grads = S21s[1:] - S21s[:-1]
    S21_grads.append(grads)

    """ Good news! Was getting 0 for my difference, so calculated correctly.
    #-- Run for now as check to see that quick grad calculation is correct
    check_gradding = []
    for i in range(1, len(S21s)):
        gradi = S21s[i] - S21s[i-1]
        check = gradi - grads[i-1]
        check_gradding.append(check)

    if np.max(check_gradding) > 0:
        print(np.max(check_gradding))
    """
    
"""
#-- Visual checks
for i in range(grad_checkOffset, grad_checkOffset + grad_checkRange):
    freq, grad = midFreqs[i], S21_grads[i]
    plt.figure()
    plt.plot(freq, grad)
"""





#-- Visual checks of all 3 bits together
for i in range(grad_checkOffset, grad_checkOffset + grad_checkRange):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.set_box_aspect(1)
    ax1.plot(originResObjs[i].I, originResObjs[i].Q)
    ax1.axvline(0, color='k')
    ax1.axhline(0, color='k')

    ax2.set_box_aspect(1)
    ax2.plot(originResObjs[i].freq, S21_Mags[i])

    ax3.set_box_aspect(1)
    ax3.plot(midFreqs[i], S21_grads[i])

plt.show()
 # %%
