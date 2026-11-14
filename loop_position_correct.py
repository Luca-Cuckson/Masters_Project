#%%

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
import hanger_resonator_ccw as hrc


#%% ======================================================================================================= 
# LOAD IN DATA
# ======================================================================================================= 

df = pd.read_csv("D:/LucaC_L4_Project/11-11-26_Data/Initial_Powers/LT402#4_2026_09_12.csv")
dataFile = "D:/LucaC_L4_Project/11-11-26_Data/Initial_Powers/sweep_20261111_094457.h5"


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

def sendHome(Resonator_list, sweep):

    thetas = sweep['Rotation (degrees)'] * np.pi / 180 
    I_centres = sweep['I Centre']
    Q_centres = sweep['Q Centre']
    powers = sweep['Powers (dB)']

    resonators = []

    for i in range(len(Resonator_list)):

        #print(Resonator_list[i].freq[0])
        resname = Resonator_list[i].name
        freq = Resonator_list[i].freq

        I_old = Resonator_list[i].I.copy()
        Q_old = Resonator_list[i].Q.copy()
        
        # use rotation matrix to send IQ points home
        I_new = I_old*np.cos(thetas[i]) - Q_old*np.sin(thetas[i])
        Q_new = I_old*np.sin(thetas[i]) + Q_old*np.cos(thetas[i])

        # shift centres!
        I_new += I_centres[i]
        Q_new += Q_centres[i]

        resonators.append(
            scr.makeResFromData({
            'name': resname,
            'temp': 0.1,
            'pwr': powers[i],
            'freq': freq,
            'I': I_new,
            'Q': Q_new}))
        
        """
        # time to check if I've thrown loops out of whack
        r_before = np.sqrt(I_old**2 + Q_old**2)
        r_after  = np.sqrt((Resonator_list[i].I - I_centres[i])**2 + (Resonator_list[i].Q - Q_centres[i])**2)
        check_list.append(np.max(np.abs(r_after - r_before)))
        """ # Turns out this function wasn't testing too much

    return resonators



ResObjs = load_resonators_from_sweep(dataFile)
resObjs = sendHome(ResObjs, df)


# A check to see what params are created with resonator object - verify if S21 created at this stage
#print(resObjs[15].__dict__.keys())
"""
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

    print(np.max(check_list))  # Returns largest shift in 

    return

sendResHome(resObjs, df)

print(resObjs[15].__dict__.keys())  # Check to see if uncorrected parameters present
"""
#%% =======================================================================================================
# FIT RESONATORS
# ======================================================================================================= 

# Obtain fits for resonator objects
fitresObjs = []
res_offset = 15  # Offset to pick 1st resonator read off
res_count = 6  # Number of resonators to check

for i in range(res_count):  # Using group github code counter-clockwise versions of fitting parameters (since our loops are opposite sweep direction)
    i += res_offset
    resObjs[i].load_params(hrc.hanger_ccw_params)
    resObjs[i].do_lmfit(hrc.hanger_ccw_fit)

    fitresObjs.append(resObjs[i])

#%% ======================================================================================================= 
# VISUAL CHECKING
# ======================================================================================================= 

"""  # Old plotting to check that resonators had been correctly moved to original positions
for i in range(15, 19):
    print('Do fit results exist for the first object? ', resObjs[i].hasFit)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(resObjs[i].freq, np.sqrt(resObjs[i].I**2 + resObjs[i].Q**2))
    ax2.plot(resObjs[i].I, resObjs[i].Q, '.')
    ax2.axis('equal')
"""

n = 5   # check a single resonator

# Check to see if fit parameters for resonators are obtained
resObj2 = fitresObjs[n]  # Picking a specific resonator to observe
print(resObj2.__dict__.keys())  # Check all resonator fit quantities obtained
print(resObj2.lmfit_result['default']['result'].params.keys())
print('Guess = ', resObj2.fmin, ' Hz')  # Initial guess of resonant frequency
print('Best fit = ', resObj2.lmfit_result['default']['result'].params['f0'].value, ' Hz') # If significant difference observed, plot data to see why


# Plot of IQ loop alone
fig = scr.plotResListData(
    [resObj2],
    plot_types=['IQ'],
    plot_fits = [False]
)
#plt.show()


# Plotting multiple resonator fits and data
figA = scr.plotResListData([resObj2],
                          plot_types = ['LogMag', 'Phase', 'IQ'], #Make two plots
                          num_cols = 3, #Number of columns
                          fig_size = 4, #Size in inches of each subplot
                          show_colorbar = False, #Don't need a colorbar with just one trace
                          force_square = True, #If you love square plots, this is for you!
                          plot_fits = [True, True, True]) #Overlay the best fit, need to specify for each of the plot_types
#"""

"""
for i in range(5):  # Self-plotting to check resonator loops
    offset = n
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(fitresObjs[i+offset].freq, np.sqrt(fitresObjs[i+offset].I**2 + fitresObjs[i+offset].Q**2))  # magnitude of S21?
    ax2.plot(fitresObjs[i+offset].I, fitresObjs[i+offset].Q, '.')    # Check if IQ loop poitning towards origin

    ax2.axvline(0, color='k')
    ax2.axhline(0, color='k')
    ax2.axis('equal')
"""

plt.show()

#%% ======================================================================================================= 
# MULTI-RESONATOR INFO
# ======================================================================================================= 

f0s = []
qcs = []
qis = []
for i in range(len(fitresObjs)):
    f0s.append(fitresObjs[i].lmfit_result['default']['result'].params['f0'].value)
    qcs.append(fitresObjs[i].lmfit_result['default']['result'].params['qc'].value)
    qis.append(fitresObjs[i].lmfit_result['default']['result'].params['qi'].value)

print("Resonant frequencies =", f0s)
print("Coupling quality factors =", qcs)
print("qis =", qis)

resNum = []
for i in range(res_count):
    resNum.append(i + res_offset)

fig, (ax1,ax2,ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(resNum, f0s)
ax2.plot(resNum, qcs)
ax2.set_ylim(0, 100000)
ax3.plot(resNum, qis)
ax3.set_ylim(0, 2 * 1e6)

# Quickly get some filtered data to look at majority of resonators
# some outliers are throwing the histogram fit way off
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
# %%
