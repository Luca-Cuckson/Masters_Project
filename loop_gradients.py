#%%
import h5py as h5 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scraps as scr
import os
import re


#%% ======================================================================================================= 
# LOAD IN DATA
# =========================================================================================================

#sweep_csv = pd.read_csv(r"D:\LucaC_L4_Project\11-11-26_Data\Plus3db\LT402#4_2026_09_12.csv")
#sweep_h5 = r"D:\LucaC_L4_Project\11-11-26_Data\Plus3db\sweep_20261111_150929.h5"
sweepcsv_init = pd.read_csv("D:/LucaC_L4_Project/11-11-26_Data/Initial_Powers/LT402#4_2026_09_12.csv")
sweeph5_init = "D:/LucaC_L4_Project/11-11-26_Data/Initial_Powers/sweep_20261111_094457.h5"


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

resObjs = load_resonators_from_sweep(sweeph5_init)

#%% ======================================================================================================= 
# DATA VISUALISATION
# =========================================================================================================

# Check across multiple resonators

for i in range(0, 30):
    plt.figure(1)    # Plot the IQ loop to see visual properties
    plt.plot(resObjs[i].I, resObjs[i].Q)
    plt.axvline(0, color = 'k')
    plt.axhline(0, color = 'k')

    plt.show()






# Working with a single resonator =======================

choice = 15
chosen_res = resObjs[choice]

MagS_21 = np.sqrt(chosen_res.I ** 2 + chosen_res.Q ** 2)

"""IMPORTANT: this really needs fixing, this is just a proof of concept!!!"""
S21_gradient = []    # Now find the "velocity gradient" of S_21
for i in range(len(chosen_res.freq) - 1):
    S21_gradient.append(MagS_21[i+1] - MagS_21[i])

S21_gradient.append(S21_gradient[-1])   # IMPORTANT: This is a temporary trick to get array same size as frequency array

# Actually making plots ====================================
"""
plt.figure(1)    # Plot the IQ loop to see visual properties
plt.plot(chosen_res.I, chosen_res.Q)
plt.axvline(0, color = 'k')
plt.axhline(0, color = 'k')

plt.figure(2)    # Plot |S_21| so see how varies with frequency
plt.plot(chosen_res.freq, MagS_21)

plt.figure(3)
plt.plot(chosen_res.freq, S21_gradient)
"""

plt.show()
# %%
