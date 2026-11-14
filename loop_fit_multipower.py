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

sweepcsv_init = pd.read_csv("D:/LucaC_L4_Project/11-11-26_Data/Initial_Powers/LT402#4_2026_09_12.csv")
sweeph5_init = "D:/LucaC_L4_Project/11-11-26_Data/Initial_Powers/sweep_20261111_094457.h5"
sweepcsv_plus3 = pd.read_csv(r"D:\LucaC_L4_Project\11-11-26_Data\Plus3db\LT402#4_2026_09_12.csv")
sweeph5_plus3 = "D:/LucaC_L4_Project/11-11-26_Data/Plus3db/sweep_20261111_150929.h5"
sweepcsv_minus3 = pd.read_csv(r"D:\LucaC_L4_Project\11-11-26_Data\Minus3dB\LT402#4_2026_09_12.csv")
sweeph5_minus3 = r"D:\LucaC_L4_Project\11-11-26_Data\Minus3dB\sweep_20261111_162737.h5"


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



ResObjs_i = load_resonators_from_sweep(sweeph5_init)
resObjs_i = sendHome(ResObjs_i, sweepcsv_init)
ResObjs_p3 = load_resonators_from_sweep(sweeph5_plus3)
resObjs_p3 = sendHome(ResObjs_p3, sweepcsv_plus3)
ResObjs_m3 = load_resonators_from_sweep(sweeph5_minus3)
resObjs_m3 = sendHome(ResObjs_m3, sweepcsv_minus3)


#%% =======================================================================================================
# FIT RESONATORS
# ======================================================================================================= 

res_offset = 15  # Offset to pick 1st resonator read off
res_count = 6  # Number of resonators to check

# Obtain fits for resonator objects
fitresObjs_i = []
fitresObjs_p3 = []
fitresObjs_m3 = []


for i in range(res_count):  # Using group github code counter-clockwise versions of fitting parameters (since our loops are opposite sweep direction)
    i += res_offset
    resObjs_i[i].load_params(hrc.hanger_ccw_params)
    resObjs_i[i].do_lmfit(hrc.hanger_ccw_fit)

    resObjs_p3[i].load_params(hrc.hanger_ccw_params)
    resObjs_p3[i].do_lmfit(hrc.hanger_ccw_fit)

    resObjs_m3[i].load_params(hrc.hanger_ccw_params)
    resObjs_m3[i].do_lmfit(hrc.hanger_ccw_fit)

    fitresObjs_i.append(resObjs_i[i])
    fitresObjs_p3.append(resObjs_p3[i])
    fitresObjs_m3.append(resObjs_m3[i])

# Combine resonator lists into one master list for plotting
resList = fitresObjs_i.append(fitresObjs_p3.append(fitresObjs_m3))

#%% ======================================================================================================= 
# VISUAL CHECKING
# ======================================================================================================= 

# Create a little list of resonator objects for same resonator at different powers
res_choice = 3
sameRes_diffPwr = [fitresObjs_i[res_choice], fitresObjs_p3[res_choice], fitresObjs_m3[res_choice]]

fig = scr.plotResListData(
    [fitresObjs_i[res_choice]],
    plot_types=['Phase'],
    plot_fits = [False]
)

figA = scr.plotResListData(sameRes_diffPwr,
                            plot_types=['IQ', #Real vs Imaginary part of S21
                                        'rIQ', #residual of fits in IQ plane
                                        'LogMag', #Magnitude of S21 vs frequency
                                        'Phase'], #Phase of S21 vs frequency
                            show_colorbar = False,
                            fig_size=4,
                            num_cols = 2,
                            force_square = True,
                            plot_fits = [False]*4) #<-- need to specify a boolean for each plot type

plt.show()

""" Can't do this until after made a resonator sweep! They don't call this function until later
scr.plotResSweepParamsVsPwr(sameRes_diffPwr)

plt.show()
"""
# %%
