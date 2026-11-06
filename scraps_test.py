import matplotlib.pyplot as plt

import numpy as np
import scraps as scr
import pprint as pp
import h5py as h5
import os
import re

import looping_funcs as lf

def save_resonators_from_sweep(h5_path, freqs, IQs): 
    """
    """
    n = len(freqs)

    with h5.File(h5_path, 'w') as hf:
        resonator_info = hf.create_group('Resonators')
        
        for i in range(n):
            # Resonator name is just incremental
            sweep_info = resonator_info.create_group('Res%i'%(i+1))
            
            # Save sweep data for given resonator
            sweep_info.create_dataset('IQ Data', data=IQs[i])
            sweep_info.create_dataset('Frequencies sampled', data=freqs[i])

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

            resonators.append(
                scr.makeResFromData({
                'name': resname,
                'temp': 0,
                'pwr': 0,
                'freq': freq,
                'I': np.real(IQ),
                'Q': np.imag(IQ)}))

            print(IQ.dtype)
            print(IQ.shape)
            
    # Sort resonators by name using natural sort (so 'Res10' comes after 'Res2')
    resonators.sort(key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s.name)])

    return resonators


#Load in a file
dataFile = "D:/LucaC_L4_Project/Luca E2E/Luca E2E/sweep_20261022_132646.h5"

resObj1 = load_resonators_from_sweep(dataFile)[36]
resObj2 = load_resonators_from_sweep(dataFile)[37]

print(resObj1)

print(resObj1.freq.shape)
print(resObj1.I.shape)
print(resObj1.Q.shape)


#Check the temperature and power
print('Temperature = ', resObj1.temp)
print('Power = ', resObj1.pwr)

#Check to see whether a results object exists
print('Do fit results exist for the first object? ', resObj1.hasFit)
print('Do fit results exist for the second object? ', resObj2.hasFit)


#Explicitly call the fitter on the first object.
#Here we will call it, and also override the guess for coupling Q with our own quess
resObj1.load_params(scr.cmplxIQ_params)
resObj1.do_lmfit(scr.cmplxIQ_fit, qc=5000)

resObj2.load_params(scr.cmplxIQ_params)
resObj2.do_lmfit(scr.cmplxIQ_fit, qc=5000)

#Check to see whether a results object exists again, now they are both True
print('Do fit results exist for the first object? ', resObj1.hasFit)
print('Do fit results exist for the second object? ', resObj2.hasFit)


centre1, base1, resonance1 = lf.compute_loop_reference_points(resObj1)
print(centre1, base1, resonance1)
centre2, base2, resonance2 = lf.compute_loop_reference_points(resObj2)
print(centre2, base2, resonance2)

#print(lf.compute_loop_rotation(resObj1, resonance1))
#print(lf.compute_loop_rotation(resObj2, resonance2))

#'''
#Compare the best guess for the resonant frequency (minimum of the curve) to the actual fit
#Because we didn't specify a label for our fit, the results are stored in the lmfit_result
#dict under the 'default' key. If we passed the optional label argument to the do_lmfit
#method, it would store the results under whatever string is assigned to label.
print('Guess = ', resObj2.fmin, ' Hz')
print('Best fit = ', resObj2.lmfit_result['default']['result'].params['f0'].value, ' Hz')
print('Best fit with different qc guess = ',
      resObj1.lmfit_result['default']['result'].params['f0'].value, ' Hz')


plt.figure()
plt.plot(resObj1.freq, np.sqrt(resObj1.I**2 + resObj1.Q**2))

plt.figure()
plt.plot(resObj1.I, resObj1.Q, '.')
plt.axis('equal')

plt.figure()
plt.plot(resObj2.I, resObj2.Q, '.')
plt.axis('equal')

plt.show()


#This function takes a list of resonators. It can handle a single one, just need to pass it as a list:
figA = scr.plotResListData([resObj1],
                          plot_types = ['LogMag', 'Phase'], #Make two plots
                          num_cols = 2, #Number of columns
                          fig_size = 4, #Size in inches of each subplot
                          show_colorbar = False, #Don't need a colorbar with just one trace
                          force_square = True, #If you love square plots, this is for you!
                          plot_fits = [True]*2) #Overlay the best fit, need to specify for each of the plot_types

figA = scr.plotResListData([resObj2],
                          plot_types = ['LogMag', 'Phase'], #Make two plots
                          num_cols = 2, #Number of columns
                          fig_size = 4, #Size in inches of each subplot
                          show_colorbar = False, #Don't need a colorbar with just one trace
                          force_square = True, #If you love square plots, this is for you!
                          plot_fits = [True]*2) #Overlay the best fit, need to specify for each of the plot_types

plt.show()
#'''