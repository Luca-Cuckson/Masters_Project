import numpy as np
import scraps as scr
import h5py as h5
import os
import re

from scipy.interpolate import interp1d

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
                'temp': 0.1,
                'pwr': -62,
                'freq': freq,
                'I': np.real(IQ),
                'Q': np.imag(IQ)}))
            
    # Sort resonators by name using natural sort (so 'Res10' comes after 'Res2')
    resonators.sort(key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s.name)])

    return resonators

def compute_loop_reference_points(resonator: scr.Resonator):
    """
    Computes initial reference points for a resonator loop: resonator IQ at f0,
    baseline IQ at f0, and the geometric loop center.
    """
    if not resonator.hasFit:
        raise ValueError("Provided resonator does not have a fitted model!")

    # Unpack fitted parameters for resonant frequency and amplitude gain
    f0 = resonator.lmfit_result['default']['result'].params['f0'].value
    g0 = resonator.lmfit_result['default']['result'].params['gain0'].value

    # Interpolate values to determine values at f0
    baseline_phase = interp1d(resonator.freq, resonator.phaseBaseline)(f0)
    rI = interp1d(resonator.freq, resonator.I)(f0)
    rQ = interp1d(resonator.freq, resonator.Q)(f0)

    # Determine baseline IQ values at f0
    bI = g0 * np.cos(baseline_phase)
    bQ = g0 * np.sin(baseline_phase)

    # Calculate the geometric loop center
    cI = 0.5 * (bI + rI)
    cQ = 0.5 * (bQ + rQ)

    # Package as complex values
    center = cI + 1j * cQ
    baseline = bI + 1j * bQ
    resonance = rI + 1j * rQ
    
    return center, baseline, resonance


def compute_loop_rotation(center: complex, resonance: complex):
    """
    """
    # Relative positive of the resonance point with respect to the center
    dI = resonance.real - center.real
    dQ = resonance.imag - center.imag

    # Calculate rotation angle relative to -Q axis
    theta = np.arctan2(dQ, dI) + (np.pi / 2)  
    
    # Wrap to (-pi, pi]
    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    
    # Convert to degrees
    return np.degrees(theta)


