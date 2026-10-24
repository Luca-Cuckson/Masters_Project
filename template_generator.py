#%%
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import h5py as h5

from scipy.signal import find_peaks

def load_phase16(filepath: str, n_res: int = 32) -> np.ndarray:
    '''
    Loads a .phase16 file and returns the phase data as a 2D numpy array.
    
    Parameters
    ----------
    filepath : str
        The path to the .phase16 file to be loaded.
    n_res : int
        The number of resonators in the .phase16 file. Default is 32.

    Returns
    -------
    np.ndarray
        A 2D numpy array containing the phase data in radians, with shape (n_res, n_samples).
    '''
    return np.fromfile(filepath, dtype=np.int16).reshape(n_res, -1) * (np.pi / 32768)

def load_template_h5(fn):
    '''
    Loads template data from an h5 file.

    Parameters
    ----------
    fn : str
        Path to the .h5 file.

    Returns
    -------
    templates : np.ndarray
        Array of shape (num_res, n_samples) containing the template data as float64.
    '''
    with h5.File(fn, 'r') as f:
        templates = f['data'][:]

    return templates

def save_template_h5(fn, templates):
    '''
    Saves template data to an h5 file.

    Parameters
    ----------
    fn : str
        Path to the .h5 file.
    templates : np.ndarray
        Array of shape (num_res, n_samples) containing the template data as float64.
    
    Returns
    -------
    None.
    '''
    if templates.ndim != 2:
        raise ValueError("templates must be 2D array of shape (num_res, n_samples)")
    
    if templates.dtype != np.float64:
        templates = templates.astype(np.float64)
    
    with h5.File(fn, 'w') as f:
        f.create_dataset('data', data=templates.astype(np.float64))

def save_thresholds_h5(filename: str, thresholds: np.typing.ArrayLike):
    '''
    '''
    thresholds = np.asarray(thresholds).astype(np.float64)

    if thresholds.ndim != 1:
        raise ValueError("Thresholds array must be 1D.")

    with h5.File(filename, 'w') as f:
        f.create_dataset('data', data=thresholds)

def pulse_model(ts: npt.ArrayLike, tau_d: float, tau_r: float, A: float = 1.0, t0: float = 0.0, B0: float = 0.0, B1: float = 0.0) -> npt.NDArray:
    """
    A standard MKID pulse model with exponential rise and decay, including baseline,
    amplitude and arrival time parameters.
   
    Parameters
    ----------
    ts : ArrayLike
        Time series for the pulse model.
    tau_d : float
        Decay time constant.
    tau_r : float
        Rise time constant.
    A : float, optional
        Pulse amplitude. Default is 1.0.
    t0 : float, optional
        Arrival time (pulse start time). Default is 0.0.
    B0 : float, optional
        Baseline offset. Default is 0.0.
    B1 : float, optional
        Linear baseline slope coefficient. Default is 0.0.
 
    Returns
    -------
    NDArray
        The pulse model evaluated at each time point in `ts`.
    """
    t = ts - t0
    y = np.zeros_like(t, dtype=float)
 
    m = t >= 0
    tp = t[m]
 
    # Difference-of-exponentials MKID pulse
    pulse = -A * (np.exp(-tp / tau_d) - np.exp(-tp / tau_r))
    y[m] = pulse
 
    # Linear baseline (mean-centered to reduce covariance)
    baseline = B0 + B1 * (ts - ts.mean())
 
    return y + baseline

def matched_filter(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    return np.correlate(signal, template, mode='same')[:-len(template)] # defines cross-correlation of two 1d arrays

def calculate_threshold_from_dark(dark: np.ndarray, template: np.ndarray, n_sigma: float = 6.0) -> float:
    '''
    '''
    matched_dark = matched_filter(dark, template)
    md_mean = np.mean(matched_dark)
    md_std = np.std(matched_dark)
    threshold = md_mean + (n_sigma * md_std)

    return threshold

###############################################################################################################################################
#%%
basedir = "D:/LucaC_L4_Project/Luca E2E/Luca E2E"

n_res = 1024                    # Total number of resonators in the dataset
n_res_block = 32                # Number of resonators per .phase16 file
template_len = 32               # Template length (number of samples)

photon_minimum = 300            # Minimum number of photon events required to create a template for a resonator. 

# Estimated template
default_template = pulse_model(np.arange(template_len), tau_d=5, tau_r=2, A=1.0)
default_template /= np.sqrt(np.sum(default_template**2))   #compound operand - modifies LHS in place
default_template = default_template[::-1]
template = default_template

###############################################################################################################################################
#%% View A single resonator's dark and photon data
k = 325
n = 32 * (k // 32)  # floor division. divides then rounds down to nearest integer
j = k % 32  #getting the remainder

print(k, n, j)

dark_filepath = f"{basedir}/dark_{n:04d}.phase16"
photon_filepath = f"{basedir}/photons_{n:04d}.phase16"

dark_phase = load_phase16(dark_filepath, n_res_block)
photon_phase = load_phase16(photon_filepath, n_res_block)

# Foreach resonator in the the block
dark = dark_phase[j]
photon = photon_phase[j]

plt.title(f"Resonator {k}")
plt.plot(np.rad2deg(photon), label='Photon', color='k')
plt.plot(np.rad2deg(dark), label='Dark', alpha=0.5, color='r')
plt.legend()
plt.xlim(0000, 20_000)
plt.show()
    
        
###############################################################################################################################################
#%%
thresholds = []
out_templates = []

# Resonators are saved in files in blocks
for i in range(0, n_res, n_res_block):
    dark_filepath = f"{basedir}/dark_{i:04d}.phase16"
    photon_filepath = f"{basedir}/photons_{i:04d}.phase16"

    dark_phase = load_phase16(dark_filepath, n_res_block)
    photon_phase = load_phase16(photon_filepath, n_res_block)

    # Foreach resonator in the the block
    for j in range(n_res_block):
        dark = dark_phase[j]
        photon = photon_phase[j]

        initial_threshold = calculate_threshold_from_dark(dark, template, n_sigma=6.0)

        matched_phase = matched_filter(photon, template)
        peaks, _ = find_peaks(matched_phase, height=initial_threshold, distance=template_len)

        if len(peaks) < photon_minimum:
            threshold = 1e6
            out_template = np.zeros(template_len)
        else:
            stamps = []

            # Produce a phase stamp for each detected peak, normalizing by the pre-peak baseline and the post-peak maximum
            for p in peaks:
                try:
                    stamp = np.copy(photon[p-template_len:p+template_len])
                    stamp -= np.mean(stamp[:template_len])
                    stamp /= np.max(np.abs(stamp[template_len:]))
                    stamps.append(stamp)
                except:
                    continue

            # New template is the average of the phase stamps
            out_template = np.mean(stamps, axis=0)

            # Normalize the new template
            out_template -= np.mean(out_template[:template_len])
            out_template = out_template[template_len:]
            out_template /= np.sqrt(np.sum(out_template**2))
            out_template = out_template[::-1]

            # Recalculate the threshold using the new template
            threshold = calculate_threshold_from_dark(dark, out_template, n_sigma=6.0)
            
        # Collect the threshold and template for this resonator
        thresholds.append(threshold)
        out_templates.append(out_template)
        print(f"Res{i+j}: Detected {len(peaks)} photon events. Threshold set to {threshold:.2f}.")

###############################################################################################################################################
#%%
np.sum(np.array(thresholds) < 1e6)

###############################################################################################################################################
#%%
for temp, thresh in zip(out_templates, thresholds):
    if thresh < 1e6:
        plt.plot(temp, alpha=0.8)

###############################################################################################################################################
#%% Plots each template for inspection
for i in range(0, len(thresholds), 8):
    for j in range(8):
        k = i + j
        thresh = thresholds[k]
        temp = out_templates[k]
        if thresh < 1e6:
            plt.plot(temp, label=f'Res{i + j}')
            plt.legend()
    plt.show()

###############################################################################################################################################
#%% Manually threshold out 'bad' templates
thresholds[41] = 1e6
thresholds[66] = 1e6
thresholds[109] = 1e6
thresholds[124] = 1e6
thresholds[333] = 1e6
thresholds[532] = 1e6
thresholds[563] = 1e6
thresholds[565] = 1e6
thresholds[1019] = 1e6
thresholds[1022] = 1e6
thresholds[1023] = 1e6

###############################################################################################################################################
#%% Save template and threshold data
print(np.sum(np.array(thresholds) < 1e6))

save_template_h5("generated_templates.h5", np.array(out_templates))
save_thresholds_h5("generated_thresholds.h5", np.array(thresholds))


# %%
