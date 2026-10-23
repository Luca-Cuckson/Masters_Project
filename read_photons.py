#%%
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from glob import glob

def load_photon_h5(fileglob: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Load photon counts from a set of HDF5 files matching the given glob pattern.

    Parameters
    ----------
    fileglob : str
        A glob pattern to match HDF5 files containing photon count data.

    Returns
    -------
    resonator_ids : np.ndarray
        An array of resonator IDs corresponding to the photon counts.
    arrival_samples : np.ndarray
        An array of arrival sample times corresponding to the photon counts.
    '''
    files = sorted(glob(fileglob))  # Assume lexicographical order (000, 001, 002, ...)
    rids = []
    arrivals = []
    baselines = []
    peaks = []

    for filename in files:
        with h5.File(filename, "r") as f:
            print(f"Loading {filename}...")

            spf = f["metadata"].attrs["samples_per_frame"]
            data = f["plist"][:]

            # Casting to int16 and intt32 to minimise memory usage
            # 16-bit ID 0-> 65535
            # 32-bit arrival time 0-> 4,294,967,295 (> 1 hour)
            res_ids = data["res_id"].astype(np.int16)
            arrival_sample = ((spf * data["frame_id"]) + data["peak_offset"]).astype(np.int32)
            
            rids.append(res_ids)
            arrivals.append(arrival_sample)
            baselines.append(data["base_val"])
            peaks.append(data["peak_val"])

    return np.concatenate(rids), np.concatenate(arrivals), np.concatenate(baselines), np.concatenate(peaks)

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

#%%
if __name__ == "__main__":
    
    # Loads photon data from a number of HDF5 files matching the given glob pattern.

    data_dir = "D:/LucaC_L4_Project/first_beginnings"
    resonator_ids, arrival_times, baselines, peaks = load_photon_h5(f"{data_dir}/red/*_red_*.h5")

    # Select peaks by resonator ID
    nres=1024
    for resonator_id in range(nres):
#    for resonator_id in [1023]:  # Example: only process resonator ID 100

        ps = peaks[resonator_ids == resonator_id]
        ps = np.abs(np.rad2deg(ps))
        """
        backs = baselines[resonator_ids == resonator_id]
        backs = np.abs(np.rad2deg(backs))

        ps += backs  # Add baseline to peak values
        ps -= np.mean(backs)  # Subtract mean baseline to center the data
        
        plt.scatter(backs, ps+backs, s=1, alpha=0.5)
        plt.xlabel("Baseline (degrees)")
        plt.ylabel("Peak value (degrees)")
        plt.title(f"Baseline vs Peak Value for Resonator ID {resonator_id}")
        plt.show()
        """

        print(f"Resonator ID: {resonator_id}, Number of peaks: {len(ps)}")

        if len(ps) < 10:  # Not enough data to fit a Gaussian
            print(f"Resonator ID: {resonator_id}, Not enough data for fitting.")
            mean, stddev, fwhm, energy_res = np.nan, np.nan, np.nan, np.nan
        else:
            counts, bin_edges = np.histogram(ps, bins=200)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            try:
                popt, _ = curve_fit(
                    gaussian,
                    bin_centers,
                    counts,
                    p0=[counts.max(), bin_centers[np.argmax(counts)], np.std(ps)],
                )
                amplitude, mean, stddev = popt
                fwhm = 2.355 * stddev
                print(
                    f"Gaussian mean: {mean:.3f}, stddev: {stddev:.3f}, "
                    f"FWHM: {fwhm:.3f}, "
                )
                energy_res = mean / fwhm
                print(
                    f"Energy resolution: {energy_res:.3f}"
                )
                plt.plot(bin_centers, gaussian(bin_centers, *popt), color="red", linewidth=2, label="Gaussian fit")
                plt.legend()
            except (RuntimeError, ValueError) as exc:
                print(f"Resonator ID: {resonator_id}, Gaussian fit failed: {exc}")
                mean, stddev, fwhm, energy_res = np.nan, np.nan, np.nan, np.nan

        # append resonator ID and energy resolution to a single text file
        with open(f"{data_dir}/energy_resolution.txt", "a") as f:
            f.write(f"{resonator_id}\t{energy_res:.3f}\n")

        # append resonator ID and mean, fwhm to a single text file
        with open(f"{data_dir}/mean_fwhm.txt", "a") as f:
            f.write(f"{resonator_id}\t{mean:.3f}\t{fwhm:.3f}\n")

        plot = 'N'
        # skip plot?
        if plot == 'Y':
            plt.hist(ps, bins=200)
            plt.xlabel("Peak value (degrees)")
            plt.ylabel("Counts")
            plt.title(f"Histogram of Peak Values for Resonator ID {resonator_id}")
            plt.show()


# %%
