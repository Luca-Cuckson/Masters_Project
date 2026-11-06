from scraps import hanger_fit, hanger_params

import numpy as np
import scipy.signal as sps

def hanger_ccw_fit(paramsVec, res, residual=True, **kwargs):
    """Return complex S21 of a counter-clockwise hanger resonator model or, if data is specified, a residual.

    Parameters
    ----------
    params : list-like
        A an ``lmfit.Parameters`` object containing (df, f0, qc, qi, gain0, gain1, gain2, pgain0, pgain1, pgain2)
    res : scraps.Resonator object
        A Resonator object.
    residual : bool
        Whether to return a residual (True) or to return the model calcuated at the frequencies present in res (False).

    Keyword Arguments
    -----------------
    freqs : list-like
        A list of frequency points at which to calculate the model. Only used if `residual=False`

    remove_baseline : bool
        Whether or not to remove the baseline during calculation (i.e. ignore pgain and gain polynomials). Default is False.

    only_baseline: bool
        Whether or not to calculate and return only the baseline. Default is False.

    Returns
    -------
    model or (model-data)/eps : ``numpy.array``
        If residual=True is specified, the return is the residuals weighted by the uncertainties. If residual=False, the return is the model
        values calculated at the frequency points. The returned array is in the form
        ``I + Q`` or ``residualI + residualQ``.

    """  
    # Repackage resonator data in 1D vector form
    data = np.concatenate((res.I, res.Q), axis=0)

    if residual:
        if (res.sigmaI is not None) and (res.sigmaQ is not None):
            cmplxSigma = np.concatenate((res.sigmaI, res.sigmaQ), axis=0)
        else:
            # Calculate eps from stdev of first 10 pts of data if not supplied
            epsI = np.std(sps.detrend(res.I[0:10]))
            epsQ = np.std(sps.detrend(res.Q[0:10]))
            cmplxSigma = np.concatenate((np.full_like(res.I, epsI), np.full_like(res.Q, epsQ)))

    # Use the clockwise hanger_resonator model - returns 1D array of concatenated [I, Q]
    model = hanger_fit(paramsVec, res, residual=False, **kwargs)

    # For counterclockwise convention we take the conjugate of the model (flip the sign of Q)  
    model[model.size // 2:] *= -1

    # Return model or residual
    if residual == True:
        return (model - data) / cmplxSigma
    else:
        return model

def hanger_ccw_params(res, **kwargs):
    """Initialize fitting parameters used by the hanger_ccw_fit function.

    Parameters
    ----------
    res : ``scraps.Resonator`` object
        The object you want to calculate parameter guesses for.

    Keyword Arguments
    -----------------
    fit_quadratic_phase : bool
        This determines whether the phase baseline is fit by a line or a
        quadratic function. Default is False for fitting only a line.

    hardware : string {'VNA', 'mixer'}
        This determines whether or not the Ioffset and Qoffset parameters are
        allowed to vary by default.

    use_filter : bool
        Whether or not to use a smoothing filter on the data before calculating
        parameter guesses. This is especially useful for very noisy data where
        the noise spikes might be lower than the resonance minimum.

    filter_win_length : int
        The length of the window used in the Savitsky-Golay filter that smoothes
        the data when ``use_filter == True``. Default is ``0.1 * len(data)`` or
        3, whichever is larger.

    Returns
    -------
    params : ``lmfit.Parameters`` object

    """
    params = hanger_params(res, **kwargs)

    # For the CCW convention we flip the sign of the pgain parameters
    params.get('pgain0').value *= -1
    params.get('pgain1').value *= -1
    params.get('pgain2').value *= -1

    return params