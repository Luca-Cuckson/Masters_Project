from scipy.optimize import curve_fit, least_squares

import numpy as np
import numpy.typing as npt
import lmfit as lf
import scraps as scr

from functions.hanger_resonator_ccw import hanger_ccw_fit, hanger_ccw_params
from functions.looping_funcs import compute_loop_rotation


_DISABLED_POWER_DB = -120


def find_resonator_window(resonator: scr.Resonator, tones: npt.ArrayLike) -> tuple[float, float]:
    """Find the frequency window around a resonator that contains its tone."""
    tones = np.asarray(tones)
    left_bound = resonator.freq[0]
    right_bound = resonator.freq[-1]
    tone = (left_bound + right_bound) / 2

    left_candidates = tones[tones < tone]
    right_candidates = tones[tones > tone]

    left_tone = np.amax(left_candidates) if left_candidates.size > 0 else 4e9
    right_tone = np.amin(right_candidates) if right_candidates.size > 0 else 8e9

    if left_bound < left_tone:
        left_bound = (left_tone + tone) / 2
    else:
        left_bound += 150e3

    if right_bound > right_tone:
        right_bound = (right_tone + tone) / 2
    else:
        right_bound -= 150e3

    return left_bound, right_bound

def calculate_scan_tones(resonators: list[scr.Resonator]) -> np.ndarray:
    """Calculate scan tones as the midpoint of each resonator's frequency span."""
    return np.array([(resonator.freq[0] + resonator.freq[-1]) / 2 for resonator in resonators])

def calculate_default_resonator_parameters(
    res: scr.Resonator, scan_tones: np.ndarray, tone_idx: int
) -> lf.Parameters:
    """Create default fit parameters for a resonator's calculated fit window."""
    tone = scan_tones[tone_idx]
    left_bound, right_bound = find_resonator_window(res, scan_tones)
    res.freq_window = (left_bound, right_bound)

    mask = (res.freq >= left_bound) & (res.freq <= right_bound)
    if not np.any(mask):
        raise ValueError("Selected fit window does not contain any frequency samples")

    res_window = scr.makeResFromData({
        "name": res.name,
        "temp": res.temp,
        "pwr": res.pwr,
        "freq": res.freq[mask],
        "I": res.I[mask],
        "Q": res.Q[mask],
    })
    res_window.load_params(hanger_ccw_params)
    res_window.params["f0"].set(min=tone - 100e3, max=tone + 100e3)
    return res_window.params

def fit_resonator_window(resonator: scr.Resonator, tone: float, freq_window: tuple[float, float], params: lf.Parameters | None = None) -> lf.Parameters:
    """Fit a resonator in a bounded window and copy the fit onto the resonator."""
    left_bound, right_bound = freq_window
    mask = (resonator.freq >= left_bound) & (resonator.freq <= right_bound)
    if not np.any(mask):
        raise ValueError("Selected fit window does not contain any frequency samples")

    res_window = scr.makeResFromData({
        "name": resonator.name,
        "temp": resonator.temp,
        "pwr": resonator.pwr,
        "freq": resonator.freq[mask],
        "I": resonator.I[mask],
        "Q": resonator.Q[mask],
    })

    res_window.load_params(hanger_ccw_params if params is None else (lambda _: params))

    if params is None:
        res_window.params["f0"].set(min=tone - 100e3, max=tone + 100e3)

    res_window = lmfit_resonator(res_window, hanger_ccw_fit)
    fit_result = res_window.lmfit_result
    fit_params = fit_result["default"]["result"].params

    resonator.lmfit_result = fit_result
    resonator.params = res_window.params
    resonator.hasFit = True
    resonator.hasParams = True
    resonator.resultFreq = res_window.freq
    resonator.resultMag = res_window.resultMag
    resonator.resultPhase = res_window.resultPhase
    resonator.resultI = res_window.resultI
    resonator.resultQ = res_window.resultQ

    return fit_params

def fit_resonator(res: scr.Resonator, scan_tones: np.ndarray, tone_idx: int, params: lf.Parameters | None = None) -> tuple[bool, float | str | None]:
    """Fit one resonator and update its circle geometry and fit state."""
    res.center = 0 + 0j
    res.rotation = 0

    if res.pwr <= _DISABLED_POWER_DB:
        return True, None

    tone = scan_tones[tone_idx]
    left_bound, right_bound = find_resonator_window(res, scan_tones)
    res.freq_window = (left_bound, right_bound)
    res.hasFit = False

    fit_params = fit_resonator_window(res, tone, (left_bound, right_bound), params)
    f0 = fit_params["f0"].value
    qi = fit_params["qi"].value
    qc = fit_params["qc"].value
    q0 = 1 / (1 / qc + 1 / qi)
    linewidth = (f0 / q0) / 2

    f0_err = (fit_params["f0"].stderr or np.inf) / f0
    qi_err = (fit_params["qi"].stderr or np.inf) / qi
    qc_err = (fit_params["qc"].stderr or np.inf) / qc


    # Reject fit based on parameter errors
    if (f0_err > 0.01) or (qi_err > 1) or (qc_err > 1):
        # Flag as a failed fit
        res.hasFit = False

        # TODO: Changes from fixed 100kHz linewidth to the configured linewidth
        fit_resonator_circle(res, f0=tone, fa=tone-100e3, fb=tone+100e3)

        # TODO: Preference to throw and exception and catch it from the calling function
        return (False, f"Fit failed due to high parameter error: f0_err={f0_err:.3f}, qi_err={qi_err:.3f}, qc_err={qc_err:.3f}")
    else:
        # Use f0 and gamma to select fitting window for circle fit
        fit_resonator_circle(res, f0=f0, fa=f0-linewidth, fb=f0+linewidth)
        return (True, f0)


#TODO: Refactor required in where we split the fitting and resonator model to avoid this kind of hack
def lmfit_resonator(resonator: scr.Resonator, fitFn: callable, **kwargs):
    '''
    Resonator fitting helper that allows passing kwargs to the Minimizer.
    '''
    # Default parameters
    label = 'default'
    fit_type = 'IQ'

    #
    assert resonator.hasParams == True, "Must load params before running a fit."

    #Update any of the default Parameter guesses
    if kwargs is not None:
        for key, val in kwargs.items():
            #Allow for turning on and off parameter variation
            if '_vary' in key:
                key = key.split('_')[0]
                if key in resonator.params.keys():
                    if (val is True) or (val is False):
                        resonator.params[key].vary = val
            elif key in resonator.params.keys():
                resonator.params[key].value = val
            else:
                raise ValueError("Unknown key: "+key)

    # #Make complex vectors of the form cData = [reData, imData]
    # cmplxData = np.concatenate((self.I, self.Q), axis=0)

    # if (self.sigmaI is not None) and (self.sigmaQ is not None):
    #     cmplxSigma = np.concatenate((self.sigmaI, self.sigmaQ), axis=0)
    # else:
    #     cmplxSigma = None

    # #Create a lmfit minimizer object
    # minObj = lf.Minimizer(fitFn, self.params, fcn_args=(self.freq, cmplxData, cmplxSigma))

    #Create a lmfit minimizer object
    minObj = lf.Minimizer(fitFn, resonator.params, fcn_args=(resonator, True), max_nfev=300)

    lmfit_result = minObj.minimize(method = 'leastsq')

    #Call the lmfit minimizer method and minimize the residual
    if resonator.lmfit_result is None:
        resonator.lmfit_result = {}

    resonator.lmfit_result[label] = {}
    resonator.lmfit_result[label]['fit_type'] = fit_type
    resonator.lmfit_result[label]['result'] = lmfit_result
    resonator.lmfit_result[label]['values'] = np.asarray([val.value for key, val in lmfit_result.params.items() if val.vary is True])
    resonator.lmfit_result[label]['labels'] = [key for key, val in lmfit_result.params.items() if val.vary is True]

    #NOTE: These are likely to be deprecated
    if label == 'default':
        resonator.lmfit_vals = resonator.lmfit_result[label]['values']
        resonator.lmfit_labels = resonator.lmfit_result[label]['labels']


    #Set the hasFit flag NOTE:(scheduled for deprecation)
    resonator.hasFit = True

    #NOTE: This whole block may be deprecated
    if (fit_type == 'IQ') and (label == 'default'):
        #Add the data back to the final minimized residual to get the final fit
        #Also calculate all relevant curves
        cmplxResult = fitFn(resonator.lmfit_result[label]['result'].params, resonator, residual=False)
        cmplxResidual = resonator.lmfit_result[label]['result'].residual

        #Split the complex data back up into real and imaginary parts
        residualI, residualQ = np.split(cmplxResidual, 2)
        resultI, resultQ = np.split(cmplxResult, 2)

        resultMag = np.abs(resultI + 1j*resultQ)
        resultPhase = np.arctan2(resultQ,resultI)

        #Add some results back to the resonator object
        resonator.residualI = residualI
        resonator.residualQ = residualQ
        resonator.resultI = resultI
        resonator.resultQ = resultQ
        resonator.resultMag = resultMag
        resonator.resultPhase = resultPhase

    return resonator

def fit_resonator_circle(res: scr.Resonator, f0: float | None = None, fa: float | None = None, fb: float | None = None) -> None:
    """Fit a circle to resonator IQ data within the supplied frequency bounds."""
    f0 = res.fmin if f0 is None else f0
    fa = res.freq[0] if fa is None else fa
    fb = res.freq[-1] if fb is None else fb

    idx_f0 = np.argmin(np.abs(res.freq - f0))
    mask = (res.freq >= fa) & (res.freq <= fb)
    xc, yc, r = fit_circle(res.I[mask], res.Q[mask])
    center = xc + 1j * yc
    resonance = res.I[idx_f0] + 1j * res.Q[idx_f0]

    res.resonance = resonance
    res.center = center
    res.radius = r
    res.rotation = compute_loop_rotation(center, resonance)

def find_closed_region(x: npt.NDArray, y: npt.NDArray, idx_center: int) -> tuple[int, int]:
    """
    Find the indices of the closest points on either side of a given index.
    Useful for finding the loop cross-over point if the center point is known.

    Parameters
    ----------
    x, y : array-like
        Curve data points.
    idx_center : int
        Index corresponding to the center point, and the index to search
        either side of.
    
    Returns
    -------
    idx1, idx2 : int
        Indices of the closest points on either side of idx_center, representing
        points before and after the center.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if (idx_center <= 0) or (idx_center >= len(x) - 1):
        raise ValueError("idx_center must be a valid index within the range of x and y")

    # Compute pairwise distances for point sets either side of center
    dx = x[:idx_center, None] - x[None, idx_center:]
    dy = y[:idx_center, None] - y[None, idx_center:]
    dS2 = dx**2 + dy**2

    # Find the indices of the minimum distance
    idx1, idx2 = np.unravel_index(np.argmin(dS2), dS2.shape)
    idx2 += idx_center

    return idx1, idx2

def find_curvature_center(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[float, float]:
    """
    Estimate the center of curvature by intersecting normals to the curve.

    Uses the condition that the center lies along the normal direction,
    i.e. (center - point) ⟂ tangent.

    Parameters
    ----------
    x, y : array-like
        Curve coordinates.

    Returns
    -------
    xc, yc : float
        Estimated center of curvature.
    """
    # Tangent vectors
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Optional: normalize tangents for better conditioning
    mag = np.hypot(dx, dy)
    mask = mag > 1e-12  # avoid division by zero
    dx = dx[mask] / mag[mask]
    dy = dy[mask] / mag[mask]
    x = x[mask]
    y = y[mask]

    # Build linear system:
    # (xc - xi)*dx + (yc - yi)*dy = 0
    # → xc*dx + yc*dy = xi*dx + yi*dy
    A = np.column_stack([dx, dy])
    b = x * dx + y * dy

    # Solve least squares
    xc, yc = np.linalg.lstsq(A, b, rcond=None)[0]

    return xc, yc

def lorentzian(x: npt.ArrayLike, x0: float, gamma: float, A: float, offset: float = 0, slope: float = 0) -> np.ndarray:
    """
    Calculate a Lorentzian function with optional linear background.

    Parameters
    ----------
    x : np.ndarray
        Array of independent variable values at which to evaluate the function.
    x0 : float
        Center position (resonance frequency) of the Lorentzian peak.
    gamma : float
        Half-width at half-maximum (HWHM) of the Lorentzian peak.
        Controls the width of the peak.
    A : float
        Amplitude coefficient that scales the peak height.
        Peak maximum = A/gamma at x = x0 (when offset=0, slope=0).
    offset : float, optional
        Constant vertical offset (baseline). Default is 0.
    slope : float, optional
        Linear slope of the background as a function of (x - x0). Default is 0.

    Returns
    -------
    np.ndarray
        Array of Lorentzian values evaluated at each point in x, including
        the linear background contribution.

    Notes
    -----
    The functional form is:
        L(x) = A * gamma² / ((x - x0)² + gamma²) + offset + slope * (x - x0)

    The full width at half maximum (FWHM) is 2*gamma.
    """
    """Lorentzian with linear offset"""
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + offset + slope * (x - x0)

def fit_lorentzian(x: npt.ArrayLike, y:  npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a Lorentzian with linear offset to data.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable data (e.g., frequency).
    y : np.ndarray
        Dependent variable data (e.g., phase derivative).
    
    Returns
    -------
    popt : np.ndarray
        Optimal parameters [x0, gamma, A, offset, slope].
    pcov : np.ndarray
        Covariance matrix of the parameters.
    """
    # Initial guess for parameters
    x0 = x[np.argmax(y)]
    gamma = (x[-1] - x[0]) / 20
    amplitude = np.max(y) * gamma
    offset = np.median(y)
    slope = 0

    # Fit the data
    return curve_fit(lorentzian, x, y, p0=[x0, gamma, amplitude, offset, slope])

def circle(t: npt.ArrayLike, xc: float, yc: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Parametric circle coordinates.

    Parameters
    ----------
    t : array-like
        Parameter along circle (0 to 2*pi)
    xc, yc : float
        Center coordinates
    r : float
        Circle radius

    Returns
    -------
    x, y : ndarray
        Coordinates of points on the circle
    """
    x = xc + r * np.cos(t)
    y = yc + r * np.sin(t)

    return x, y

def fit_circle(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[float, float, float]:
    """
    Fit a circle to x/y data points using nonlinear least squares.

    Parameters
    ----------
    x : array-like
        x data points
    y : array-like
        y data points

    Returns
    -------
    xc : float
        Fitted center x-coordinate
    yc : float
        Fitted center y-coordinate
    r : float
        Fitted radius
    """
    def circle_residuals(params, x, y):
        """Calculate residuals for circle fitting."""
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r
    
    # Initial guesses for center and radius
    xc, yc = find_curvature_center(x, y)
    r = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))

    # Fit the circle using least squares
    result = least_squares(circle_residuals, [xc, yc, r], args=(x, y))
    xc, yc, r = result.x

    return xc, yc, r

