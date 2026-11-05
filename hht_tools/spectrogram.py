"""EMD-based Hilbert amplitude spectrograms and analytic-signal utilities."""

from PyEMD import EMD
from scipy.signal import hilbert
import numpy as np

from ._validation import (
    imf_limit, positive_integer, real_scalar, signal_array, time_array,
)


def instant_phase(imfs):
    """Return unwrapped Hilbert phase along the last (sample) axis.

    Accept a real signal or a 2D IMF stack, with at least two samples.
    """
    imfs = np.asarray(imfs)
    if (imfs.ndim not in (1, 2) or imfs.size == 0 or imfs.shape[-1] < 2
            or imfs.dtype.kind not in "biuf" or not np.all(np.isfinite(imfs))):
        raise ValueError("imfs must be a finite real signal or 2D IMF stack")
    return np.unwrap(np.angle(hilbert(imfs)), axis=-1)


def spectrogram(time, data, dt, period_lims, Ntime_bins=200, period_bins=None,
                Nperiod_bins=100, Nimfs=10, return_imfs=False):
    """Bin EMD Hilbert amplitude by time and instantaneous period.

    Parameters
    ----------
    time, data : array_like
        Finite 1D arrays of equal length. Time must increase uniformly.
    dt : float
        Positive sampling interval in seconds, matching time.
    period_lims : (float, float)
        Positive increasing period limits in seconds. Used only when
        period_bins is None.
    Ntime_bins, Nperiod_bins : int, optional
        Numbers of bins, not numbers of edges. Defaults are 200 and 100.
        Nperiod_bins is used only when period_bins is None.
    period_bins : array_like, optional
        Positive, strictly increasing period-bin edges in seconds.
        Overrides period_lims and Nperiod_bins.
    Nimfs : int, optional
        Maximum number of oscillatory IMFs to extract; -1 means no
        explicit limit. The residual is excluded from the spectrogram.
    return_imfs : bool, optional
        Also return the full-length oscillatory IMFs (without residual).

    Returns
    -------
    time_out, periods_out, amplitude_out : ndarray
        Arrays of shape (n_period_bins, Ntime_bins). Coordinates are bin
        centers. Amplitudes are sums of Hilbert envelope samples, in the
        input signal's units, not power, a density, or decibels. Values
        depend on sampling rate and bin width. Frequency uses forward
        phase differences; envelope weights and times are averaged at
        the corresponding sample-interval midpoints. Nonpositive IF is
        excluded rather than folded to positive frequency.
    imfs : ndarray, optional
        Full-length IMF stack, returned only when return_imfs=True.
    """
    data = signal_array(data, "data")
    real_scalar(dt, "dt", strictly_positive=True)
    time = time_array(time, data.size, dt)
    positive_integer(Ntime_bins, "Ntime_bins")
    imf_limit(Nimfs)
    if not isinstance(return_imfs, (bool, np.bool_)):
        raise ValueError("return_imfs must be a boolean")
    if period_bins is None:
        limits = signal_array(period_lims, "period_lims")
        if limits.size != 2 or limits[0] <= 0 or limits[1] <= limits[0]:
            raise ValueError("period_lims must contain two positive increasing limits")
        positive_integer(Nperiod_bins, "Nperiod_bins")
        y_edges = np.linspace(limits[0], limits[1], Nperiod_bins + 1)
    else:
        y_edges = signal_array(period_bins, "period_bins")
        if y_edges[0] <= 0 or np.any(np.diff(y_edges) <= 0):
            raise ValueError("period_bins must be positive strictly increasing edges")

    emd = EMD()
    emd.emd(data, time, max_imf=Nimfs)
    imfs, _ = emd.get_imfs_and_residue()
    if imfs.shape[0] == 0:
        raise ValueError("no oscillatory IMFs found")

    analytic = hilbert(imfs)
    freqs = np.diff(np.unwrap(np.angle(analytic)), axis=-1) / (2 * np.pi * dt)
    amp = np.abs(analytic)
    weights = (amp[:, :-1] + amp[:, 1:]) / 2
    sample_times = np.broadcast_to((time[:-1] + time[1:]) / 2, freqs.shape)
    good = np.isfinite(freqs) & (freqs > 0)
    x_edges = np.linspace(time[0], time[-1], Ntime_bins + 1)
    stat, _, _ = np.histogram2d(
        sample_times[good], 1 / freqs[good], weights=weights[good],
        bins=(x_edges, y_edges),
    )
    time_out, periods_out = np.meshgrid(
        (x_edges[:-1] + x_edges[1:]) / 2,
        (y_edges[:-1] + y_edges[1:]) / 2,
    )
    result = (time_out, periods_out, stat.T)
    if return_imfs:
        return (*result, imfs)
    return result


def envelope(signal):
    """Return the Hilbert amplitude envelope of a finite real 1D signal."""
    return np.abs(hilbert(signal_array(signal, "signal")))
