"""
Instantaneous frequency (IF) readers for IMF stacks.

IF is defined per IMF, never for the composite record. At a peak we
report a triplet: IF of the amplitude-dominant IMF, amplitude-weighted
mean IF across IMFs, and the amplitude-weighted IF spread. The spread is
a result, not a nuisance: it measures whether the peak is monochromatic
or an interference of scales. zero_crossing_freq() is the model-free
cross-check that must always accompany the triplet.
"""

import numpy as np
from scipy.signal import hilbert


def instantaneous_frequency(imf, fs, edge_mask_s=0.0):
    """
    IF and amplitude envelope of one IMF via the Hilbert transform.

    Parameters
    ----------
    imf : np.array
        One intrinsic mode function
    fs : float
        Sampling rate in Hz
    edge_mask_s : float, optional
        Set IF to NaN within this many seconds of each record edge,
        where the Hilbert transform is unreliable. The default is 0
        (no masking).

    Returns
    -------
    f : np.array
        Instantaneous frequency in Hz
    amp : np.array
        Hilbert amplitude envelope
    """

    z = hilbert(imf)
    phase = np.unwrap(np.angle(z))
    f = np.gradient(phase) * fs / (2.0 * np.pi)
    amp = np.abs(z)
    n = int(round(edge_mask_s * fs))
    if n > 0:
        f[:n] = np.nan
        f[-n:] = np.nan
    return f, amp


def if_at_peak(imfs, x, fs, t, amp_threshold=0.05, edge_mask_s=1.0):
    """
    IF triplet at the time of max |x|.

    Parameters
    ----------
    imfs : np.array
        2D IMF stack (n_imfs, n_samples) from a decomposition of x
    x : np.array
        The composite record (used only to locate the peak)
    fs : float
        Sampling rate in Hz
    t : np.array
        Time vector for x
    amp_threshold : float, optional
        Exclude IMFs whose Hilbert amplitude at the peak is below this
        fraction of the amplitude-dominant IMF (CEEMDAN leaves a noise
        floor in low-order modes). The default is 0.05.
    edge_mask_s : float, optional
        Flag the result if the peak falls within this many seconds of a
        record edge. The default is 1.0.

    Returns
    -------
    dict with keys
        t_peak, i_peak : peak time and sample index
        f_dom : IF of the amplitude-dominant IMF (Hz)
        f_mean : amplitude-weighted mean IF across retained IMFs (Hz)
        f_spread : amplitude-weighted IF spread (Hz)
        f_all, a_all : per-IMF IF and amplitude at the peak, after screening
        k_dom : index into f_all of the dominant IMF
        n_used : number of IMFs retained
        peak_in_edge : True if the peak lies inside the edge mask
    """

    x = np.asarray(x, dtype=float)
    ipk = int(np.argmax(np.abs(x)))

    fis, amps = [], []
    for imf in imfs:
        f, a = instantaneous_frequency(imf, fs)
        fis.append(f[ipk])
        amps.append(a[ipk])
    fis, amps = np.array(fis), np.array(amps)

    # keep physically meaningful IF values, then amplitude-threshold
    good = np.isfinite(fis) & (fis > 0) & (fis < fs / 2)
    fis, amps = fis[good], amps[good]
    if fis.size == 0:
        raise ValueError("no IMF has a valid IF at the peak")
    keep = amps >= amp_threshold * amps.max()
    fis, amps = fis[keep], amps[keep]

    kdom = int(np.argmax(amps))
    w = amps / amps.sum()
    fmean = float(np.sum(w * fis))
    fspread = float(np.sqrt(np.sum(w * (fis - fmean) ** 2)))

    peak_in_edge = (t[ipk] < t[0] + edge_mask_s) or (t[ipk] > t[-1] - edge_mask_s)

    return dict(t_peak=float(t[ipk]), i_peak=ipk, f_dom=float(fis[kdom]),
                f_mean=fmean, f_spread=fspread, f_all=fis, a_all=amps,
                k_dom=kdom, n_used=int(fis.size), peak_in_edge=peak_in_edge)


def zero_crossing_freq(x, fs):
    """
    Model-free IF check at the peak of |x|: half period between the zero
    crossings that bracket the peak.

    Parameters
    ----------
    x : np.array
        The composite record
    fs : float
        Sampling rate in Hz

    Returns
    -------
    f : float
        1 / (2 * half period) in Hz, or NaN if the peak is not bracketed
    """

    x = np.asarray(x, dtype=float)
    ipk = int(np.argmax(np.abs(x)))
    s = np.sign(x)
    zc = np.where(np.diff(s) != 0)[0]
    before = zc[zc < ipk]
    after = zc[zc >= ipk]
    if len(before) == 0 or len(after) == 0:
        return np.nan
    half_period = (after[0] - before[-1]) / fs
    return 1.0 / (2.0 * half_period)
