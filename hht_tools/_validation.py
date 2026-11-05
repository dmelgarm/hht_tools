"""Input checks shared by the public signal-processing functions."""

from numbers import Integral, Real

import numpy as np


def signal_array(x, name):
    x = np.asarray(x)
    if (x.ndim != 1 or x.size < 2 or x.dtype.kind not in "biuf"
            or not np.all(np.isfinite(x))):
        raise ValueError(f"{name} must be a finite real 1D array with at least two samples")
    return x


def real_scalar(value, name, *, strictly_positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0
            or (strictly_positive and value == 0)):
        bound = "positive" if strictly_positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {bound} real number")
    return value


def positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return value


def imf_limit(value):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)):
        raise ValueError("max_imf must be -1 or a positive integer")
    if value != -1:
        positive_integer(value, "max_imf")
    return value


def time_array(t, n_samples, dt):
    t = signal_array(t, "time")
    if t.size != n_samples:
        raise ValueError("time and signal must have the same number of samples")
    # Allow floating-point subtraction error when time has a nonzero origin.
    precision = np.finfo(t.dtype).eps if t.dtype.kind == "f" else np.finfo(float).eps
    atol = 8 * precision * max(1.0, np.max(np.abs(t)))
    steps = np.diff(t)
    if np.any(steps <= 0) or not np.allclose(steps, dt, rtol=1e-6, atol=atol):
        raise ValueError("time must increase uniformly with spacing matching the sampling rate")
    return t
