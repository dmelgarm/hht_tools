import numpy as np
import pytest

from hht_tools import instantaneous_frequency, if_at_peak, zero_crossing_freq

FS = 100.0


def tone(freq, amp=1.0, dur=10.0):
    t = np.arange(0, dur, 1 / FS)
    return t, amp * np.sin(2 * np.pi * freq * t)


def test_if_of_pure_tone():
    _, x = tone(3.0)
    f, amp = instantaneous_frequency(x, FS)
    interior = slice(int(FS), -int(FS))
    assert np.allclose(f[interior], 3.0, atol=0.05)
    assert np.allclose(amp[interior], 1.0, atol=0.05)


def test_edge_mask_sets_nan():
    _, x = tone(3.0)
    f, _ = instantaneous_frequency(x, FS, edge_mask_s=1.0)
    n = int(FS)
    assert np.all(np.isnan(f[:n]))
    assert np.all(np.isnan(f[-n:]))
    assert np.all(np.isfinite(f[n:-n]))


def test_zero_crossing_freq_tone():
    _, x = tone(3.0)
    assert zero_crossing_freq(x, FS) == pytest.approx(3.0, rel=0.05)


def test_zero_crossing_freq_unbracketed():
    # peak at the very start, no zero crossing before it
    x = np.exp(-np.arange(0, 5, 1 / FS))
    assert np.isnan(zero_crossing_freq(x, FS))


def test_if_at_peak_triplet():
    # two synthetic "IMFs": dominant 2 Hz and weaker 10 Hz
    t, x1 = tone(2.0, amp=1.0)
    _, x2 = tone(10.0, amp=0.3)
    imfs = np.vstack([x2, x1])
    x = imfs.sum(axis=0)
    res = if_at_peak(imfs, x, FS, t)
    assert res["n_used"] == 2
    assert res["f_dom"] == pytest.approx(2.0, abs=0.1)
    assert 2.0 < res["f_mean"] < 10.0
    assert res["f_spread"] > 0
    assert not res["peak_in_edge"]


def test_if_at_peak_amplitude_threshold():
    # a third mode far below the noise-floor threshold must be excluded
    t, x1 = tone(2.0, amp=1.0)
    _, x2 = tone(10.0, amp=0.3)
    _, x3 = tone(30.0, amp=1e-3)
    imfs = np.vstack([x3, x2, x1])
    x = imfs.sum(axis=0)
    res = if_at_peak(imfs, x, FS, t, amp_threshold=0.05)
    assert res["n_used"] == 2


def test_if_at_peak_edge_flag():
    # ramp puts the peak at the record end, inside the edge mask
    t = np.arange(0, 10, 1 / FS)
    x = t / t.max() * np.sin(2 * np.pi * 2.0 * t)
    imfs = x[np.newaxis, :]
    res = if_at_peak(imfs, x, FS, t, edge_mask_s=1.0)
    assert res["peak_in_edge"]
