import numpy as np
import pytest

from hht_tools import (
    decompose_emd, instantaneous_frequency, if_at_peak, zero_crossing_freq,
)

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


def test_separate_residual_prevents_trend_from_dominating_peak_frequency():
    t, lo = tone(3.)
    _, hi = tone(8., amp=0.5)
    x = lo + hi + 5 * t / t[-1]
    imfs, residual = decompose_emd(x, return_residual=True)
    result = if_at_peak(imfs, x, FS, t)
    assert result["f_dom"] == pytest.approx(3., abs=0.2)
    # Full-stack analysis remains explicit and available for reproducing
    # the original published measurement method.
    full = if_at_peak(np.vstack([imfs, residual]), x, FS, t)
    assert full["f_dom"] < 0.5


@pytest.mark.parametrize("x", [[1, 0, 1, 2, 1, 0, 1], [0]*10,
                                [0, 0, 1, 2, 1, 0, 0]])
def test_zero_touches_and_end_zeros_are_not_crossings(x):
    assert np.isnan(zero_crossing_freq(x, FS))


@pytest.mark.parametrize("x, expected", [
    ([-1, 0, 1, 2, 1, 0, -1], FS / 8),
    ([-1, 0, 0, 1, 2, 1, 0, 0, -1], FS / 10),
    ([-1, 1, 2, 1, -1], FS / 6),
])
def test_zero_crossings_count_each_sign_change_once(x, expected):
    assert zero_crossing_freq(x, FS) == expected


@pytest.mark.parametrize("fs", [0, -1, np.nan, np.inf])
def test_frequency_readers_reject_invalid_sample_rate(fs):
    t, x = tone(3.)
    for method, args in [
        (instantaneous_frequency, (x, fs)),
        (zero_crossing_freq, (x, fs)),
        (if_at_peak, (x[None, :], x, fs, t)),
    ]:
        with pytest.raises(ValueError, match="fs"):
            method(*args)


def test_peak_reader_rejects_nan_record():
    t, x = tone(3.)
    imfs = x[None, :].copy()
    x[100] = np.nan
    with pytest.raises(ValueError, match="x must"):
        if_at_peak(imfs, x, FS, t)


@pytest.mark.parametrize("bad_time", ["short", "wrong_spacing", "reversed", "nan"])
def test_peak_reader_checks_time(bad_time):
    t, x = tone(3.)
    t = {"short": t[:-1], "wrong_spacing": t * 100,
         "reversed": t[::-1], "nan": np.full_like(t, np.nan)}[bad_time]
    with pytest.raises(ValueError, match="time"):
        if_at_peak(x[None, :], x, FS, t)


@pytest.mark.parametrize("shape", [(1, 999), (1000,), (0, 1000)])
def test_peak_reader_checks_imf_shape(shape):
    t, x = tone(3.)
    with pytest.raises(ValueError, match="imfs"):
        if_at_peak(np.zeros(shape), x, FS, t)


@pytest.mark.parametrize("threshold", [-1, 1.01, np.nan])
def test_peak_reader_checks_amplitude_threshold(threshold):
    t, x = tone(3.)
    with pytest.raises(ValueError, match="amp_threshold"):
        if_at_peak(x[None, :], x, FS, t, amp_threshold=threshold)


def test_peak_reader_reports_no_valid_if_clearly():
    t, x = tone(3.)
    with pytest.raises(ValueError, match="no IMF has a valid IF"):
        if_at_peak(np.zeros((1, x.size)), x, FS, t)


def test_negative_edge_mask_is_rejected():
    t, x = tone(3.)
    with pytest.raises(ValueError, match="edge_mask_s"):
        instantaneous_frequency(x, FS, edge_mask_s=-1)
    with pytest.raises(ValueError, match="edge_mask_s"):
        if_at_peak(x[None, :], x, FS, t, edge_mask_s=-1)


def test_time_validation_accepts_float32_roundoff():
    t, x = tone(3.)
    result = if_at_peak(x[None, :], x, FS, t.astype(np.float32))
    assert result["f_dom"] == pytest.approx(3., abs=.05)
