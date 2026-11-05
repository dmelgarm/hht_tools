import importlib

import numpy as np
import pytest

from hht_tools import envelope, instant_phase, spectrogram

spectrogram_module = importlib.import_module("hht_tools.spectrogram")


@pytest.fixture
def exact_imf(monkeypatch):
    """Isolate amplitude/binning from numerical EMD on a known exact IMF."""
    calls = []

    class ExactEMD:
        def emd(self, data, time, max_imf):
            calls.append(max_imf)
            self.imfs = data[None, :]

        def get_imfs_and_residue(self):
            return self.imfs, np.zeros(self.imfs.shape[1])

    monkeypatch.setattr(spectrogram_module, "EMD", ExactEMD)
    t = np.arange(1000) / 100.
    return t, np.sin(2 * np.pi * 5 * t), calls


def test_spectrogram_has_requested_bins_and_hilbert_amplitude(exact_imf):
    t, x, calls = exact_imf
    tx, periods, amp, imfs = spectrogram(
        t, x, 0.01, [0.1, 0.3], Ntime_bins=20, Nperiod_bins=10,
        Nimfs=1, return_imfs=True,
    )
    assert calls == [1]
    assert tx.shape == periods.shape == amp.shape == (10, 20)
    np.testing.assert_array_equal(imfs[0], x)
    # The unit-amplitude analytic sine has envelope one even at its zeros.
    assert amp.sum() == pytest.approx(len(x) - 1)
    np.testing.assert_allclose(tx[0], (np.arange(20) + .5) * t[-1] / 20)
    np.testing.assert_allclose(periods[:, 0], .1 + (np.arange(10) + .5) * .02)


def test_custom_period_edges_override_limits_and_bin_count(exact_imf):
    t, x, _ = exact_imf
    _, periods, amp = spectrogram(
        t, x, .01, None, Ntime_bins=1, period_bins=[.05, .15, .25, .5],
        Nperiod_bins=0,
    )
    np.testing.assert_allclose(periods[:, 0], [.1, .2, .375])
    assert amp.shape == (3, 1)
    assert amp[1, 0] == pytest.approx(len(x) - 1)
    assert amp[[0, 2], 0].sum() == 0


def test_nonpositive_frequency_is_excluded_without_divide_by_zero(exact_imf, monkeypatch):
    t, x, _ = exact_imf
    analytic = np.exp(-2j * np.pi * 5 * t)[None, :]
    analytic[:, :100] = 1  # Include a zero-frequency interval too.
    monkeypatch.setattr(spectrogram_module, "hilbert", lambda imfs: analytic)
    with np.errstate(divide="raise", invalid="raise"):
        _, _, amp = spectrogram(t, x, .01, [.1, .3])
    assert amp.sum() == 0


def test_real_emd_respects_imf_limit_and_returns_full_samples():
    t = np.arange(1000) / 100.
    x = np.sin(2 * np.pi * 3 * t) + .5 * np.sin(2 * np.pi * 12 * t)
    _, _, _, imfs = spectrogram(t, x, .01, [.02, 2], Nimfs=1, return_imfs=True)
    assert imfs.shape == (1, len(x))


@pytest.mark.parametrize("kwargs", [
    {"dt": 0}, {"dt": .02}, {"Ntime_bins": 0}, {"Nperiod_bins": 1.5},
    {"Nimfs": 0}, {"period_lims": [0, 1]}, {"period_lims": [1, .1]},
    {"period_bins": [.1, .1, .3]}, {"period_bins": [-1, .1]},
    {"return_imfs": "False"},
])
def test_spectrogram_rejects_invalid_parameters(exact_imf, kwargs):
    t, x, _ = exact_imf
    parameters = dict(dt=.01, period_lims=[.1, .3])
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        spectrogram(t, x, **parameters)


def test_analytic_helpers_match_a_unit_tone():
    t = np.arange(1000) / 100.
    x = np.cos(2 * np.pi * 5 * t)
    np.testing.assert_allclose(envelope(x), 1, atol=1e-12)
    np.testing.assert_allclose(instant_phase(x), 2 * np.pi * 5 * t, atol=1e-12)
    assert instant_phase(x[None, :]).shape == (1, len(x))


@pytest.mark.parametrize("x", [[1, np.nan], [1j, 2j], [1]])
def test_analytic_helpers_reject_invalid_signals(x):
    for method in [envelope, instant_phase]:
        with pytest.raises(ValueError):
            method(x)
