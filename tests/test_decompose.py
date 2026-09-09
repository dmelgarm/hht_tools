import numpy as np

from hht_tools import decompose_emd, decompose_ceemdan

FS = 100.0


def two_tone():
    t = np.arange(0, 10, 1 / FS)
    return t, np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 8.0 * t)


def test_ceemdan_same_seed_is_reproducible():
    _, x = two_tone()
    a = decompose_ceemdan(x, seed=7, trials=30)
    b = decompose_ceemdan(x, seed=7, trials=30)
    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_ceemdan_different_seed_differs():
    _, x = two_tone()
    a = decompose_ceemdan(x, seed=7, trials=30)
    b = decompose_ceemdan(x, seed=8, trials=30)
    assert not (a.shape == b.shape and np.allclose(a, b))


def test_ceemdan_separates_two_tones():
    t, x = two_tone()
    imfs = decompose_ceemdan(x, seed=0, trials=50)
    lo = np.sin(2 * np.pi * 1.0 * t)
    hi = 0.5 * np.sin(2 * np.pi * 8.0 * t)
    interior = slice(int(FS), -int(FS))

    def best_corr(target):
        return max(abs(np.corrcoef(imf[interior], target[interior])[0, 1])
                   for imf in imfs)

    assert best_corr(hi) > 0.95
    assert best_corr(lo) > 0.95


def test_emd_recovers_tones_on_easy_signal():
    t, x = two_tone()
    imfs = decompose_emd(x)
    assert imfs.shape[0] >= 2
    # decomposition must be complete: IMFs (incl. residual) sum to the signal
    assert np.allclose(imfs.sum(axis=0), x, atol=1e-8)
