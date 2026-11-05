import numpy as np
import pytest

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


@pytest.mark.parametrize("seed", [None, -1, 2**32, 1.5, True])
def test_ceemdan_rejects_invalid_seed(seed):
    _, x = two_tone()
    with pytest.raises(ValueError, match="seed"):
        decompose_ceemdan(x, seed=seed)


@pytest.mark.parametrize("x", [[], [1], [0, np.nan], [0, np.inf],
                                [[0, 1], [2, 3]], [1j, 2j]])
@pytest.mark.parametrize("method", [decompose_emd, decompose_ceemdan])
def test_decomposition_rejects_invalid_signal(method, x):
    kwargs = {"seed": 0} if method is decompose_ceemdan else {}
    with pytest.raises(ValueError, match="x must"):
        method(x, **kwargs)


@pytest.mark.parametrize("level", [0., 3.])
def test_ceemdan_rejects_constant_signal(level):
    with pytest.raises(ValueError, match="nonzero standard deviation"):
        decompose_ceemdan(np.full(100, level), seed=0)


@pytest.mark.parametrize("kwargs", [
    {"trials": 0}, {"trials": 1.5}, {"epsilon": -1}, {"epsilon": np.nan},
    {"max_imf": 0}, {"max_imf": -1.0}, {"parallel": "False"},
    {"return_residual": "False"},
])
def test_ceemdan_rejects_invalid_parameters(kwargs):
    _, x = two_tone()
    with pytest.raises(ValueError):
        decompose_ceemdan(x, seed=0, **kwargs)


@pytest.mark.parametrize("method", [decompose_emd, decompose_ceemdan])
def test_residual_can_be_separated_without_changing_decomposition(method):
    t, x = two_tone()
    x = x + t / t[-1]
    kwargs = {"seed": 7, "trials": 3} if method is decompose_ceemdan else {}
    components = method(x, **kwargs)
    imfs, residual = method(x, return_residual=True, **kwargs)
    assert imfs.shape[1] == x.size
    assert residual.shape == x.shape
    np.testing.assert_array_equal(imfs, components[:-1])
    np.testing.assert_array_equal(residual, components[-1])
    np.testing.assert_allclose(imfs.sum(axis=0) + residual, x, atol=1e-12)


def test_emd_does_not_discard_a_real_final_imf_when_residual_is_zero():
    t = np.arange(1000) / FS
    x = np.sin(2 * np.pi * 3 * t)
    imfs, residual = decompose_emd(x, return_residual=True)
    assert imfs.shape[0] >= 1
    np.testing.assert_allclose(imfs.sum(axis=0) + residual, x, atol=1e-12)
