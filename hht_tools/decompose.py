"""
EMD and CEEMDAN decomposition wrappers.

Each call owns its decomposition object and random state. The default
serial CEEMDAN execution is reproducible within a fixed dependency
environment when given the same signal, seed, and parameters.
"""

import numpy as np
from PyEMD import EMD, CEEMDAN

from ._validation import imf_limit, positive_integer, real_scalar, signal_array


def decompose_emd(x, max_imf=10, *, return_residual=False):
    """
    Plain EMD decomposition.

    Parameters
    ----------
    x : np.array
        Signal to decompose
    max_imf : int, optional
        Maximum number of IMFs, excluding the residual. Use -1 for no
        explicit limit. The default is 10.
    return_residual : bool, optional
        Return (imfs, residual) separately. The default False preserves
        the original PyEMD component stack.

    Returns
    -------
    imfs : np.array
        2D component stack (n_components, n_samples). PyEMD appends a
        nonzero residual as the last row; it may omit a zero residual.
        With return_residual=True, return a tuple containing only the
        oscillatory IMFs and a separate 1D residual. This is the preferred
        input to per-IMF frequency analysis.
    """

    x = signal_array(x, "x").astype(float)
    imf_limit(max_imf)
    if not isinstance(return_residual, (bool, np.bool_)):
        raise ValueError("return_residual must be a boolean")
    emd = EMD()
    components = emd.emd(x, max_imf=max_imf)
    if return_residual:
        return emd.get_imfs_and_residue()
    return components


def decompose_ceemdan(x, seed, trials=200, epsilon=0.15, max_imf=10,
                      parallel=False, *, return_residual=False):
    """
    CEEMDAN decomposition, seeded for reproducibility.

    Parameters
    ----------
    x : np.array
        Signal to decompose
    seed : int
        Integer seed in [0, 2**32 - 1] for the noise realizations.
        None is not allowed.
    trials : int, optional
        Positive ensemble size. The default is 200.
    epsilon : float, optional
        Nonnegative noise scale relative to the evolving residual's
        standard deviation. The default is 0.15.
    max_imf : int, optional
        Maximum number of IMFs, excluding the residual. Use -1 for no
        explicit limit. The default is 10.
    parallel : bool, optional
        Parallelize over trials. The default is False. Parallel reduction
        order is not deterministic; seeded reproducibility is guaranteed
        only with parallel=False in a fixed dependency environment.
    return_residual : bool, optional
        Return (imfs, residual) separately. The default False preserves
        the original component stack, including its final residual.

    Returns
    -------
    imfs : np.array
        2D component stack (n_components, n_samples), with the residual
        in the last row. With return_residual=True, return a tuple of
        oscillatory IMFs and a separate 1D residual.
    """

    x = signal_array(x, "x").astype(float)
    if np.std(x) == 0:
        raise ValueError("CEEMDAN requires a signal with nonzero standard deviation")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32):
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    positive_integer(trials, "trials")
    real_scalar(epsilon, "epsilon")
    imf_limit(max_imf)
    if not isinstance(parallel, (bool, np.bool_)):
        raise ValueError("parallel must be a boolean")
    if not isinstance(return_residual, (bool, np.bool_)):
        raise ValueError("return_residual must be a boolean")
    ce = CEEMDAN(trials=trials, epsilon=epsilon, parallel=parallel)
    ce.noise_seed(seed)
    components = ce.ceemdan(x, max_imf=max_imf)
    if return_residual:
        # ce.residue is the reconstruction error of the complete stack,
        # not its final trend component. CEEMDAN always appends that trend.
        return components[:-1], components[-1]
    return components
