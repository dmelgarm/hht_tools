"""
EMD and CEEMDAN decomposition wrappers.

CEEMDAN is the default for strong motion: plain EMD mode-mixes on
intermittent signals. Seeding is mandatory here — reproducibility is a
hard requirement of the PGA/PGV frequency study — so the seed is a
required, explicit argument with no default RNG state fallback.
"""

import numpy as np
from PyEMD import EMD, CEEMDAN


def decompose_emd(x, max_imf=10):
    """
    Plain EMD decomposition.

    Parameters
    ----------
    x : np.array
        Signal to decompose
    max_imf : int, optional
        Maximum number of IMFs. The default is 10.

    Returns
    -------
    imfs : np.array
        2D array (n_imfs, n_samples), residual included as last row
    """

    emd = EMD()
    return emd.emd(np.asarray(x, dtype=float), max_imf=max_imf)


def decompose_ceemdan(x, seed, trials=200, epsilon=0.15, max_imf=10,
                      parallel=False):
    """
    CEEMDAN decomposition, seeded for reproducibility.

    Parameters
    ----------
    x : np.array
        Signal to decompose
    seed : int
        Seed for the noise realizations. Required: unseeded runs are not
        reproducible and are not allowed in this study.
    trials : int, optional
        Ensemble size. The default is 200; sensitivity tests should
        sweep a few hundred.
    epsilon : float, optional
        Added-noise std as a fraction of the signal std. The default is
        0.15; sensitivity tests should sweep 0.1 to 0.2.
    max_imf : int, optional
        Maximum number of IMFs. The default is 10.
    parallel : bool, optional
        Parallelize over trials. The default is False.

    Returns
    -------
    imfs : np.array
        2D array (n_imfs, n_samples)
    """

    ce = CEEMDAN(trials=trials, epsilon=epsilon, parallel=parallel)
    # noise_seed() works on every PyEMD version; the seed= constructor
    # kwarg is silently ignored by older PyEMD (<1.5), so never use it.
    ce.noise_seed(seed)
    return ce.ceemdan(np.asarray(x, dtype=float), max_imf=max_imf)
