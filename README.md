# hht_tools

Hilbert-Huang transform utilities for seismology: seeded EMD/CEEMDAN
decomposition wrappers, instantaneous frequency (IF) readers at signal
peaks, and HHT spectrograms.

## Install

```bash
git clone https://github.com/dmelgarm/hht_tools.git
cd hht_tools
python -m pip install .
```

Requires Python >= 3.9, NumPy, SciPy, and
[EMD-signal](https://pypi.org/project/EMD-signal/) >= 1.5 (imported as
`PyEMD`; this is different from the Earth Mover's Distance package `pyemd`).
To reproduce a release, check out its tag before installing. For development,
use `python -m pip install -e ".[test]"`.

## What's in it

- `decompose_ceemdan(x, seed, trials, epsilon, ...)` / `decompose_emd(x)` —
  decomposition wrappers. CEEMDAN requires an integer seed. Set
  `return_residual=True` to obtain `(imfs, residual)` separately.
- `instantaneous_frequency(imf, fs, edge_mask_s)` — per-IMF IF and Hilbert
  amplitude, with optional masking of the unreliable record edges.
- `if_at_peak(imfs, x, fs, t)` — the IF triplet at the time of max |x|:
  IF of the amplitude-dominant IMF, amplitude-weighted mean IF, and
  amplitude-weighted IF spread (a measure of whether the peak is
  monochromatic or an interference of scales). The default screen excludes
  rows below 5% of the largest valid Hilbert amplitude at the peak; this is
  a relative amplitude threshold, not an estimated CEEMDAN noise floor.
- `zero_crossing_freq(x, fs)` — model-free cross-check: half period
  between the zero crossings that bracket the peak.
- `spectrogram(time, data, dt, period_lims, ...)` — EMD-based Hilbert
  amplitude sums in time/period bins. Counts specify numbers of bins;
  returned grids contain bin centers. Custom `period_bins` are edges.
  The residual is excluded. Amplitudes are sums, not power or a density,
  and depend on bin widths and sampling rate. See the function docstring.
- `envelope`, `instant_phase` — Hilbert amplitude and unwrapped phase.

Inputs must be finite real arrays. Sampling rates/intervals must be positive,
and time vectors must be uniformly increasing and consistent with them.
CEEMDAN rejects constant signals because it normalizes by signal standard
deviation. Expected unbracketed zero-crossing measurements return NaN;
invalid inputs raise ValueError.

## Example

```python
import numpy as np
from hht_tools import decompose_ceemdan, if_at_peak, zero_crossing_freq

fs = 100.0
t = np.arange(0, 40, 1 / fs)
x = np.sin(2 * np.pi * 2 * t) + 0.3 * np.sin(2 * np.pi * 10 * t)

imfs, residual = decompose_ceemdan(
    x, seed=0, trials=200, epsilon=0.15, return_residual=True,
)
assert np.allclose(imfs.sum(axis=0) + residual, x)
res = if_at_peak(imfs, x, fs, t)
print(res["f_dom"], res["f_mean"], res["f_spread"])
print("Peak near an edge:", res["peak_in_edge"])
print(zero_crossing_freq(x, fs))  # model-free check
```

The synthetic `demo.ipynb` demonstrates the spectrogram. Install its optional
dependencies with `python -m pip install ".[demo]"`, then open it in Jupyter.

## Reproducibility and residual handling

Seeded CEEMDAN is reproducible with `parallel=False` (the default) in a fixed
dependency environment. Parallel accumulation order can change the result.
Each call creates its own decomposition object and random state; there is no
shared RNG. Record the seed, parameters, and dependency versions with results.
Identical results across different dependency versions are not guaranteed.

For compatibility, the default decomposition return value is the original
PyEMD component stack. CEEMDAN appends a residual; EMD appends a residual only
when it is nonzero. **Do not blindly discard the final EMD row:** it may be
an oscillatory IMF. Use `return_residual=True` to separate it reliably.

`if_at_peak` screens every supplied row and does not infer which one is a
residual. A supplied trend can contribute if its Hilbert IF passes the screen.
The original full-stack peak-frequency calculation is preserved:

```python
components = decompose_ceemdan(x, seed=0)
original_result = if_at_peak(components, x, fs, t)
```

Separating the residual changes the set of components used in a measurement;
it is an explicit methodological choice, not an automatic cleanup. The
decomposition defaults, phase-gradient IF estimator, amplitude weighting,
and peak-screening thresholds are unchanged in 0.3.0. The original study
environment used Python 3.11.13, NumPy 1.26.4, SciPy 1.17.0, and EMD-signal
1.10.0. This repository supplies reusable tools and synthetic examples;
study-specific analysis scripts and observational data are separate.

## Tests

```bash
python -m pip install ".[test]"
python -m pytest tests/
```

## Citation and license

Distributed under the MIT license. Citation metadata is in `CITATION.cff`.
When using an archived release, cite that release's Zenodo DOI and version.
See `CHANGELOG.md` for behavior changes and release preparation instructions.
