# hht_tools

Hilbert-Huang transform utilities for seismology: seeded EMD/CEEMDAN
decomposition wrappers, instantaneous frequency (IF) readers at signal
peaks, and HHT spectrograms.

## Install

```
pip install -e .
```

Requires numpy, scipy, and [EMD-signal](https://pypi.org/project/EMD-signal/) >= 1.5
(older versions silently ignore the CEEMDAN seed, breaking reproducibility).

## What's in it

- `decompose_ceemdan(x, seed, trials, epsilon, ...)` / `decompose_emd(x)` —
  decomposition wrappers. CEEMDAN is the default for intermittent signals
  (strong motion); the seed is a required argument so every run is
  reproducible.
- `instantaneous_frequency(imf, fs, edge_mask_s)` — per-IMF IF and Hilbert
  amplitude, with optional masking of the unreliable record edges.
- `if_at_peak(imfs, x, fs, t)` — the IF triplet at the time of max |x|:
  IF of the amplitude-dominant IMF, amplitude-weighted mean IF, and
  amplitude-weighted IF spread (a measure of whether the peak is
  monochromatic or an interference of scales). Low-amplitude IMFs below
  the CEEMDAN noise floor are excluded.
- `zero_crossing_freq(x, fs)` — model-free cross-check: half period
  between the zero crossings that bracket the peak.
- `spectrogram`, `envelope`, `instant_phase` — the original EMD-based
  HHT spectrogram utilities.

## Example

```python
import numpy as np
from hht_tools import decompose_ceemdan, if_at_peak, zero_crossing_freq

fs = 100.0
t = np.arange(0, 40, 1 / fs)
x = ...  # your record

imfs = decompose_ceemdan(x, seed=0, trials=200, epsilon=0.15)
res = if_at_peak(imfs, x, fs, t)
print(res["f_dom"], res["f_mean"], res["f_spread"])
print(zero_crossing_freq(x, fs))  # model-free check
```

## Tests

```
pytest tests/
```
