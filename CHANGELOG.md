# Changelog

## 0.3.0 — 2026-11-05

- Preserve the default decomposition outputs and full-stack peak-frequency
  measurement method from 0.2.0: decomposition settings, seed application,
  phase-gradient estimator, amplitude weighting, thresholds, and edge flags.
- Add `return_residual=True` to both decomposition wrappers, returning
  oscillatory IMFs and the residual separately. This is opt-in because
  excluding a residual can change frequency summaries. EMD uses the upstream
  residual accessor so a true final IMF is never accidentally discarded.
- Reject nonfinite/misshaped signals, inconsistent time vectors, invalid
  rates and parameters, unseeded CEEMDAN, and constant CEEMDAN signals.
- Count only sign-changing zero crossings. Exact-zero runs contribute one
  midpoint crossing when signs differ across the run; zero touches and
  leading/trailing zeros are excluded. Signals without exact zeros retain
  the original sample-resolved crossing positions.
- Correct `spectrogram`: honor `Nimfs`, exclude the residual, use Hilbert
  envelope weights, exclude nonpositive IF, return the requested numbers of
  bins with center coordinates, and return full-length IMFs. This changes
  spectrogram values, coordinates, and shapes relative to 0.2.0; it does not
  change the separate peak-frequency estimator.
- Add input, residual, zero-crossing, and spectrogram regression tests;
  automated Python-version checks; citation metadata; modern package
  metadata; and a runnable synthetic example. Clear stale notebook output.
- Remove `hht_tools.txt`, an obsolete duplicate of the original 2022 module.
- Replace the legacy demo with a Markdown-guided synthetic-then-Napa
  walkthrough: fresh seeded CEEMDAN, Fourier spectrograms, Morlet
  scalograms, Hilbert trajectories, and peak-frequency comparisons.
  Bundle the processed Napa example with provenance and provide an
  optional ObsPy download/response-removal recipe. Demo dependencies
  require SciPy >= 1.12 for `ShortTimeFFT`.

### Preparing the release

1. Run tests and confirm the CI Python-version checks pass. Build and inspect
   the source distribution and wheel (`python -m pip install build`, then
   `python -m build`). No publishing step is automated.
2. Confirm version 0.3.0 in `pyproject.toml` and `CITATION.cff`, set the actual
   release date here and in `CITATION.cff`, and review the tagged contents.
3. Enable the repository in Zenodo before publishing the matching GitHub
   release. Use the release-specific DOI in citations; add it to the README
   and citation metadata when available. Do not insert a placeholder DOI.

## 0.2.0

- Installable package with seeded EMD/CEEMDAN wrappers, IF-at-peak triplets,
  zero-crossing checks, the original EMD spectrogram, tests, and MIT license.
